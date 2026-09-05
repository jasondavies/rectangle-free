// Instrumented CPU census only. Validate regrouping exactly, but never substitute
// its contribution into the production result. Timings exclude validation solves.
#include "../../src/partition/partition_poly_internal.h"
#include "partition_residual_census.h"

typedef struct {
    int used;
    uint64_t hash;
    Graph graph;
    Poly weight;
} Entry;
static Entry* tables[2];
static size_t capacity;
static uint64_t flushes, sampled, inputs, unique[2], constants;
static int active;
static Poly expected;
static double normalized_seconds, key_seconds;
static long long stride, max_samples;
static RowGraphCache validation_cache;

static void report(void) {
    printf("RESIDUAL_CENSUS flushes=%llu sampled=%llu inputs=%llu normalized=%llu wl_keyed=%llu empty=%llu normalize_seconds=%.6f key_seconds=%.6f exact=OK\n",
           (unsigned long long)flushes,(unsigned long long)sampled,
           (unsigned long long)inputs,(unsigned long long)unique[0],
           (unsigned long long)unique[1],(unsigned long long)constants,
           normalized_seconds,key_seconds);
    free(tables[0]); free(tables[1]);
    free(validation_cache.keys); free(validation_cache.stamps);
    free(validation_cache.rows); free(validation_cache.coeffs);
    free(validation_cache.x_pows); free(validation_cache.degs);
}
static long long option(const char* name, long long fallback) {
    const char* value=getenv(name);
    long long result=value?parse_ll_or_die(value,name):fallback;
    if(result<=0){fprintf(stderr,"positive census limits required\n");exit(1);}
    return result;
}
void residual_census_begin(void) {
    if(!capacity) {
        if(omp_get_max_threads()!=1){fprintf(stderr,"census requires OMP_NUM_THREADS=1\n");exit(1);}
        const char* hard_bits=getenv("RECT_HARD_CACHE_BITS");
        if(!hard_bits || strcmp(hard_bits,"0")) {
            fprintf(stderr,"census requires RECT_HARD_CACHE_BITS=0 to isolate validation solves\n");exit(1);
        }
        stride=option("RECT_CENSUS_STRIDE",32);
        max_samples=option("RECT_CENSUS_FLUSHES",64);
        validation_cache.mask=63; validation_cache.probe=8;
        validation_cache.poly_len=MAX_GRAPH_VERTICES+1;
        validation_cache.keys=checked_calloc(64,sizeof(*validation_cache.keys),"validation keys");
        validation_cache.stamps=checked_calloc(64,sizeof(*validation_cache.stamps),"validation stamps");
        validation_cache.rows=checked_calloc(64*MAX_GRAPH_VERTICES,sizeof(*validation_cache.rows),"validation rows");
        validation_cache.coeffs=checked_calloc(64*validation_cache.poly_len,sizeof(*validation_cache.coeffs),"validation coefficients");
        validation_cache.x_pows=checked_calloc(64,1,"validation powers");
        validation_cache.degs=checked_calloc(64,1,"validation degrees");
        capacity=(size_t)1<<(g_terminal_aggregate_bits+1);
        for(int i=0;i<2;i++)tables[i]=checked_calloc(capacity,sizeof(Entry),"residual census");
        atexit(report);
    }
    active=(flushes++%(uint64_t)stride==0 && sampled<(uint64_t)max_samples);
    if(!active)return;
    sampled++;
    poly_zero(&expected);
    for(int i=0;i<2;i++)memset(tables[i],0,capacity*sizeof(Entry));
}
static void insert(int table,const Graph* graph,const Poly* weight) {
    uint64_t hash=hash_graph(graph);
    size_t slot=(size_t)hash&(capacity-1);
    while(tables[table][slot].used) {
        Entry* e=&tables[table][slot];
        if(e->hash==hash && e->graph.n==graph->n &&
           !memcmp(e->graph.adj,graph->adj,graph->n*sizeof(AdjWord))) {
            poly_accumulate_checked(&e->weight,weight);return;
        }
        slot=(slot+1)&(capacity-1);
    }
    Entry* e=&tables[table][slot];
    e->used=1;e->hash=hash;e->graph=*graph;e->weight=*weight;
    unique[table]++;
}
void residual_census_visit(const Graph* graph,const Poly* weight,const Poly* contribution,
                           GraphCanonWorkspace* ws) {
    if(!active)return;
    inputs++;
    poly_accumulate_checked(&expected,contribution);
    double t=omp_get_wtime();
    Graph reduced=*graph, dense;
    GraphPoly factor;
    Poly combined;
    graph_poly_one_ref(&factor);
    simplify_graph_poly_multiplier(&reduced,&factor);
    dense.n=(uint8_t)graph_build_dense_rows(&reduced,dense.adj);
    dense.vertex_mask=graph_row_mask(dense.n);
    poly_mul_graph_ref(weight,&factor,&combined);
    if(!dense.n)constants++;
    insert(0,&dense,&combined);
    normalized_seconds+=omp_get_wtime()-t;
    t=omp_get_wtime();
    Graph keyed;
    get_canonical_graph(&dense,&keyed,ws,NULL);
    insert(1,&keyed,&combined);
    key_seconds+=omp_get_wtime()-t;
}
void residual_census_end(RowGraphCache* cache,RowGraphCache* raw_cache,GraphCanonWorkspace* ws) {
    (void)cache; (void)raw_cache;
    if(!active)return;
    for(int table=0;table<2;table++) {
        Poly total;poly_zero(&total);
        long long canon=0,hits=0,raw_hits=0;
        for(size_t i=0;i<capacity;i++)if(tables[table][i].used) {
            Entry* e=&tables[table][i];
            GraphResult result;
            Poly contribution;
            // Do not charge these independent checks to adaptive work budgets.
            long long* saved=tls_adaptive_work_counter;
            tls_adaptive_work_counter=NULL;
            GraphHardStats* saved_stats=tls_hard_graph_stats;
            tls_hard_graph_stats=NULL;
            solve_graph_poly(&e->graph,&validation_cache,&validation_cache,ws,&canon,&hits,&raw_hits,NULL,&result);
            tls_hard_graph_stats=saved_stats;
            tls_adaptive_work_counter=saved;
            poly_mul_graph_ref(&e->weight,&result,&contribution);
            poly_accumulate_checked(&total,&contribution);
        }
        if(total.deg!=expected.deg || memcmp(total.coeffs,expected.coeffs,
                                            (total.deg+1)*sizeof(PolyCoeff))) {
            fprintf(stderr,"residual aggregation mismatch\n");abort();
        }
    }
}
