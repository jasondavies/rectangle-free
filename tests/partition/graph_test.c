#include "../../src/partition/partition_poly_internal.h"
#include "../../src/common/sha256_c.h"
#include <assert.h>

static RowGraphCache new_cache(void) {
    RowGraphCache c={0};
    c.mask=63;c.probe=8;c.poly_len=MAX_GRAPH_VERTICES+1;
    c.keys=calloc(64,sizeof(*c.keys));c.stamps=calloc(64,sizeof(*c.stamps));
    c.rows=calloc(64*MAX_GRAPH_VERTICES,sizeof(*c.rows));
    c.coeffs=calloc(64*c.poly_len,sizeof(*c.coeffs));
    c.x_pows=calloc(64,1);c.degs=calloc(64,1);
    assert(c.keys&&c.stamps&&c.rows&&c.coeffs&&c.x_pows&&c.degs);
    return c;
}
static void free_cache(RowGraphCache* c) {
    free(c->keys);free(c->stamps);free(c->rows);free(c->coeffs);
    free(c->x_pows);free(c->degs);
}
static int same(const GraphPoly* a,const GraphPoly* b) {
    return a->x_pow==b->x_pow && a->deg==b->deg &&
           !memcmp(a->coeffs,b->coeffs,(a->deg+1)*sizeof(PolyCoeff));
}
static void edge(Graph* g,int a,int b) {g->adj[a]|=(AdjWord)1<<b;g->adj[b]|=(AdjWord)1<<a;}
static void solve(const Graph* g, RowGraphCache* c, RowGraphCache* raw,GraphPoly* result) {
    GraphCanonWorkspace ws={0};long long ca=0,hi=0,rh=0;
    solve_graph_poly(g,c,raw,&ws,&ca,&hi,&rh,NULL,result);
}
int main(int argc,char** argv) {
    if(argc==2 && !strcmp(argv[1],"--overflow")) {
        Poly a,b,out;poly_one_ref(&a);poly_one_ref(&b);
        a.coeffs[0]=(PolyCoeff)1<<126;b.coeffs[0]=4;
        poly_mul_ref(&a,&b,&out);return 0;
    }
    for(unsigned n=0;n<140;n++) {
        RectSha256 h;char hex[65];rect_sha256_init(&h);
        for(unsigned i=0;i<n;i++){unsigned char x=(unsigned char)i;rect_sha256_update(&h,&x,1);}
        rect_sha256_finish(&h,hex);printf("SHA %u %s\n",n,hex);
    }
    RowGraphCache c=new_cache(),raw=new_cache();
    // Collision-heavy inserts, replacement, overwrite, and both interfaces.
    for(int i=0;i<200;i++) {
        Graph g={.n=2,.vertex_mask=3};g.adj[0]=(AdjWord)(i%17);g.adj[1]=(AdjWord)(i%7);
        GraphPoly p,got;graph_poly_one_ref(&p);p.coeffs[0]=i+1;
        if(i&1)store_row_graph_cache_entry(&c,0,2,&g,(AdjWord)ADJWORD_MASK,&p);
        else store_row_graph_cache_entry_rows(&c,0,2,g.adj,&p);
        assert(row_graph_cache_lookup_rows(&c,0,2,g.adj,&got,1)&&same(&p,&got));
        assert(row_graph_cache_lookup_poly(&c,0,2,&g,(AdjWord)ADJWORD_MASK,&got,1)&&same(&p,&got));
    }
    Graph masked={.n=2,.vertex_mask=3};
    masked.adj[0]=258;masked.adj[1]=513;
    AdjWord clean_rows[2]={2,1};
    GraphPoly masked_value,masked_result;graph_poly_one_ref(&masked_value);
    masked_value.coeffs[0]=37;
    store_row_graph_cache_entry(&c,0,2,&masked,3,&masked_value);
    assert(row_graph_cache_lookup_rows(&c,0,2,clean_rows,&masked_result,1));
    assert(same(&masked_value,&masked_result));
    free_cache(&c);free_cache(&raw);c=new_cache();raw=new_cache();
    small_graph_lookup_init();
    // Independent treewidth polynomials versus deletion/addition-contraction,
    // including permutations and non-dense vertex masks.
    uint32_t seed=1234567;
    for(int sample=0;sample<48;sample++) {
        Graph g={.n=9,.vertex_mask=511};
        for(int i=0;i<9;i++)edge(&g,i,(i+1)%9);
        for(int i=0;i<9;i++)for(int j=i+2;j<9;j++) {
            seed=seed*1664525U+1013904223U;
            if((seed>>28)==0)edge(&g,i,j);
        }
        GraphPoly expected;
        assert(solve_graph_poly_treewidth(&g,6,&expected,NULL));
        for(int mode=0;mode<3;mode++) {
            free_cache(&c);free_cache(&raw);c=new_cache();raw=new_cache();
            g_addition_contraction_fill_limit=mode==0?0:2;
            g_treewidth_limit=mode==2?6:0;g_treewidth_min_n=8;
            GraphPoly got;solve(&g,&c,&raw,&got);assert(same(&got,&expected));
            Graph sparse={.n=9,.vertex_mask=0};
            for(int i=0;i<9;i++) {
                int pi=2*((i+3)%9);sparse.vertex_mask|=UINT64_C(1)<<pi;
                for(int j=0;j<9;j++)if(g.adj[i]&((AdjWord)1<<j))
                    sparse.adj[pi]|=(AdjWord)1<<(2*((j+3)%9));
            }
            solve(&sparse,&c,&raw,&got);assert(same(&got,&expected));
        }
    }
    free_cache(&c);free_cache(&raw);small_graph_lookup_free();
    puts("PARTITION_GRAPH_TEST exact=OK");return 0;
}
