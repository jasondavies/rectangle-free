// Research-only: exact family grouping and a sampled production-layout tile model.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <nauty/nauty.h>

#include "cut_reference_model.hpp"
extern "C" uint64_t shared_column_production_key(uint64_t);

namespace {
struct CoreRef {
    uint64_t key=0;
    unsigned extension=0;
    std::array<unsigned,8> rows{};
    std::array<unsigned,8> columns{};
};
static CoreRef canonical_bipartite(uint64_t key,unsigned width) {
    const unsigned n=8+width,m=SETWORDSNEEDED(16);
    graph input[16*SETWORDSNEEDED(16)]{},output[16*SETWORDSNEEDED(16)]{};
    int lab[16],ptn[16],orbits[16];
    for(unsigned i=0;i<n;++i){lab[i]=int(i);ptn[i]=1;}
    ptn[7]=ptn[n-1]=0;
    for(unsigned r=0;r<8;++r)for(unsigned c=0;c<width;++c)
        if(key>>((7-r)*8+c)&1)ADDONEEDGE(input,r,8+c,m);
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon=TRUE;options.defaultptn=FALSE;
    statsblk stats{};
    densenauty(input,lab,ptn,orbits,&options,&stats,m,n,output);
    if(stats.errstatus)throw std::runtime_error("nauty core canonicalisation failed");
    CoreRef ref;
    for(unsigned r=0;r<8;++r) {
        if(lab[r]<0||lab[r]>=8)throw std::runtime_error("row partition changed");
        ref.rows[r]=unsigned(lab[r]);unsigned pattern=0;
        for(unsigned c=0;c<width;++c)if(ISELEMENT(GRAPHROW(output,r,m),8+c))pattern|=1u<<c;
        ref.key=(ref.key<<width)|pattern;
        ref.extension|=unsigned(key>>((7-unsigned(lab[r]))*8+7)&1)<<r;
    }
    for(unsigned c=0;c<width;++c) {
        if(lab[8+c]<8||lab[8+c]>=int(n))throw std::runtime_error("column partition changed");
        ref.columns[c]=unsigned(lab[8+c]-8);
    }
    return ref;
}
static CoreRef canonical_core(uint64_t key){return canonical_bipartite(key,7);}
static uint64_t move_last(uint64_t key,unsigned column) {
    uint64_t result=0;
    for(unsigned r=0;r<8;++r) {
        unsigned row=unsigned(key>>((7-r)*8))&255;
        unsigned next=(row&((1u<<column)-1))|((row>>(column+1))<<column)|(((row>>column)&1)<<7);
        result|=uint64_t(next)<<((7-r)*8);
    }
    return result;
}
static uint64_t leaf_key(uint64_t key) {
    if(__builtin_popcountll(key)>32)key=~key;
    uint64_t best=std::min(canonical_bipartite(key,8).key,canonical_bipartite(transpose(key),8).key);
    // Production solve_representative() also identifies balanced complements.
    if(__builtin_popcountll(key)==32)
        best=std::min(best,std::min(canonical_bipartite(~key,8).key,canonical_bipartite(transpose(~key),8).key));
    return best;
}
static CoreRef deletion_parent(uint64_t key) {
    if(__builtin_popcountll(key)>32)key=~key;
    CoreRef best;best.key=UINT64_MAX;
    unsigned orientations=__builtin_popcountll(key)==32?4:2;
    for(unsigned orientation=0;orientation<orientations;++orientation) {
        uint64_t base=orientation/2?~key:key;
        uint64_t k=orientation%2?transpose(base):base;
        for(unsigned c=0;c<8;++c) {
            auto candidate=canonical_core(move_last(k,c));
            if(candidate.key<best.key)best=candidate;
        }
    }
    return best;
}
static uint64_t reconstructed(const CoreRef& ref) {
    uint64_t key=0;
    for(unsigned r=0;r<8;++r) {
        unsigned row=unsigned(ref.key>>((7-r)*7))&127;
        row|=((ref.extension>>r)&1)<<7;
        key|=uint64_t(row)<<((7-r)*8);
    }
    return key;
}
static std::map<uint64_t,unsigned> assigned_children(uint64_t parent) {
    std::map<uint64_t,unsigned> leaves;
    for(unsigned extension=0;extension<256;++extension) {
        CoreRef ref;ref.key=parent;ref.extension=extension;
        uint64_t child=reconstructed(ref);
        // Match the retained production population, including its existing
        // balanced-complement quotient via leaf_key()/deletion_parent().
        if(__builtin_popcountll(child)>32)continue;
        if(deletion_parent(child).key!=parent)continue;
        leaves.emplace(leaf_key(child),extension);
    }
    return leaves;
}
struct Group {
    uint64_t records=0;
    std::array<uint64_t,4> extensions{};
    void add(unsigned a){++records;extensions[a/64]|=uint64_t(1)<<(a%64);}
    unsigned demand()const {unsigned n=0;for(auto x:extensions)n+=__builtin_popcountll(x);return n;}
};
using Groups=std::unordered_map<uint64_t,Group>;
static unsigned band(unsigned n) {return n>=20?3:n>=8?2:n>=2?1:0;}
static void print_groups(const char* mode,const Groups& groups) {
    std::array<uint64_t,4> records{};std::map<unsigned,uint64_t> histogram;
    uint64_t queries=0;
    for(const auto& [key,g]:groups) {unsigned n=g.demand();
        records[band(n)]+=g.records;++histogram[n];queries+=n;}
    std::cout<<"{\"type\":\"groups\",\"mode\":\""<<mode<<"\",\"families\":"<<groups.size()
        <<",\"distinct_queries\":"<<queries<<",\"band_records\":[";
    for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<records[i];}
    std::cout<<"],\"demand_histogram\":{";
    bool first=true;for(auto [k,v]:histogram){if(!first)std::cout<<',';first=false;std::cout<<'"'<<k<<"\":"<<v;}
    std::cout<<"}}"<<std::endl;
}
struct File {
    std::string path;uint64_t count=0,used=0;
    explicit File(const std::string& p,uint64_t limit):path(p) {
        std::ifstream in(path,std::ios::binary);char magic[8];uint32_t width;
        in.read(magic,8);in.read(reinterpret_cast<char*>(&width),4);in.read(reinterpret_cast<char*>(&count),8);
        if(!in||width!=8||(std::memcmp(magic,"R8ORB01",7)&&std::memcmp(magic,"R8SQT01",7)))
            throw std::runtime_error("invalid 8x8 orbit header: "+path);
        in.seekg(0,std::ios::end);
        if(U128(in.tellg())!=20+U128(16)*count)throw std::runtime_error("orbit file length: "+path);
        used=std::min(limit,count);
    }
};
struct Sample {size_t file;uint64_t index,key=0,stratum=0,weight=0;};
static void self_test() {
    nauty_check(WORDSIZE,SETWORDSNEEDED(15),15,NAUTYVERSIONID);
    for(unsigned t=0;t<100;++t) {
        uint64_t key=t<2?(t?~uint64_t(0):0):mix64(t);
        uint64_t production=shared_column_production_key(key);
        if(leaf_key(production)!=leaf_key(key)||shared_column_production_key(production)!=production)
            throw std::runtime_error("production representative mismatch");
        CoreRef ref=canonical_core(key);
        uint64_t image=0;
        for(unsigned r=0;r<8;++r) {
            unsigned pattern=unsigned(key>>((7-ref.rows[r])*8))&255,newrow=pattern&128;
            for(unsigned c=0;c<7;++c)newrow|=((pattern>>ref.columns[c])&1)<<c;
            image|=uint64_t(newrow)<<((7-r)*8);
        }
        if(image!=reconstructed(ref))throw std::runtime_error("canonical extension map mismatch");
        std::array<unsigned,8> rp{0,1,2,3,4,5,6,7};
        std::array<unsigned,7> cp{0,1,2,3,4,5,6};
        for(unsigned i=0;i<8;++i)std::swap(rp[i],rp[mix64(t*31+i)%8]);
        for(unsigned i=0;i<7;++i)std::swap(cp[i],cp[mix64(t*47+i)%7]);
        uint64_t permuted=0;
        for(unsigned r=0;r<8;++r) {
            unsigned pattern=unsigned(key>>((7-rp[r])*8))&255,newrow=pattern&128;
            for(unsigned c=0;c<7;++c)newrow|=((pattern>>cp[c])&1)<<c;
            permuted|=uint64_t(newrow)<<((7-r)*8);
        }
        if(canonical_core(permuted).key!=ref.key)throw std::runtime_error("core isomorphism mismatch");
        if(t<12) {
            uint64_t k=__builtin_popcountll(key)>32?~key:key;
            auto parent=deletion_parent(k);
            if(deletion_parent(transpose(k)).key!=parent.key||deletion_parent(leaf_key(k)).key!=parent.key
               ||deletion_parent(~k).key!=parent.key)
                throw std::runtime_error("deletion parent invariance");
            auto children=assigned_children(parent.key);
            if(!children.count(leaf_key(k)))throw std::runtime_error("parent cover missed input orbit");
            for(auto [leaf,extension]:children) {
                CoreRef child;child.key=parent.key;child.extension=extension;
                if(leaf_key(reconstructed(child))!=leaf)throw std::runtime_error("child query reconstruction");
            }
            // Include permutations moving the deleted column itself.
            uint64_t moved=move_last(k,t%8);
            if(deletion_parent(moved).key!=parent.key||leaf_key(moved)!=leaf_key(k))
                throw std::runtime_error("whole-column parent invariance");
        }
    }
    // Check production canonical row-map/orientation against a complete direct
    // half distribution, not against a second copy of the canonicalizer.
    for(uint64_t raw:{0ull,0xffffffffull,0x01234567ull,0x12344321ull,0x5555aaaaull}) {
        auto form=canonical_prefix(raw,4);
        auto q=quotient_token_planes(build_distribution(form.key,4,false));
        std::map<uint64_t,uint64_t> expanded,expected;
        for(auto e:q.entries){auto m=transform_pair_mask(e.mask,form.row_map);expanded[m]+=e.weight;
            if(m!=swap_token_planes(m))expanded[swap_token_planes(m)]+=e.weight;}
        for(auto e:build_distribution(raw,4,false).entries)expected[e.mask]=e.weight;
        if(expanded!=expected)throw std::runtime_error("production half row map mismatch");
    }
    CostModel model;
    if(model.cost(0)!=model.cost(~uint64_t(0)))throw std::runtime_error("complement tile-model symmetry");
    if(assigned_children(0).size()!=9)throw std::runtime_error("empty parent must have nine child orbits");
    const uint64_t balanced=0x55555555aaaaaaaaull;
    if(leaf_key(balanced)!=leaf_key(~balanced)||deletion_parent(balanced).key!=deletion_parent(~balanced).key)
        throw std::runtime_error("balanced complement identity");
    std::cout<<"SHARED_COLUMN_FAMILY_TEST exact=OK"<<std::endl;
}

static void run(const std::vector<File>& files,unsigned sample_count,uint64_t seed) {
    Groups simple,canonical;std::vector<Sample> samples;
    uint64_t total=0;double started=response::now();
    for(size_t f=0;f<files.size();++f) {
        const auto& file=files[f];total+=file.used;
        std::vector<Sample> selected;
        unsigned n=unsigned(std::min<uint64_t>(sample_count,file.used));
        for(unsigned i=0;i<n;++i) {
            uint64_t begin=U128(i)*file.used/n,end=U128(i+1)*file.used/n;
            selected.push_back({f,begin+mix64(seed^mix64(f*1000000+i))%(end-begin),0,i,end-begin});
        }
        size_t next=0;std::ifstream in(file.path,std::ios::binary);in.seekg(20);
        for(uint64_t i=0;i<file.used;++i) {
            OrbitRecord rec{};in.read(reinterpret_cast<char*>(&rec),16);
            if(!in||!rec.weight)throw std::runtime_error("invalid record");
            if(__builtin_popcountll(rec.key)>32)throw std::runtime_error("expected retained <=32-cell corpus");
            auto row=response::family_key(rec.key);simple[row.first].add(row.second);
            auto full=canonical_core(rec.key);canonical[full.key].add(full.extension);
            if(next<selected.size()&&i==selected[next].index){selected[next].key=rec.key;++next;}
            if(i && i%1000000==0)std::cerr<<"census file="<<f<<" records="<<i<<" seconds="<<response::now()-started<<std::endl;
        }
        samples.insert(samples.end(),selected.begin(),selected.end());
    }
    std::cout<<"{\"type\":\"census\",\"records\":"<<total<<",\"files\":"<<files.size()
        <<",\"seconds\":"<<response::now()-started<<",\"seed\":"<<seed
        <<",\"sample_count\":"<<samples.size()<<"}"<<std::endl;
    print_groups("row",simple);print_groups("row_column",canonical);
    CostModel model;
    std::array<long double,4> row_work{},canon_work{};
    for(size_t i=0;i<samples.size();++i) {
        const auto& s=samples[i];auto row=response::family_key(s.key);auto full=canonical_core(s.key);
        unsigned nr=simple.at(row.first).demand(),nc=canonical.at(full.key).demand();
        uint64_t tiles=model.cost(s.key);
        row_work[band(nr)]+=static_cast<long double>(tiles)*s.weight;
        canon_work[band(nc)]+=static_cast<long double>(tiles)*s.weight;
        std::cout<<"{\"type\":\"sample\",\"file\":"<<s.file<<",\"index\":"<<s.index
            <<",\"key\":\""<<s.key<<"\",\"row_core\":\""<<row.first<<"\",\"canonical_core\":\""<<full.key
            <<"\",\"extension\":"<<full.extension<<",\"row_demand\":"<<nr<<",\"canonical_demand\":"<<nc
            <<",\"stratum\":"<<s.stratum<<",\"stratum_records\":"<<s.weight<<",\"tiles\":"<<tiles<<"}"<<std::endl;
        if(i%32==0)std::cerr<<"cost samples="<<i<<"/"<<samples.size()<<" seconds="<<response::now()-started<<std::endl;
    }
    std::cout<<"{\"type\":\"complete\",\"seconds\":"<<response::now()-started
        <<",\"canonical_builds\":"<<model.builds<<",\"canonical_hits\":"<<model.hits<<",\"weighted_row_tiles\":[";
    for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<row_work[i];}
    std::cout<<"],\"weighted_canonical_tiles\":[";
    for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<canon_work[i];}
    std::cout<<"]}"<<std::endl;
}

// Input is a TSV projection of a completed workload sample: file, index, key,
// stratum population, tile cost. Never recompute costs in a new row gauge.
// Time exactly the assigned children, not all 256 possible extensions. Both CPU
// methods use the parent gauge; this is NOT a warmed production GPU comparison.
static void benchmark_parent(uint64_t parent,size_t cache,uint64_t work,double seconds) {
    if(parent>>56||!cache||cache>1000000||!work||work>100000000000ull||
       !std::isfinite(seconds)||seconds<=0||seconds>300)
        throw std::runtime_error("invalid parent benchmark limits");
    const double started=response::now();
    auto children=assigned_children(parent);
    const double assignment_seconds=response::now()-started;
    if(children.empty())throw std::runtime_error("parent has no assigned children");
    std::vector<unsigned> demand;
    for(auto [leaf,query]:children)demand.push_back(query);
    std::sort(demand.begin(),demand.end());
    std::cout<<"{\"type\":\"family_begin\",\"parent\":\""<<parent<<"\",\"fanout\":"<<children.size()
        <<",\"assignment_seconds\":"<<assignment_seconds<<",\"cache_cap\":"<<cache
        <<",\"work_cap\":"<<work<<",\"seconds_cap_per_method_side\":"<<seconds<<"}"<<std::endl;
    response::Engine engine(8);
    auto core=response::columns(8,7,parent);
    unsigned checked=0,capped=0;
    for(unsigned side=0;side<2;++side) {
        auto c=side?response::complement(core,8):core;
        auto queries=demand;if(side)for(auto& a:queries)a^=255;
        auto reference=response::independent(engine,c,queries,response::Budget(work,seconds));
        response::print_result("independent",side,reference);
        capped+=reference.status!="complete";
        for(unsigned variant=0;variant<3;++variant) {
            auto result=variant==2?response::independent(engine,c,queries,response::Budget(work,seconds)):
                response::shared(engine,c,queries,variant?0:cache,response::Budget(work,seconds));
            const char* method=variant==2?"independent_repeat":variant?"streamed":"cached";
            response::print_result(method,side,result);
            capped+=result.status!="complete";
            if(reference.status=="complete"&&result.status=="complete") {
                for(unsigned a:queries)if(reference.values[a]!=result.values[a])
                    throw std::runtime_error("ASSIGNED FAMILY COUNT MISMATCH");
                checked+=queries.size();
                std::cout<<"{\"type\":\"family_parity\",\"method\":\""<<method
                    <<"\",\"side\":"<<side<<",\"answers\":"<<queries.size()<<"}"<<std::endl;
            }
        }
    }
    // Production canonical half representatives, but the reconstructed parent
    // row/column gauge, NOT the original corpus's lexicographic representative.
    const double model_started=response::now();
    CostModel model;uint64_t tiles=0;
    for(auto [leaf,query]:children) {
        CoreRef ref;ref.key=parent;ref.extension=query;
        tiles+=model.cost(reconstructed(ref));
    }
    std::cout<<"{\"type\":\"family_complete\",\"parent\":\""<<parent<<"\",\"fanout\":"<<children.size()
        <<",\"checked_answers\":"<<checked<<",\"capped_method_sides\":"<<capped
        <<",\"production_builder_parent_gauge_tiles\":"<<tiles
        <<",\"production_model_seconds\":"<<response::now()-model_started
        <<",\"wall_seconds\":"<<response::now()-started<<"}"<<std::endl;
}
static void model_parent(uint64_t parent) {
    if(parent>>56)throw std::runtime_error("invalid parent key");
    auto children=assigned_children(parent);
    if(children.empty())throw std::runtime_error("parent has no assigned children");
    CostModel model;
    uint64_t production_tiles=0,parent_tiles=0;
    double started=response::now();
    for(auto [leaf,query]:children) {
        CoreRef ref;ref.key=parent;ref.extension=query;
        uint64_t raw=reconstructed(ref),production=shared_column_production_key(raw);
        if(leaf_key(production)!=leaf||shared_column_production_key(production)!=production)
            throw std::runtime_error("production representative changed orbit/idempotence");
        uint64_t p=model.cost(production),g=model.cost(raw);
        production_tiles+=p;parent_tiles+=g;
        std::cout<<"{\"type\":\"child_model\",\"query\":"<<query<<",\"production_key\":\""<<production
            <<"\",\"production_tiles\":"<<p<<",\"parent_gauge_tiles\":"<<g<<"}"<<std::endl;
    }
    std::cout<<"{\"type\":\"parent_model_complete\",\"parent\":\""<<parent<<"\",\"fanout\":"<<children.size()
        <<",\"production_tiles\":"<<production_tiles<<",\"parent_gauge_tiles\":"<<parent_tiles
        <<",\"seconds\":"<<response::now()-started<<"}"<<std::endl;
}
static void check_production_keys(const std::string& path) {
    std::ifstream input(path);std::string line;unsigned count=0;
    while(std::getline(input,line)) {
        std::istringstream fields(line);uint64_t file,index,key,weight,tiles;std::string extra;
        if(!(fields>>file>>index>>key>>weight>>tiles)||fields>>extra||!weight||++count>8192)
            throw std::runtime_error("invalid production key TSV");
        if(shared_column_production_key(key)!=key)
            throw std::runtime_error("source key differs from production representative");
    }
    if(!input.eof()||!count)throw std::runtime_error("empty/invalid production key TSV");
    std::cout<<"{\"type\":\"production_keys_checked\",\"keys\":"<<count<<"}"<<std::endl;
}
static void parent_census(const std::string& path,unsigned limit,uint64_t seed) {
    struct Row {uint64_t file,index,key,weight,tiles;};
    std::ifstream in(path);std::vector<Row> rows;std::string line;
    while(std::getline(in,line)) {
        Row row;std::string extra;std::istringstream fields(line);
        if(!(fields>>row.file>>row.index>>row.key>>row.weight>>row.tiles)||fields>>extra||!row.weight)
            throw std::runtime_error("invalid parent sample TSV row");
        rows.push_back(row);
        if(rows.size()>8192)throw std::runtime_error("parent sample too large");
    }
    if(!in.eof()||rows.empty())throw std::runtime_error("invalid parent sample TSV");
    unsigned n=std::min<size_t>(limit,rows.size());
    if(!n||n>4096)throw std::runtime_error("invalid parent sample size");
    std::unordered_map<uint64_t,std::map<uint64_t,unsigned>> cache;
    std::array<long double,4> work{},population{};
    double started=response::now();
    for(unsigned i=0;i<n;++i) {
        size_t begin=uint64_t(i)*rows.size()/n,end=uint64_t(i+1)*rows.size()/n;
        const auto& s=rows[begin+mix64(seed^i)%(end-begin)];
        if(__builtin_popcountll(s.key)>32)throw std::runtime_error("parent input not retained");
        auto parent=deletion_parent(s.key);
        auto it=cache.find(parent.key);
        if(it==cache.end())it=cache.emplace(parent.key,assigned_children(parent.key)).first;
        auto leaf=it->second.find(leaf_key(s.key));
        if(leaf==it->second.end())throw std::runtime_error("assigned family missing sampled orbit");
        unsigned fanout=it->second.size();uint64_t weight=s.weight*(end-begin);
        work[band(fanout)]+=static_cast<long double>(weight)*s.tiles;population[band(fanout)]+=weight;
        std::cout<<"{\"type\":\"parent_sample\",\"file\":"<<s.file<<",\"index\":"<<s.index
            <<",\"key\":\""<<s.key<<"\",\"parent\":\""<<parent.key<<"\",\"query\":"<<leaf->second
            <<",\"fanout\":"<<fanout<<",\"population_weight\":"<<weight<<",\"tiles\":"<<s.tiles<<"}"<<std::endl;
        if(i%32==0)std::cerr<<"parent samples="<<i<<"/"<<n<<" seconds="<<response::now()-started<<std::endl;
    }
    std::cout<<"{\"type\":\"parent_complete\",\"samples\":"<<n<<",\"parents\":"<<cache.size()
        <<",\"seconds\":"<<response::now()-started<<",\"seed\":"<<seed<<",\"weighted_tiles\":[";
    for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<work[i];}
    std::cout<<"],\"weighted_records\":[";
    for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<population[i];}
    std::cout<<"]}"<<std::endl;
}
} // namespace
int main(int argc,char** argv) try {
    nauty_check(WORDSIZE,SETWORDSNEEDED(16),16,NAUTYVERSIONID);
    initialise_tables();
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc==3&&std::string(argv[1])=="model-parent") {
        model_parent(std::stoull(argv[2]));return 0;
    }
    if(argc==3&&std::string(argv[1])=="check-production-keys") {
        check_production_keys(argv[2]);return 0;
    }
    if(argc==6&&std::string(argv[1])=="family") {
        benchmark_parent(std::stoull(argv[2]),std::stoull(argv[3]),std::stoull(argv[4]),std::stod(argv[5]));return 0;
    }
    if(argc==5&&std::string(argv[1])=="parents") {
        parent_census(argv[2],std::stoul(argv[3]),std::stoull(argv[4]));return 0;
    }
    if(argc!=5)throw std::runtime_error("usage: SHARD[,SHARD...] MAX_RECORDS_PER_SHARD SAMPLES_PER_SHARD SEED | parents SAMPLE_TSV LIMIT SEED | family PARENT CACHE_ENTRIES WORK_CAP SECONDS | model-parent PARENT | check-production-keys SAMPLE_TSV | --self-test");
    uint64_t limit=std::stoull(argv[2]);unsigned samples=std::stoul(argv[3]);
    if(!limit||limit>10000000||!samples||samples>2048)throw std::runtime_error("invalid limits");
    std::stringstream ss(argv[1]);std::string path;std::vector<File> files;std::set<std::string> unique;
    while(std::getline(ss,path,',')){
        if(files.size()==4)throw std::runtime_error("at most four files");
        if(!unique.insert(path).second)throw std::runtime_error("duplicate file");
        files.emplace_back(path,limit);
    }
    if(files.empty())throw std::runtime_error("no files");
    run(files,samples,std::stoull(argv[4]));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
