// CPU-only structural census of the unchanged 8x8 production join.
// These are operation counts, not measured device instructions or timings.
#include "cut_histogram_model.hpp"
#include <cmath>

enum Feature {
    screened, compatible, forward, swapped, both, class_pairs,
    forward_calls, swapped_calls, dual_calls, reverse_calls,
    tiles, dual_tiles, useful_pairs, a_fragments, b_fragments,
    swapped_fragments, max_bucket_tiles, feature_count
};
static const char* names[] = {
    "screened_pairs", "compatible_pairs", "forward_pairs", "swapped_pairs",
    "both_pairs", "class_pairs", "forward_calls", "swapped_calls", "dual_calls",
    "reverse_calls", "tiles", "dual_tiles", "useful_pairs",
    "a_fragments", "b_fragments", "swapped_fragments", "max_bucket_tiles"
};
using Features=std::array<uint64_t,feature_count>;

static Features measure(const CutTileIndex& a,const CutTileIndex& b,
                        CutTileScorer& walker,bool indexed=true) {
    Features out{};
    out[screened]=uint64_t(a.buckets.size())*b.buckets.size();
    auto visit=[&](auto x,auto y,bool f,bool s) {
        ++out[compatible];out[forward]+=f;out[swapped]+=s;out[both]+=f&&s;
        uint64_t bucket_tiles=0;
        for(unsigned i=x.begin;i<x.end;++i)for(unsigned j=y.begin;j<y.end;++j) {
            ++out[class_pairs];auto l=a.classes[i],r=b.classes[j];
            bool swap=s&&r.orbit==2;
            unsigned orientations=unsigned(f)+unsigned(swap);
            if(!orientations)continue;
            bool reverse=r.n16*l.n8<l.n16*r.n8 ||
                (r.n16*l.n8==l.n16*r.n8&&r.count<l.count);
            uint64_t t=std::min(l.n16*r.n8,r.n16*l.n8);
            uint64_t outer=reverse?r.n16:l.n16;
            out[reverse_calls]+=reverse;
            if(f&&swap){++out[dual_calls];out[dual_tiles]+=2*t;}
            else if(f)++out[forward_calls];
            else ++out[swapped_calls];
            bucket_tiles+=t*orientations;
            out[useful_pairs]+=l.count*r.count*orientations;
            // FP4 dual shares source loads/ordinary packing, but still issues
            // TWO tensor operations per tile. Opposite packing is extra work.
            out[a_fragments]+=outer;out[b_fragments]+=t;
            if(swap)out[swapped_fragments]+=reverse?outer:t;
        }
        out[tiles]+=bucket_tiles;
        out[max_bucket_tiles]=std::max(out[max_bucket_tiles],bucket_tiles);
    };
    if(indexed)walker.visit_pairs(a,b,visit);
    else for(auto x:a.buckets)for(auto y:b.buckets) {
        bool f=!(x.prefix&y.prefix);
        bool s=!(x.prefix&(((y.prefix&127)<<7)|(y.prefix>>7)));
        if(f||s)visit(x,y,f,s);
    }
    return out;
}

static void self_test() {
    CutTileScorer walker;response::Engine engine(8);
    const uint64_t sizes[]={1,7,8,9,15,16,17,31,32,33};
    for(unsigned trial=0;trial<40;++trial) {
        response::Dist a,b;
        for(unsigned side=0;side<2;++side)for(unsigned prefix:{0,1,63,64,127,128,4095,16383}) {
            response::Bucket bucket{prefix,{},{}};
            for(unsigned c=0;c<2;++c)
                bucket.classes.push_back({sizes[(trial+prefix+c)%10],1+(trial+c+side)%2});
            (side?b:a).push_back(bucket);
        }
        CutTileIndex x(a),y(b);
        auto indexed=measure(x,y,walker),naive=measure(x,y,walker,false);
        response::Budget budget;engine.tile_model(a,b,budget);
        if(indexed!=naive||indexed[tiles]!=budget.modeled_tiles||
           indexed[useful_pairs]>128*indexed[tiles])
            throw std::runtime_error("feature/reference mismatch");
    }
    if(measure(CutTileIndex(),CutTileIndex(),walker)!=Features{})
        throw std::runtime_error("empty join mismatch");
    // One fixed-point class and one size-two class: 1 ordinary + 2 dual tiles.
    response::Dist a{{0,{},{{1,1}}}},b{{0,{},{{1,1},{1,2}}}};
    auto f=measure(CutTileIndex(a),CutTileIndex(b),walker);
    if(f[tiles]!=3||f[forward_calls]!=1||f[dual_calls]!=1||f[useful_pairs]!=3)
        throw std::runtime_error("orbit multiplicity feature mismatch");
    std::cout<<"JOIN_FEATURE_TEST exact=OK\n";
}

int main(int argc,char** argv) try {
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc!=3)throw std::runtime_error("usage: INPUT_ORBITS MAX_SECONDS | --self-test");
    double seconds=std::stod(argv[2]);
    if(!std::isfinite(seconds)||seconds<=0||seconds>600)throw std::runtime_error("invalid time cap");
    std::ifstream in(argv[1],std::ios::binary);char magic[8];uint32_t width;uint64_t count;
    in.read(magic,8);in.read(reinterpret_cast<char*>(&width),4);in.read(reinterpret_cast<char*>(&count),8);
    if(!in||std::memcmp(magic,"R8SQT01\0",8)||width!=8||!count||count>16384)
        throw std::runtime_error("invalid bounded input header");
    in.seekg(0,std::ios::end);
    if(in.tellg()!=std::streamoff(20+16*count))throw std::runtime_error("input length mismatch");
    in.seekg(20);initialise_tables();CutHistogramModel model;CutTileScorer walker;
    std::unordered_map<uint32_t,CutTileIndex> left_cache;
    auto left=[&](uint32_t key)->const CutTileIndex& {
        auto it=left_cache.find(key);
        if(it==left_cache.end())it=left_cache.emplace(key,CutTileIndex(model.layout(key))).first;
        return it->second;
    };
    double start=response::now();
    for(uint64_t i=0;i<count;++i) {
        uint64_t key,weight;in.read(reinterpret_cast<char*>(&key),8);in.read(reinterpret_cast<char*>(&weight),8);
        if(!in||!weight||__builtin_popcountll(key)>32)throw std::runtime_error("invalid input record");
        for(unsigned side=0;side<2;++side) {
            uint64_t k=side?~key:key;
            const auto& a=left(reuse_cut::half(k,15));
            CutTileIndex b(model.layout(reuse_cut::half(k,240)));
            auto features=measure(a,b,walker);
            std::cout<<"{\"type\":\"join\",\"record\":"<<i<<",\"side\":"<<side;
            for(unsigned j=0;j<feature_count;++j)std::cout<<",\""<<names[j]<<"\":"<<features[j];
            std::cout<<"}\n";
        }
        if(response::now()-start>seconds)throw std::runtime_error("census time cap; incomplete output");
        if(i%512==0)std::cerr<<"records="<<i<<"/"<<count<<" seconds="<<response::now()-start<<'\n';
    }
    std::cout<<"{\"type\":\"complete\",\"records\":"<<count<<",\"seconds\":"<<response::now()-start
        <<",\"canonical_sources\":"<<model.cache.size()<<",\"canonical_entries\":"<<model.entries<<"}\n";
    if(!std::cout)throw std::runtime_error("output write failed");
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
