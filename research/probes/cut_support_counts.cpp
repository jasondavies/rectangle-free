// Cheap candidate scores without constructing labelled prefix layouts or tiles.
#define REUSE_BUDGET_CUT_NO_MAIN
#include "reuse_budget_cut_census.cpp"

int main(int argc,char** argv) try {
    if(argc!=3&&argc!=4)throw std::runtime_error("usage: INPUT_TSV OUTPUT_CACHE [INPUT_CACHE]");
    if(std::ifstream(argv[2]).good())throw std::runtime_error("output cache already exists");
    double process_started=response::now();
    initialise_tables();
    std::map<uint32_t,uint32_t> cache;
    if(argc==4) {
        std::ifstream in(argv[3]);std::string magic;uint32_t key;uint64_t count;
        if(!(in>>magic)||magic!="R8SUPPORT1")throw std::runtime_error("invalid count cache header");
        while(in>>key) {
            if(!(in>>count)||count>UINT32_MAX||canonical_prefix(key,4).key!=key||!cache.emplace(key,uint32_t(count)).second)
                throw std::runtime_error("invalid count cache entry");
        }
        if(!in.eof())throw std::runtime_error("invalid count cache");
    }
    uint64_t builds=0,lookups=0;double build_seconds=0;
    std::unordered_map<uint32_t,std::pair<uint32_t,uint32_t>> raw_counts;
    // Row sorting removes redundant canonicalization for equal row multisets.
    std::unordered_map<uint32_t,uint32_t> row_multiset_keys;
    auto canonical=[&](uint32_t raw) {
        std::array<unsigned,8> rows{};
        for(unsigned i=0;i<8;++i)rows[i]=(raw>>(4*i))&15;
        std::sort(rows.begin(),rows.end());uint32_t sorted=0;
        for(auto r:rows)sorted=(sorted<<4)|r;
        auto it=row_multiset_keys.find(sorted);
        if(it!=row_multiset_keys.end())return it->second;
        uint32_t key=canonical_prefix(sorted,4).key;
        row_multiset_keys.emplace(sorted,key);return key;
    };
    auto count=[&](uint32_t raw) {
        ++lookups;uint32_t key=canonical(raw);auto it=cache.find(key);
        if(it!=cache.end())return it->second;
        double start=response::now();
        auto d=quotient_token_planes(build_distribution(key,4,false));
        if(d.entries.size()>UINT32_MAX)throw std::runtime_error("count overflow");
        uint32_t n=d.entries.size();cache.emplace(key,n);++builds;
        build_seconds+=response::now()-start;return n;
    };
    auto pair=[&](uint32_t raw) {
        auto it=raw_counts.find(raw);if(it!=raw_counts.end())return it->second;
        auto c=std::make_pair(count(raw),count(~raw));raw_counts.emplace(raw,c);
        std::cout<<"{\"type\":\"support\",\"key\":"<<raw<<",\"selected\":"<<c.first<<",\"complement\":"<<c.second<<"}\n";
        return c;
    };
    double started=response::now();unsigned records=0;
    std::ifstream input(argv[1]);std::string line;
    const auto cuts=reuse_cut::enumerate("all");
    while(std::getline(input,line)) {
        std::istringstream f(line);unsigned source;uint64_t index,key,weight;std::string extra;
        if(!(f>>source>>index>>key>>weight)||f>>extra||!weight||__builtin_popcountll(key)>32||++records>16384)
            throw std::runtime_error("invalid support input TSV");
        struct Candidate {uint32_t left,right;uint64_t product;};
        std::vector<Candidate> candidates;
        for(auto [tr,cut]:cuts) {
            uint64_t oriented=tr?transpose(key):key;
            uint32_t l=reuse_cut::half(oriented,cut),r=reuse_cut::half(oriented,255^cut);
            auto a=pair(l),b=pair(r);
            U128 product=U128(a.first)*b.first+U128(a.second)*b.second;
            if(product>UINT64_MAX)throw std::runtime_error("support product overflow");
            candidates.push_back({l,r,uint64_t(product)});
            candidates.push_back({r,l,uint64_t(product)});
        }
        std::cout<<"{\"type\":\"record\",\"source\":"<<source<<",\"index\":"<<index<<",\"choices\":[";
        for(size_t i=0;i<candidates.size();++i) {
            if(i)std::cout<<',';auto c=candidates[i];
            std::cout<<"{\"left\":"<<c.left<<",\"right\":"<<c.right<<",\"product\":"<<c.product<<"}";
        }
        std::cout<<"]}"<<std::endl;
    }
    if(!input.eof()||!records)throw std::runtime_error("empty/invalid support input");
    std::ofstream out(argv[2]);if(!out)throw std::runtime_error("cannot write count cache");
    out<<"R8SUPPORT1\n";for(auto [k,n]:cache)out<<k<<' '<<n<<'\n';out.close();
    if(!out)throw std::runtime_error("count cache write failed");
    std::cout<<"{\"type\":\"complete\",\"records\":"<<records<<",\"supports\":"<<raw_counts.size()
        <<",\"canonical_sources\":"<<cache.size()<<",\"new_builds\":"<<builds<<",\"lookups\":"<<lookups
        <<",\"build_seconds\":"<<build_seconds<<",\"seconds\":"<<response::now()-started
        <<",\"process_seconds\":"<<response::now()-process_started<<"}"<<std::endl;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
