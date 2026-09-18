#include "response_model.hpp"
using namespace response;
static void family(unsigned rows,unsigned n,uint64_t key,std::vector<unsigned> demand,
                   size_t cache,uint64_t work,double seconds) {
    if(rows<2||rows>8||n<1||n>7||rows*n>56||key>>(rows*n))throw std::runtime_error("invalid core geometry/key");
    if(cache>1000000||!work||work>100000000000ull||!std::isfinite(seconds)||seconds<=0||seconds>300)
        throw std::runtime_error("invalid resource caps");
    Engine e(rows);
    if(demand.empty()){demand.resize(1u<<rows);std::iota(demand.begin(),demand.end(),0);}
    std::sort(demand.begin(),demand.end());demand.erase(std::unique(demand.begin(),demand.end()),demand.end());
    if(demand.back()>=(1u<<rows))throw std::runtime_error("extension out of range");
    std::cout<<"{\"rows\":"<<rows<<",\"core_columns\":"<<n<<",\"key\":\""<<key
        <<"\",\"demand\":"<<demand.size()<<",\"cache_cap\":"<<cache<<",\"work_cap\":"<<work<<"}"<<std::endl;
    auto core=columns(rows,n,key);
    for(unsigned side=0;side<2;++side) {
        auto c=side?complement(core,rows):core;
        auto queries=demand;if(side)for(auto& a:queries)a^=(1u<<rows)-1;
        auto ref=independent(e,c,queries,Budget(work,seconds)); print_result("independent",side,ref);
        auto got=shared(e,c,queries,cache,Budget(work,seconds));print_result("shared",side,got);
        if(ref.status=="complete"&&got.status=="complete") {
            for(unsigned a:queries)if(ref.values[a]!=got.values[a])throw std::runtime_error("FAMILY COUNT MISMATCH");
            std::cout<<"{\"parity\":\"exact\",\"side\":"<<side<<",\"answers\":"<<queries.size()<<"}"<<std::endl;
        }
    }
}

static void census(const std::string& path,uint64_t limit) {
    if(!limit||limit>10000000)throw std::runtime_error("census limit must be 1..10000000");
    std::ifstream in(path,std::ios::binary);char magic[8];uint64_t count;uint32_t width;
    in.read(magic,8);in.read(reinterpret_cast<char*>(&width),4);in.read(reinterpret_cast<char*>(&count),8);
    if(!in||width!=8||(std::memcmp(magic,"R8ORB01",7)&&std::memcmp(magic,"R8SQT01",7)))throw std::runtime_error("not an 8x8 orbit file");
    in.seekg(0,std::ios::end);if(U128(in.tellg())!=20+U128(16)*count)throw std::runtime_error("bad file length");
    in.seekg(20);limit=std::min(limit,count);
    struct Group { uint64_t records=0;std::array<uint64_t,4> active{}; };
    std::unordered_map<uint64_t,Group> groups;
    std::set<uint64_t> raw;
    for(uint64_t i=0;i<limit;++i) {
        uint64_t key,weight;in.read(reinterpret_cast<char*>(&key),8);in.read(reinterpret_cast<char*>(&weight),8);
        if(!in||!weight)throw std::runtime_error("bad record");
        auto [core,last]=family_key(key);auto& g=groups[core];++g.records;g.active[last/64]|=uint64_t(1)<<(last%64);
        uint64_t k=0;for(unsigned r=0;r<8;++r)k=(k<<7)|((key>>(8*(7-r)))&127);raw.insert(k);
    }
    std::vector<std::pair<uint64_t,Group>> sorted(groups.begin(),groups.end());
    auto demand_count=[](const Group& g){unsigned n=0;for(auto a:g.active)n+=__builtin_popcountll(a);return n;};
    std::sort(sorted.begin(),sorted.end(),[&](const auto& a,const auto& b){
        unsigned x=demand_count(a.second),y=demand_count(b.second);
        return x!=y?x>y:(a.second.records!=b.second.records?a.second.records>b.second.records:a.first<b.first);});
    std::map<unsigned,uint64_t> hist;for(const auto& x:sorted)++hist[demand_count(x.second)];
    std::cout<<"{\"records\":"<<limit<<",\"file_records\":"<<count<<",\"raw_families\":"<<raw.size()
        <<",\"row_gauge_families\":"<<groups.size()<<",\"demand_histogram\":{";
    bool first=true;for(auto [k,v]:hist){if(!first)std::cout<<",";first=false;std::cout<<"\""<<k<<"\":"<<v;}std::cout<<"}}\n";
    std::vector<size_t> picks;
    for(size_t i=0;i<std::min<size_t>(8,sorted.size());++i)picks.push_back(i);
    for(size_t i=0;i<8&&i<sorted.size();++i)picks.push_back(i*(sorted.size()-1)/7);
    unsigned balanced=0;
    for(size_t i=0;i<sorted.size()&&balanced<8;++i) {
        unsigned active=__builtin_popcountll(sorted[i].first);
        if(active>=26&&active<=30){picks.push_back(i);++balanced;}
    }
    std::sort(picks.begin(),picks.end());picks.erase(std::unique(picks.begin(),picks.end()),picks.end());
    for(size_t i:picks) {
        auto [key,g]=sorted[i];std::cout<<"{\"rank\":"<<i<<",\"key\":\""<<key<<"\",\"records\":"<<g.records<<",\"extensions\":[";
        bool comma=false;for(unsigned a=0;a<256;++a)if(g.active[a/64]>>(a%64)&1){if(comma)std::cout<<",";comma=true;std::cout<<a;}
        std::cout<<"]}\n";
    }
}

static void self_test() {
    for(unsigned r=2;r<=3;++r) {
        Engine e(r);std::vector<unsigned> demand(1u<<r);std::iota(demand.begin(),demand.end(),0);
        U128 grid_count=0;
        for(uint64_t key=0;key<(uint64_t(1)<<(2*r));++key) {
            auto core=columns(r,2,key);
            Budget budget;auto d=e.full(core,budget);
            auto ref=independent(e,core,demand,Budget());
            for(size_t cache:{size_t(0),size_t(1),size_t(7),size_t(1000)}) {
                auto got=shared(e,core,demand,cache,Budget());
                if(got.status!="complete"||ref.status!="complete")throw std::runtime_error("unexpected cap");
                for(unsigned a:demand) {
                    U128 exact=0;for(auto [m,w]:d)for(auto x:e.increments[a])if(!(m&x.mask))exact+=U128(w)*x.weight;
                    if(exact!=got.values[a]||exact!=ref.values[a])throw std::runtime_error("self test count");
                }
            }
            std::vector<unsigned> subset={1,(1u<<r)-1};
            auto got=shared(e,core,subset,7,Budget());
            for(unsigned a:subset)if(got.values[a]!=ref.values[a])throw std::runtime_error("demand trie");
            auto comp=shared(e,complement(core,r),demand,7,Budget());
            for(unsigned a:demand)grid_count+=ref.values[a]*comp.values[a^((1u<<r)-1)];
        }
        if(grid_count!=(r==2?3912:228984))throw std::runtime_error("known labelled grid count");
    }
    for(unsigned r=4;r<=8;++r) {
        Engine e(r);std::vector<unsigned> demand(1u<<r);std::iota(demand.begin(),demand.end(),0);
        std::vector<unsigned> core={3,12u&((1u<<r)-1),48u&((1u<<r)-1)};
        Budget budget;auto full=e.full(core,budget);
        auto baseline=independent(e,core,demand,Budget());
        for(size_t cache:{size_t(0),size_t(7)}) {
            auto got=shared(e,core,demand,cache,Budget());
            if(got.status!="complete"||baseline.status!="complete")throw std::runtime_error("wide fixture cap");
            for(unsigned a:demand) {
                U128 expected=0;
                for(auto [m,w]:full)for(auto x:e.increments[a])if(!(m&x.mask))expected+=U128(w)*x.weight;
                if(expected!=got.values[a]||expected!=baseline.values[a])throw std::runtime_error("wide unquotiented fixture");
            }
        }
    }
    Engine e(3);auto capped=shared(e,{7,7},{7},1,Budget(1));
    if(capped.status=="complete"||!capped.values.empty())throw std::runtime_error("partial result accepted");
    // The census row relabelling preserves every column, including the extension.
    for(uint64_t key:{0ull,~0ull,0x0102030405060708ull,0x8888444422221111ull}) {
        auto [core,last]=family_key(key);auto c=columns(8,7,core);c.push_back(last);
        std::array<unsigned,8> before{},after{};
        for(unsigned r=0;r<8;++r){before[r]=unsigned(key>>(8*r))&255;
            for(unsigned col=0;col<8;++col)after[r]|=((c[col]>>r)&1)<<col;}
        std::sort(before.begin(),before.end());std::sort(after.begin(),after.end());
        if(before!=after)throw std::runtime_error("family row map");
    }
    std::cout<<"SHARED_COLUMN_RESPONSE_TEST exact=OK\n";
}
int main(int argc,char** argv) try {
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc==4&&std::string(argv[1])=="census"){census(argv[2],std::stoull(argv[3]));return 0;}
    if(argc==9&&std::string(argv[1])=="family") {
        std::vector<unsigned> queries;
        if(std::string(argv[5])!="all") {std::stringstream s(argv[5]);std::string x;while(std::getline(s,x,','))queries.push_back(std::stoul(x));}
        family(std::stoul(argv[2]),std::stoul(argv[3]),std::stoull(argv[4],nullptr,0),queries,
               std::stoull(argv[6]),std::stoull(argv[7]),std::stod(argv[8]));
        return 0;
    }
    throw std::runtime_error("usage: --self-test | census R8ORB LIMIT | family ROWS CORE_COLUMNS KEY EXTENSIONS_OR_all CACHE_ENTRIES WORK_CAP SECONDS");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
