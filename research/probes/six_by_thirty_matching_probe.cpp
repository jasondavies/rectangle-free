// Exact symmetry-reduced CPU perfect-matching gate for T_4(6,30).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
#endif

using Clock = std::chrono::steady_clock;

namespace {

constexpr unsigned ROWS = 6, COLOURS = 4, PAIRS = 15;
constexpr uint16_t FULL = (UINT16_C(1) << PAIRS) - 1;

template <size_t LIMBS> struct BigUInt {
    std::array<uint64_t,LIMBS> limb{};
    BigUInt(uint64_t value=0) { limb[0]=value; }
    template <size_t OTHER> explicit BigUInt(const BigUInt<OTHER>& other) {
        for(size_t i=0;i<std::min(LIMBS,OTHER);++i) limb[i]=other.limb[i];
    }
    explicit operator bool() const {
        for(uint64_t value:limb) if(value) return true;
        return false;
    }
    void add_mul(const BigUInt& other,uint64_t multiplier) {
        unsigned __int128 carry=0;
        for(size_t i=0;i<LIMBS;++i) {
            unsigned __int128 value=(unsigned __int128)other.limb[i]*multiplier+limb[i]+carry;
            limb[i]=uint64_t(value); carry=value>>64;
        }
        if(carry) throw std::overflow_error("BigUInt add overflow");
    }
    void mul_small(uint64_t multiplier) {
        unsigned __int128 carry=0;
        for(size_t i=0;i<LIMBS;++i) {
            unsigned __int128 value=(unsigned __int128)limb[i]*multiplier+carry;
            limb[i]=uint64_t(value); carry=value>>64;
        }
        if(carry) throw std::overflow_error("BigUInt multiply overflow");
    }
    void shift_left(unsigned bits) {
        if(!bits||bits>=64) throw std::runtime_error("unsupported BigUInt shift");
        for(size_t i=LIMBS;i-->0;)
            limb[i]=(limb[i]<<bits)|(i?limb[i-1]>>(64-bits):0);
    }
    unsigned bit_length() const {
        for(size_t i=LIMBS;i-->0;) if(limb[i])
            return unsigned(64*i+64-__builtin_clzll(limb[i]));
        return 0;
    }
    std::string decimal() const {
        BigUInt copy=*this; std::string reversed;
        while(copy) {
            unsigned __int128 remainder=0;
            for(size_t i=LIMBS;i-->0;) {
                unsigned __int128 value=(remainder<<64)|copy.limb[i];
                copy.limb[i]=uint64_t(value/10); remainder=value%10;
            }
            reversed.push_back(char('0'+unsigned(remainder)));
        }
        if(reversed.empty()) return "0";
        return std::string(reversed.rbegin(),reversed.rend());
    }
};

using Big = BigUInt<3>;
using Wide = BigUInt<6>;

struct Limit : std::runtime_error { using std::runtime_error::runtime_error; };
struct Options {
    uint64_t max_states=0, max_frontier=0, max_configs=0;
    double max_seconds=0;
    uint64_t progress=100000;
    unsigned threads=1, slack=0;
    bool census=false, defect_census=false;
};

struct Geometry {
    std::array<std::pair<uint8_t,uint8_t>,PAIRS> pairs{};
    std::array<uint16_t,PAIRS> disjoint{};
    std::vector<std::array<std::array<uint16_t,256>,2>> tables;

    Geometry() {
        unsigned k=0;
        for(unsigned i=0;i<ROWS;++i) for(unsigned j=i+1;j<ROWS;++j)
            pairs[k++]={uint8_t(i),uint8_t(j)};
        auto pair_index=[&](unsigned a,unsigned b){
            if(a>b) std::swap(a,b);
            for(unsigned p=0;p<PAIRS;++p)
                if(pairs[p].first==a && pairs[p].second==b) return p;
            throw std::runtime_error("pair index failure");
        };
        for(unsigned p=0;p<PAIRS;++p) {
            auto [a,b]=pairs[p];
            for(unsigned q=0;q<PAIRS;++q) {
                auto [c,d]=pairs[q];
                if(a!=c && a!=d && b!=c && b!=d) disjoint[p]|=uint16_t(1U<<q);
            }
        }
        std::array<unsigned,ROWS> perm{};
        for(unsigned i=0;i<ROWS;++i) perm[i]=i;
        do {
            std::array<unsigned,PAIRS> image{};
            for(unsigned p=0;p<PAIRS;++p)
                image[p]=pair_index(perm[pairs[p].first],perm[pairs[p].second]);
            std::array<std::array<uint16_t,256>,2> t{};
            for(unsigned chunk=0;chunk<2;++chunk) for(unsigned byte=0;byte<256;++byte)
                for(unsigned bit=0;bit<8;++bit) {
                    unsigned source=8*chunk+bit;
                    if(source<PAIRS && (byte&(1U<<bit))) t[chunk][byte]|=uint16_t(1U<<image[source]);
                }
            tables.push_back(t);
        } while(std::next_permutation(perm.begin(),perm.end()));
        if(tables.size()!=720) throw std::runtime_error("S6 size failure");
    }
};

uint64_t pack(std::array<uint16_t,COLOURS> lanes) {
    std::sort(lanes.begin(),lanes.end());
    uint64_t state=0;
    for(unsigned c=0;c<COLOURS;++c) state|=uint64_t(lanes[c])<<(PAIRS*c);
    return state;
}
std::array<uint16_t,COLOURS> unpack(uint64_t state) {
    std::array<uint16_t,COLOURS> lanes{};
    for(unsigned c=0;c<COLOURS;++c) lanes[c]=uint16_t((state>>(PAIRS*c))&FULL);
    return lanes;
}

uint64_t canonicalize(const Geometry& geometry,uint64_t state) {
    auto lanes=unpack(state); uint64_t best=UINT64_MAX;
    for(const auto& table:geometry.tables) {
        std::array<uint16_t,COLOURS> image{};
        for(unsigned colour=0;colour<COLOURS;++colour)
            image[colour]=table[0][lanes[colour]&255]|table[1][lanes[colour]>>8];
        best=std::min(best,pack(image));
    }
    return best;
}

struct DefectKey {
    uint64_t occupied=0;
    uint8_t count=0;
    bool operator==(const DefectKey& other) const {
        return occupied==other.occupied && count==other.count;
    }
};
struct DefectKeyHash {
    size_t operator()(const DefectKey& key) const {
        uint64_t x=key.occupied^(uint64_t(key.count)*UINT64_C(0x9e3779b97f4a7c15));
        x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9);
        x^=x>>27; x*=UINT64_C(0x94d049bb133111eb);
        return size_t(x^(x>>31));
    }
};
struct WeightedSupport { uint64_t mask=0; uint16_t weight=0; uint8_t excess=0; };

std::vector<WeightedSupport> build_weighted_supports(const Geometry& geometry) {
    std::unordered_map<uint64_t,uint16_t> weights;
    weights.reserve(4096);
    for(unsigned encoded=0;encoded<4096;++encoded) {
        unsigned value=encoded;
        std::array<unsigned,ROWS> colour{};
        for(unsigned row=0;row<ROWS;++row) { colour[row]=value&3U; value>>=2; }
        uint64_t mask=0;
        for(unsigned c=0;c<COLOURS;++c) for(unsigned pair=0;pair<PAIRS;++pair) {
            auto [a,b]=geometry.pairs[pair];
            if(colour[a]==c && colour[b]==c) mask|=UINT64_C(1)<<(c*PAIRS+pair);
        }
        ++weights[mask];
    }
    std::vector<WeightedSupport> supports;
    supports.reserve(weights.size());
    uint64_t physical=0;
    std::array<uint64_t,16> unique_by_size{},physical_by_size{};
    for(auto [mask,weight]:weights) {
        unsigned size=unsigned(__builtin_popcountll(mask));
        if(size<2) throw std::runtime_error("six-row support has fewer than two tokens");
        supports.push_back({mask,weight,uint8_t(size-2)});
        ++unique_by_size[size]; physical_by_size[size]+=weight; physical+=weight;
    }
    std::sort(supports.begin(),supports.end(),[](const auto& a,const auto& b){
        if(a.excess!=b.excess) return a.excess<b.excess;
        return a.mask<b.mask;
    });
    std::printf("DEFECT_SUPPORTS unique=%zu physical=%llu sizes=",supports.size(),
        (unsigned long long)physical);
    bool first=true;
    for(unsigned size=0;size<unique_by_size.size();++size) if(unique_by_size[size]) {
        std::printf("%s%u:%llu/%llu",first?"":",",size,
            (unsigned long long)unique_by_size[size],
            (unsigned long long)physical_by_size[size]);
        first=false;
    }
    std::printf(" exact=%s\n",physical==4096&&supports.size()==2088?"OK":"FAIL");
    if(physical!=4096||supports.size()!=2088) throw std::runtime_error("weighted support census mismatch");
    return supports;
}

class DefectCensus {
  public:
    DefectCensus(const Geometry& geometry,Options options)
        :geometry_(geometry),options_(options),start_(Clock::now()),
         supports_(build_weighted_supports(geometry)) {
        budget_=2*options_.slack;
        for(const auto& support:supports_)
            if(support.excess && support.excess<=budget_) defects_.push_back(support);
    }

    void run() {
        enumerate(0,0,0,0,Wide(1));
        std::printf("DEFECT_RAW slack=%u budget=%u candidates=%zu enumeration_nodes=%llu unions=%zu elapsed=%.6f\n",
            options_.slack,budget_,defects_.size(),(unsigned long long)nodes_,raw_.size(),seconds());
        std::unordered_map<DefectKey,Wide,DefectKeyHash> orbit;
        orbit.reserve(raw_.size());
        for(const auto& [key,coefficient]:raw_) {
            DefectKey canonical{canonicalize(geometry_,key.occupied),key.count};
            orbit[canonical].add_mul(coefficient,1);
        }
        struct Sector { uint64_t raw=0,orbits=0; Wide coefficient=0; };
        std::array<std::array<Sector,31>,31> sectors{};
        for(const auto& [key,coefficient]:raw_) {
            unsigned excess=unsigned(__builtin_popcountll(key.occupied))-2*key.count;
            ++sectors[excess][key.count].raw;
        }
        for(const auto& [key,coefficient]:orbit) {
            unsigned excess=unsigned(__builtin_popcountll(key.occupied))-2*key.count;
            ++sectors[excess][key.count].orbits;
            sectors[excess][key.count].coefficient.add_mul(coefficient,1);
        }
        uint64_t orbit_count=0;
        for(unsigned excess=0;excess<=budget_;++excess)
            for(unsigned count=0;count<=30;++count) {
                const auto& sector=sectors[excess][count];
                if(!sector.raw) continue;
                orbit_count+=sector.orbits;
                std::string coefficient=sector.coefficient.decimal();
                unsigned unmatched=2*options_.slack-excess;
                unsigned matching_edges=30-options_.slack-count;
                std::printf("DEFECT_SECTOR excess=%u defect_columns=%u unmatched_tokens=%u matching_edges=%u raw_unions=%llu canonical_unions=%llu coefficient_sum=%s\n",
                    excess,count,unmatched,matching_edges,
                    (unsigned long long)sector.raw,(unsigned long long)sector.orbits,
                    coefficient.c_str());
            }
        std::printf("DEFECT_CENSUS status=COMPLETE slack=%u width=%u raw_unions=%zu canonical_queries=%llu elapsed=%.6f exact=OK\n",
            options_.slack,30-options_.slack,raw_.size(),(unsigned long long)orbit_count,seconds());
    }
  private:
    const Geometry& geometry_; Options options_; Clock::time_point start_;
    std::vector<WeightedSupport> supports_,defects_;
    std::unordered_map<DefectKey,Wide,DefectKeyHash> raw_;
    unsigned budget_=0;
    uint64_t nodes_=0;

    double seconds() const { return std::chrono::duration<double>(Clock::now()-start_).count(); }
    void enumerate(size_t begin,uint64_t occupied,unsigned count,unsigned excess,const Wide& coefficient) {
        if(options_.max_configs && nodes_>=options_.max_configs)
            throw Limit("defect configuration cap exceeded");
        ++nodes_;
        raw_[DefectKey{occupied,uint8_t(count)}].add_mul(coefficient,1);
        for(size_t i=begin;i<defects_.size();++i) {
            const auto& support=defects_[i];
            if(excess+support.excess>budget_ || (occupied&support.mask)) continue;
            Wide next=coefficient; next.mul_small(support.weight);
            enumerate(i+1,occupied|support.mask,count+1,excess+support.excess,next);
        }
    }
};

// Emit the children selected by the same minimum-degree recurrence as Counter.
// The returned states are colour-canonical but not yet row-canonical.
unsigned expand_raw(const Geometry& geometry,uint64_t state,uint64_t children[18]) {
    if(!state) return 0;
    auto lanes=unpack(state);
    unsigned pivot_colour=0,pivot_pair=0,min_degree=1000;
    std::array<std::pair<unsigned,unsigned>,18> neighbours{};
    unsigned neighbour_count=0;
    for(unsigned colour=0;colour<COLOURS && min_degree!=1;++colour) {
        uint16_t bits=lanes[colour];
        while(bits && min_degree!=1) {
            unsigned pair=unsigned(__builtin_ctz(bits)); bits&=uint16_t(bits-1);
            std::array<std::pair<unsigned,unsigned>,18> candidate{};
            unsigned count=0;
            for(unsigned other=0;other<COLOURS;++other) if(other!=colour) {
                uint16_t compatible=lanes[other]&geometry.disjoint[pair];
                while(compatible) {
                    unsigned q=unsigned(__builtin_ctz(compatible));
                    compatible&=uint16_t(compatible-1);
                    candidate[count++]={other,q};
                }
            }
            if(!count) return 0;
            if(count<min_degree) {
                min_degree=count; pivot_colour=colour; pivot_pair=pair;
                neighbour_count=count;
                std::copy_n(candidate.begin(),count,neighbours.begin());
            }
        }
    }
    for(unsigned i=0;i<neighbour_count;++i) {
        auto [other,q]=neighbours[i];
        auto child=lanes;
        child[pivot_colour]&=uint16_t(~(1U<<pivot_pair));
        child[other]&=uint16_t(~(1U<<q));
        children[i]=pack(child);
    }
    return neighbour_count;
}

void exact_sort(std::vector<uint64_t>& values,unsigned threads) {
#ifdef _OPENMP
    if(threads>1 && values.size()>=1000000) {
        omp_set_num_threads(int(threads));
        __gnu_parallel::sort(values.begin(),values.end());
        return;
    }
#else
    (void)threads;
#endif
    std::sort(values.begin(),values.end());
}

void unique_in_place(std::vector<uint64_t>& values) {
    values.erase(std::unique(values.begin(),values.end()),values.end());
}

class FrontierCensus {
  public:
    FrontierCensus(const Geometry& geometry,Options options)
        :geometry_(geometry),options_(options),start_(Clock::now()) {}

    void run() {
#ifndef _OPENMP
        if(options_.threads>1) throw std::runtime_error("census binary lacks OpenMP support");
#else
        omp_set_dynamic(0);
        omp_set_num_threads(int(options_.threads));
#endif
        std::array<uint16_t,COLOURS> full{}; full.fill(FULL);
        std::vector<uint64_t> frontier{canonicalize(geometry_,pack(full))};
        uint64_t total=0;
        for(unsigned depth=0;;++depth) {
            total+=frontier.size();
            unsigned remaining=60-2*depth;
            std::printf("MATCHING_CENSUS_LEVEL depth=%u remaining=%u frontier=%zu total=%llu elapsed=%.6f\n",
                depth,remaining,frontier.size(),(unsigned long long)total,seconds());
            std::fflush(stdout);
            if(!remaining) {
                std::printf("MATCHING_CENSUS status=COMPLETE levels=%u total_states=%llu elapsed=%.6f exact=OK\n",
                    depth+1,(unsigned long long)total,seconds());
                return;
            }
            if(options_.max_seconds && seconds()>options_.max_seconds)
                throw Limit("census time cap exceeded");

            auto phase=Clock::now();
            unsigned worker_count=std::max(1U,options_.threads);
            std::vector<std::vector<uint64_t>> chunks(worker_count);
            size_t reserve=(frontier.size()*5/worker_count)+1024;
            for(auto& chunk:chunks) chunk.reserve(reserve);
            uint64_t dead=0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:dead)
#endif
            {
                unsigned worker=0;
#ifdef _OPENMP
                worker=unsigned(omp_get_thread_num());
#endif
                auto& output=chunks[worker];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for(size_t i=0;i<frontier.size();++i) {
                    uint64_t children[18];
                    unsigned count=expand_raw(geometry_,frontier[i],children);
                    if(!count) ++dead;
                    output.insert(output.end(),children,children+count);
                }
            }
            size_t edge_count=0;
            for(const auto& chunk:chunks) edge_count+=chunk.size();
            std::vector<uint64_t> raw(edge_count);
            std::vector<size_t> offsets(worker_count+1);
            for(unsigned i=0;i<worker_count;++i) offsets[i+1]=offsets[i]+chunks[i].size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(unsigned i=0;i<worker_count;++i)
                std::memcpy(raw.data()+offsets[i],chunks[i].data(),chunks[i].size()*sizeof(uint64_t));
            chunks.clear();
            double expand_seconds=elapsed(phase); phase=Clock::now();

            exact_sort(raw,options_.threads); unique_in_place(raw);
            size_t raw_unique=raw.size();
            double raw_sort_seconds=elapsed(phase); phase=Clock::now();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(size_t i=0;i<raw.size();++i) raw[i]=canonicalize(geometry_,raw[i]);
            double canonical_seconds=elapsed(phase); phase=Clock::now();

            exact_sort(raw,options_.threads); unique_in_place(raw);
            double canonical_sort_seconds=elapsed(phase);
            std::printf("MATCHING_CENSUS_TRANSITION depth=%u edges=%zu dead=%llu raw_unique=%zu canonical_unique=%zu symmetry_collapse=%.6f expand_seconds=%.6f raw_sort_seconds=%.6f canonical_seconds=%.6f canonical_sort_seconds=%.6f\n",
                depth,edge_count,(unsigned long long)dead,raw_unique,raw.size(),
                raw.empty()?0.0:double(raw_unique)/double(raw.size()),expand_seconds,
                raw_sort_seconds,canonical_seconds,canonical_sort_seconds);
            std::fflush(stdout);
            if(options_.max_frontier && raw.size()>options_.max_frontier)
                throw Limit("census frontier cap exceeded");
            frontier.swap(raw);
            if(frontier.empty()) throw std::runtime_error("census lost all states before empty matching");
        }
    }
  private:
    const Geometry& geometry_; Options options_; Clock::time_point start_;
    double seconds() const { return elapsed(start_); }
    static double elapsed(Clock::time_point begin) {
        return std::chrono::duration<double>(Clock::now()-begin).count();
    }
};

class Counter {
  public:
    Counter(const Geometry& g,Options o):g_(g),o_(o),start_(Clock::now()) {
        canonical_.reserve(o.max_states?o.max_states*2:1000000);
        memo_.reserve(o.max_states?o.max_states:1000000);
    }
    Big run(){ std::array<uint16_t,COLOURS> a{}; a.fill(FULL); return solve(canon(pack(a))); }
    void report(const char* status,const Big* value=nullptr) const {
        std::printf("MATCHING_CPP status=%s vertices=60 degree=18 edges=540 states=%llu branches=%llu canonical_states=%zu canonical_hits=%llu row_images=%llu memo_hits=%llu elapsed=%.6f",
            status,(unsigned long long)states_,(unsigned long long)branches_,canonical_.size(),
            (unsigned long long)canonical_hits_,(unsigned long long)row_images_,
            (unsigned long long)memo_hits_,seconds());
        if(value) {
            std::string s=value->decimal();
            std::printf(" perfect_matchings=%s bits=%u",s.c_str(),value->bit_length());
        }
        std::printf("\n");
    }
  private:
    const Geometry& g_; Options o_; Clock::time_point start_;
    std::unordered_map<uint64_t,uint64_t> canonical_;
    std::unordered_map<uint64_t,Big> memo_;
    uint64_t states_=0,branches_=0,canonical_hits_=0,row_images_=0,memo_hits_=0;
    double seconds() const { return std::chrono::duration<double>(Clock::now()-start_).count(); }

    uint64_t canon(uint64_t state) {
        auto f=canonical_.find(state);
        if(f!=canonical_.end()){++canonical_hits_;return f->second;}
        uint64_t best=canonicalize(g_,state);
        row_images_+=g_.tables.size(); canonical_.emplace(state,best); return best;
    }

    Big solve(uint64_t state) {
        if(!state) return 1;
        auto known=memo_.find(state);
        if(known!=memo_.end()){++memo_hits_;return known->second;}
        ++states_;
        if(o_.max_states && states_>o_.max_states) throw Limit("state cap exceeded");
        if((states_&4095)==0 && o_.max_seconds && seconds()>o_.max_seconds)
            throw Limit("time cap exceeded");
        if(o_.progress && states_%o_.progress==0) {
            std::printf("MATCHING_CPP_PROGRESS states=%llu branches=%llu elapsed=%.3f\n",
                (unsigned long long)states_,(unsigned long long)branches_,seconds());
            std::fflush(stdout);
        }
        auto lanes=unpack(state);
        unsigned pc=0,pp=0,min_degree=1000;
        std::vector<std::pair<unsigned,unsigned>> pivot;
        for(unsigned c=0;c<COLOURS && min_degree!=1;++c) {
            uint16_t bits=lanes[c];
            while(bits && min_degree!=1) {
                unsigned p=unsigned(__builtin_ctz(bits)); bits&=uint16_t(bits-1);
                std::vector<std::pair<unsigned,unsigned>> neighbours;
                for(unsigned d=0;d<COLOURS;++d) if(d!=c) {
                    uint16_t candidates=lanes[d]&g_.disjoint[p];
                    while(candidates) {
                        unsigned q=unsigned(__builtin_ctz(candidates));
                        candidates&=uint16_t(candidates-1); neighbours.push_back({d,q});
                    }
                }
                if(neighbours.empty()){memo_.emplace(state,0);return 0;}
                if(neighbours.size()<min_degree) {
                    min_degree=unsigned(neighbours.size());pc=c;pp=p;pivot.swap(neighbours);
                }
            }
        }
        std::vector<uint64_t> children; children.reserve(pivot.size());
        for(auto [d,q]:pivot) {
            auto child=lanes;
            child[pc]&=uint16_t(~(1U<<pp)); child[d]&=uint16_t(~(1U<<q));
            children.push_back(canon(pack(child)));
        }
        branches_+=children.size(); std::sort(children.begin(),children.end());
        Big result=0;
        for(size_t begin=0;begin<children.size();) {
            size_t end=begin+1; while(end<children.size()&&children[end]==children[begin])++end;
            Big child=solve(children[begin]);
            result.add_mul(child,end-begin); begin=end;
        }
        memo_.emplace(state,result); return result;
    }
};

uint64_t u64(const char* s){char* e=nullptr;auto v=std::strtoull(s,&e,10);if(!e||*e)throw std::runtime_error("bad integer");return v;}
double real(const char* s){char* e=nullptr;auto v=std::strtod(s,&e);if(!e||*e)throw std::runtime_error("bad number");return v;}

} // namespace

int main(int argc,char** argv) {
    try {
        Options o;
        for(int i=1;i<argc;++i) {
            std::string s=argv[i];
            if(s=="--max-states"&&i+1<argc)o.max_states=u64(argv[++i]);
            else if(s=="--max-frontier"&&i+1<argc)o.max_frontier=u64(argv[++i]);
            else if(s=="--max-configs"&&i+1<argc)o.max_configs=u64(argv[++i]);
            else if(s=="--max-seconds"&&i+1<argc)o.max_seconds=real(argv[++i]);
            else if(s=="--progress-every"&&i+1<argc)o.progress=u64(argv[++i]);
            else if(s=="--threads"&&i+1<argc)o.threads=unsigned(u64(argv[++i]));
            else if(s=="--slack"&&i+1<argc)o.slack=unsigned(u64(argv[++i]));
            else if(s=="--census")o.census=true;
            else if(s=="--defect-census")o.defect_census=true;
            else throw std::runtime_error("usage: six_by_thirty_matching_probe [--census|--defect-census --slack S] [--threads N] [--max-frontier N] [--max-configs N] [--max-states N] [--max-seconds S] [--progress-every N]");
        }
        if(!o.threads) throw std::runtime_error("threads must be positive");
        if(o.census&&o.defect_census) throw std::runtime_error("choose only one census mode");
        if(o.slack>29) throw std::runtime_error("defect slack must be at most 29");
        Geometry g;
        std::printf("MATCHING_CPP_GRAPH vertices=60 degree=18 edges=540 maximum_columns=30 column_weight=2 symmetry_order=17280 colour_sectors=136 exact=OK\n");
        if(o.defect_census) {
            try { DefectCensus(g,o).run(); return 0; }
            catch(const Limit& e) {
                std::printf("DEFECT_CENSUS status=LIMIT reason=%s\n",e.what()); return 3;
            }
        }
        if(o.census) {
            try { FrontierCensus(g,o).run(); return 0; }
            catch(const Limit& e) {
                std::printf("MATCHING_CENSUS status=LIMIT reason=%s\n",e.what()); return 3;
            }
        }
        Counter counter(g,o);
        try {
            Big pm=counter.run(); counter.report("COMPLETE",&pm);
            Wide result(pm); result.shift_left(30); for(unsigned i=2;i<=30;++i)result.mul_small(i);
            std::string text=result.decimal();
            std::printf("MATCHING_CPP_RESULT T_4(6,30)=%s bits=%u exact=OK\n",text.c_str(),result.bit_length());
            return 0;
        } catch(const Limit& e) {
            counter.report("LIMIT"); std::printf("MATCHING_CPP_LIMIT reason=%s\n",e.what()); return 3;
        }
    } catch(const std::exception& e) { std::fprintf(stderr,"error: %s\n",e.what()); return 2; }
}
