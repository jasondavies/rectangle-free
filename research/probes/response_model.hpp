#pragma once
// Independent research oracle; no production dependency.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace response {
using U128 = unsigned __int128;
using Clock = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
struct Cap : std::runtime_error { using std::runtime_error::runtime_error; };
struct Budget {
    uint64_t limit, build=0, predicates=0, response_nodes=0, buckets=0, total=0;
    double start=now(), seconds;
    size_t state_cap;
    uint64_t modeled_tiles=0;
    double build_seconds=0,model_seconds=0;
    Budget(uint64_t work=100000000, double time=30, size_t states=1000000)
        :limit(work),seconds(time),state_cap(states) {}
    void tick(uint64_t& field) {
        ++field;
        if (++total > limit) throw Cap("work_cap");
        if (!(total & 65535) && now()-start > seconds) throw Cap("time_cap");
    }
};
struct Timer {
    double& output;double start=now();
    explicit Timer(double& value):output(value){}
    ~Timer(){output+=now()-start;}
};
struct Entry { uint64_t mask, weight; };
struct ClassSize { uint64_t count;unsigned orbit; };
struct Bucket { unsigned prefix; std::vector<Entry> entries;std::vector<ClassSize> classes; };
using Dist = std::vector<Bucket>;
using DemandTrie = std::array<std::array<bool,256>,9>;
static DemandTrie demand_trie(unsigned rows,const std::vector<unsigned>& demand) {
    DemandTrie live{};
    for(unsigned a:demand)for(unsigned k=0;k<=rows;++k) live[k][a&((1u<<k)-1)]=true;
    return live;
}

struct Engine {
    unsigned rows,pairs,prefix_width;
    uint64_t plane;
    std::vector<unsigned> prefix_pairs;
    std::vector<std::vector<Entry>> increments;
    unsigned pair_index[8][8]{};
    explicit Engine(unsigned r):rows(r),pairs(r*(r-1)/2),prefix_width(std::min(7u,pairs)),
        plane((uint64_t(1)<<pairs)-1),increments(1u<<r) {
        if(r<2 || r>8) throw std::runtime_error("rows must be 2..8");
        unsigned p=0;
        for(unsigned i=0;i<r;++i) for(unsigned j=i+1;j<r;++j) pair_index[i][j]=p++;
        if(r==8) prefix_pairs={0,1,7,2,8,13,27};
        else for(unsigned i=0;i<prefix_width;++i) prefix_pairs.push_back(i);
        for(unsigned a=0;a<(1u<<r);++a) {
            std::map<uint64_t,uint64_t> counts;
            unsigned b=a;
            for(;;) {
                uint64_t mask=0;
                for(unsigned i=0;i<r;++i) for(unsigned j=i+1;j<r;++j)
                    if((a>>i&1)&&(a>>j&1)&&((b>>i&1)==(b>>j&1)))
                        mask |= uint64_t(1) << (pair_index[i][j]+(b>>i&1)*pairs);
                ++counts[mask]; if(!b)break; b=(b-1)&a;
            }
            for(auto [mask,w]:counts) increments[a].push_back({mask,w});
        }
    }
    uint64_t swap(uint64_t m) const { return ((m&plane)<<pairs)|(m>>pairs); }
    unsigned orbit(uint64_t m) const { return 1+(m!=swap(m)); }
    unsigned prefix(uint64_t m) const {
        unsigned p=0;
        for(unsigned c=0;c<2;++c) for(unsigned i=0;i<prefix_width;++i)
            p|=unsigned(m>>(c*pairs+prefix_pairs[i])&1)<<(c*prefix_width+i);
        return p;
    }
    unsigned swap_prefix(unsigned p) const {
        unsigned m=(1u<<prefix_width)-1;
        return ((p&m)<<prefix_width)|(p>>prefix_width);
    }
    static void add(std::unordered_map<uint64_t,uint64_t>& d,uint64_t m,U128 w) {
        auto& v=d[m];
        if(w > UINT64_MAX-v) throw std::overflow_error("distribution weight overflow");
        v+=uint64_t(w);
    }
    // Independent full-support DP. Only the completed half is quotiented.
    std::unordered_map<uint64_t,uint64_t> full(const std::vector<unsigned>& columns,Budget& b) const {
        std::unordered_map<uint64_t,uint64_t> d{{0,1}};
        for(unsigned a:columns) {
            std::unordered_map<uint64_t,uint64_t> next;
            for(auto [m,w]:d) for(auto x:increments.at(a)) {
                b.tick(b.build);
                if(!(m&x.mask)) { add(next,m|x.mask,U128(w)*x.weight);
                    if(next.size()>b.state_cap) throw Cap("distribution_state_cap"); }
            }
            d=std::move(next);
        }
        return d;
    }
    Dist build(const std::vector<unsigned>& columns,Budget& b) const {
        Timer timer(b.build_seconds);
        auto d=full(columns,b);
        std::map<unsigned,std::vector<Entry>> grouped;
        for(auto [m,w]:d) if(m<=swap(m)) grouped[prefix(m)].push_back({m,w});
        Dist out;
        for(auto& [p,v]:grouped) {
            std::sort(v.begin(),v.end(),[](auto a,auto b){return a.mask<b.mask;});
            std::map<std::pair<uint64_t,unsigned>,uint64_t> sizes;
            for(auto e:v)++sizes[{e.weight,orbit(e.mask)}];
            Bucket bucket{p,std::move(v),{}};
            for(auto [key,count]:sizes)bucket.classes.push_back({count,key.second});
            out.push_back(std::move(bucket));
        }
        return out;
    }
    // Instruction-count model only, not GPU execution. Union-producing joins
    // cannot use the same aggregate BMMA operation without recovering pair IDs.
    void tile_model(const Dist& a,const Dist& d,Budget& b) const {
        Timer timer(b.model_seconds);
        uint64_t steps=0;
        for(const auto& x:a)for(const auto& y:d) {
            if(!(++steps&65535)&&now()-b.start>b.seconds)throw Cap("time_cap");
            bool f=!(x.prefix&y.prefix),s=!(x.prefix&swap_prefix(y.prefix));
            if(!f&&!s)continue;
            for(auto l:x.classes)for(auto r:y.classes) {
                uint64_t tiles=std::min(((l.count+15)/16)*((r.count+7)/8),
                                        ((r.count+15)/16)*((l.count+7)/8));
                b.modeled_tiles+=tiles*(unsigned(f)+unsigned(s&&r.orbit==2));
            }
        }
    }
    // Callback receives an un-oriented union and its total contribution weight.
    // It must be invariant under simultaneous token-plane exchange.
    template<class F> void joins(const Dist& a,const Dist& d,Budget& b,F callback) const {
        for(const auto& x:a) for(const auto& y:d) {
            b.tick(b.buckets);
            bool f=!(x.prefix&y.prefix),s=!(x.prefix&swap_prefix(y.prefix));
            if(!f&&!s)continue;
            for(auto l:x.entries) for(auto r:y.entries) {
                U128 weight=U128(orbit(l.mask))*l.weight*r.weight;
                if(f) { b.tick(b.predicates); if(!(l.mask&r.mask))callback(l.mask|r.mask,weight); }
                if(s&&orbit(r.mask)==2) { uint64_t v=swap(r.mask);
                    b.tick(b.predicates); if(!(l.mask&v))callback(l.mask|v,weight); }
            }
        }
    }
    U128 count(const Dist& a,const Dist& d,Budget& b) const {
        U128 answer=0; joins(a,d,b,[&](uint64_t,U128 w){answer+=w;}); return answer;
    }
    // Count ternary row assignments: absent, binary colour zero, binary colour one.
    // A trie of demanded active masks prevents visits to unrequested extensions.
    void response(uint64_t mask,U128 weight,const DemandTrie& live,
                  std::vector<U128>& out,Budget& b) const {
        unsigned adj[2][8]{};
        for(unsigned i=0;i<rows;++i)for(unsigned j=i+1;j<rows;++j)
            for(unsigned c=0;c<2;++c)if(mask>>(c*pairs+pair_index[i][j])&1)
                adj[c][j]|=1u<<i;
        auto dfs=[&](auto&& self,unsigned row,unsigned active,unsigned zero,unsigned one)->void {
            b.tick(b.response_nodes);
            if(row==rows){out[active]+=weight;return;}
            if(live[row+1][active]) self(self,row+1,active,zero,one);
            unsigned next=active|(1u<<row);
            if(live[row+1][next]) {
                if(!(adj[0][row]&zero))self(self,row+1,next,zero|(1u<<row),one);
                if(!(adj[1][row]&one)) self(self,row+1,next,zero,one|(1u<<row));
            }
        };
        dfs(dfs,0,0,0,0);
    }
};

static std::vector<unsigned> columns(unsigned rows,unsigned n,uint64_t key) {
    std::vector<unsigned> out(n);
    for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<n;++c)
        out[c]|=unsigned(key>>((rows-1-r)*n+c)&1)<<r;
    return out;
}
static std::vector<unsigned> complement(std::vector<unsigned> c,unsigned rows) {
    for(auto& a:c)a^=(1u<<rows)-1;
    return c;
}
struct Result {
    std::vector<U128> values;
    std::string status="complete";
    Budget budget;
    double elapsed=0;
    uint64_t unions=0,flushes=0,response_evaluations=0,peak_entries=0;
    explicit Result(Budget b):budget(b){}
};
static Result shared(Engine& e,const std::vector<unsigned>& core,const std::vector<unsigned>& demand,
                     size_t cache,Budget budget) {
    Result result(budget); result.values.resize(1u<<e.rows); auto& b=result.budget;
    const auto live=demand_trie(e.rows,demand);
    try {
        size_t split=(core.size()+1)/2;
        auto a=e.build({core.begin(),core.begin()+split},b);
        if(a.empty()){result.elapsed=now()-b.start;return result;}
        auto d=e.build({core.begin()+split,core.end()},b);
        e.tile_model(a,d,b);
        std::unordered_map<uint64_t,U128> message;
        auto flush=[&]() {
            if(message.empty())return;
            ++result.flushes;
            for(auto [m,w]:message) { ++result.response_evaluations;
                e.response(m,w,live,result.values,b); }
            message.clear();
        };
        e.joins(a,d,b,[&](uint64_t m,U128 w) {
            ++result.unions;
            if(!cache) { ++result.response_evaluations; e.response(m,w,live,result.values,b); }
            else {
                m=std::min(m,e.swap(m)); message[m]+=w;
                result.peak_entries=std::max<uint64_t>(result.peak_entries,message.size());
                if(message.size()>=cache)flush();
            }
        });
        flush();
    } catch(const Cap& c){result.status=c.what(); result.values.clear();}
    result.elapsed=now()-b.start-b.model_seconds; return result;
}
static Result independent(Engine& e,const std::vector<unsigned>& core,const std::vector<unsigned>& demand,
                          Budget budget) {
    Result result(budget);result.values.resize(1u<<e.rows);auto& b=result.budget;
    try {
        size_t split=(core.size()+1)/2;
        auto a=e.build({core.begin(),core.begin()+split},b);
        if(a.empty()){result.elapsed=now()-b.start;return result;}
        std::vector<unsigned> right(core.begin()+split,core.end());
        right.push_back(0);
        for(unsigned last:demand) {right.back()=last;
            auto d=e.build(right,b);e.tile_model(a,d,b);result.values[last]=e.count(a,d,b);}
    }catch(const Cap& c){result.status=c.what();result.values.clear();}
    result.elapsed=now()-b.start-b.model_seconds;return result;
}
static void print_result(const std::string& method,unsigned side,const Result& r) {
    uint64_t nonzero=0;for(U128 value:r.values)nonzero+=value!=0;
    std::cout<<"{\"method\":\""<<method<<"\",\"side\":"<<side<<",\"status\":\""<<r.status
        <<"\",\"seconds\":"<<r.elapsed<<",\"build_steps\":"<<r.budget.build
        <<",\"predicates\":"<<r.budget.predicates<<",\"bucket_pairs\":"<<r.budget.buckets
        <<",\"response_nodes\":"<<r.budget.response_nodes<<",\"unions\":"<<r.unions
        <<",\"build_seconds\":"<<r.budget.build_seconds<<",\"model_seconds\":"<<r.budget.model_seconds
        <<",\"modeled_aggregate_bmma_tiles\":"<<r.budget.modeled_tiles
        <<",\"response_evaluations\":"<<r.response_evaluations<<",\"peak_cache_entries\":"<<r.peak_entries
        <<",\"flushes\":"<<r.flushes<<",\"nonzero_answers\":"
        <<(r.status=="complete"?std::to_string(nonzero):"null")<<"}"<<std::endl;
}
// Fixed last-column boundary, retaining column order. Sort (core row, last bit)
// to obtain a common row gauge and deterministic tie breaking. This does not
// quotient the core under column permutations, complement or transposition.
static std::pair<uint64_t,unsigned> family_key(uint64_t key) {
    std::array<unsigned,8> row{};
    for(unsigned r=0;r<8;++r)row[r]=unsigned(key>>(8*r))&255;
    std::sort(row.begin(),row.end(),[](unsigned a,unsigned b){
        return (a&127)!=(b&127)?(a&127)<(b&127):(a>>7)<(b>>7);});
    uint64_t core=0;unsigned last=0;
    for(unsigned r=0;r<8;++r){core=(core<<7)|(row[r]&127);last|=(row[r]>>7)<<r;}
    return {core,last};
}
}
