// Persistent computation service. Campaign ownership/checkpoints live in
// tools/hafnian/common_core_campaign.py; no research harness is linked here.
#define CORE_OPT_HESS 1
#define CORE_OPT_BOUNDARY 1
#define CORE_OPT_SCRATCH 1
#define CORE_OPT_WARP_POLY 1
#define CORE_OPT_SPARSE_MOMENTS 1
#define CORE_BOUNDARY_ORDER 16
#define CORE_MAX_POOL 11
#define CORE_OPT_INVERSE_CHAIN 1
#define CORE_OPT_LIVE_MOMENTS 1
#define CORE_OPT_SYNC_CLEAR 1
#define CORE_PROFILE 0
#include "hafnian_common_core_runner.cuh"
#include "hafnian_matching_bound.hpp"
#include "hafnian_term_reference.hpp"
#ifndef CORE_HOST_EMULATION
#include "hafnian_residual_engine.cuh"
#endif
#include <iostream>
#include <sstream>
#include <memory>

namespace {
constexpr uint64_t FULL=(UINT64_C(1)<<60)-1;
constexpr uint32_t PRIMES[]={2147483647U,2147483629U,2147483587U,2147483579U};
struct Matrix {
    unsigned n;
    std::vector<uint32_t> data;
    explicit Matrix(unsigned size):n(size),data(size*size){}
    uint32_t& at(unsigned i,unsigned j){return data[i*n+j];}
    uint32_t at(unsigned i,unsigned j)const{return data[i*n+j];}
};
struct Problem {Matrix adjacency;unsigned core,q;std::vector<unsigned> masks;};
struct Member {uint64_t key,removed;};
struct TailQuery : six_by_twenty_nine::Query {unsigned matching_bound_power=0;};
#ifndef CORE_HOST_EMULATION
struct TailCampaign {
    using Query=TailQuery;
    struct Catalog {std::string digest;std::vector<Query> queries;};
    static constexpr unsigned WIDTH=27;
    static constexpr const char* FORMAT="common-core-independent-internal-v1";
    static constexpr const char* CONTROL_FORMAT="unused";
};
#endif

uint32_t run_tail(const TailQuery& query,unsigned slack,unsigned pi,uint64_t begin,
                  uint64_t count,unsigned chunk,double& seconds) {
    uint64_t domain=UINT64_C(1)<<(query.vertices/2-1);
    if(pi>=4||!chunk||chunk>(1u<<20)||!count||begin>=domain||count>domain-begin)
        throw std::runtime_error("invalid independent range");
#ifdef CORE_HOST_EMULATION
    auto started=std::chrono::steady_clock::now();uint32_t sum=0;
    hafnian_reference::Mod mod{PRIMES[pi]};
    for(uint64_t i=begin;i<begin+count;++i)sum=mod.add(sum,hafnian_reference::term(query,i,mod.p));
    seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return sum;
#else
    using Engine=hafnian_residual::Engine<TailCampaign>;
    static Engine engine;
    static Engine::DeviceWorkspace workspace;
    Engine::Task task;task.prime=PRIMES[pi];task.begin=begin;task.end=begin+count;
    Engine::Options options;options.quiet=true;options.report_width=30-slack;options.chunk_terms=chunk;
    TailCampaign::Catalog catalog;catalog.digest="internal-unpublished";
    auto payload=engine.dispatch(query,catalog,task,options,workspace,"internal-unpublished");
    // Consume the proven engine's exact final range sum. Its internal text
    // record is never published; the outer journal owns all result provenance.
    std::istringstream lines(payload);std::string key,line;uint32_t value=0;
    bool got_value=false,got_time=false;
    while(std::getline(lines,line)){
        std::istringstream fields(line);fields>>key;
        if(key=="partial_glynn_sum"){fields>>value;got_value=bool(fields);}
        if(key=="elapsed_seconds"){fields>>seconds;got_time=bool(fields);}
    }
    if(!got_value||!got_time||value>=PRIMES[pi])throw std::runtime_error("invalid independent result");
    return value;
#endif
}

unsigned check_tail() {
    unsigned checks=0;uint64_t random=486;
    for(unsigned n:{42u,48u,50u,58u,62u,64u,66u})for(unsigned mode=0;mode<3;++mode){
        TailQuery query;query.vertices=n;query.adjacency.resize(n*n);
        for(unsigned i=0;i<n;++i)for(unsigned j=0;j<i;++j){
            random=random*UINT64_C(6364136223846793005)+1;
            unsigned edge=mode==0?0:mode==1?unsigned(i==j+n/2):unsigned((random>>32)%3!=0);
            query.adjacency[i*n+j]=query.adjacency[j*n+i]=edge;
        }
        for(unsigned pi=0;pi<4;++pi)for(uint64_t begin:{UINT64_C(0),UINT64_C(7),(UINT64_C(1)<<(n/2-1))-17}){
            double seconds=0;auto got=run_tail(query,3,pi,begin,17,8,seconds);
            hafnian_reference::Mod mod{PRIMES[pi]};uint32_t expected=0;
            for(unsigned i=0;i<17;++i)expected=mod.add(expected,hafnian_reference::term(query,begin+i,mod.p));
            if(got!=expected)throw std::runtime_error("independent GPU/reference mismatch at order "+std::to_string(n));
            ++checks;
        }
    }
    return checks;
}

// Validate the actual embedding, rather than trusting a plan's filename or
// assuming that different occupied masks name the same residual graph.
Problem problem(const six_by_twenty_nine::Geometry& geometry,unsigned slack,
        uint64_t parent,uint64_t boundary,const std::vector<Member>& members,
        std::vector<unsigned>& bounds,std::vector<unsigned>& primes) {
    if(slack<1||slack>3||!parent||(parent&~FULL)||(boundary&~FULL)||
       (parent&boundary)||members.size()<2||members.size()>165)
        throw std::runtime_error("invalid shared group");
    unsigned q=__builtin_popcountll(boundary);
    if(q<5||q>11||(q&1)==0)throw std::runtime_error("invalid boundary pool");
    auto first=members.front().key;
    unsigned d=first>>60,used=__builtin_popcountll(first&FULL);
    if(d>2*slack||used<2*d||used-2*d>2*slack)
        throw std::runtime_error("invalid defect sector");
    unsigned unmatched=2*slack-(used-2*d);
    std::vector<unsigned> vertices;
    for(unsigned i=0;i<60;++i)if(!((parent|boundary)>>i&1))vertices.push_back(i);
    for(unsigned i=0;i<unmatched;++i)vertices.push_back(60+i);
    unsigned core=vertices.size();
    if(core>48||(core&1)||core+q>64)throw std::runtime_error("unsupported shared core");
    for(unsigned i=0;i<60;++i)if(boundary>>i&1)vertices.push_back(i);
    Problem p{Matrix(vertices.size()),core,q,{}};
    for(unsigned i=0;i<vertices.size();++i)for(unsigned j=0;j<i;++j){
        unsigned a=vertices[i],b=vertices[j];bool edge;
        if(a>=60||b>=60)edge=(a>=60)!=(b>=60);
        else {auto [u,v]=geometry.pairs[a%15];auto [x,y]=geometry.pairs[b%15];
            edge=a/15!=b/15&&u!=x&&u!=y&&v!=x&&v!=y;}
        p.adjacency.at(i,j)=p.adjacency.at(j,i)=edge;
    }
    for(const auto& m:members){
        if((m.key>>60)!=d||unsigned(__builtin_popcountll(m.key&FULL))!=used||
           __builtin_popcountll(m.removed)!=3||(m.removed&~boundary)||
           unsigned(__builtin_popcountll(parent|m.removed))!=used||
           six_by_twenty_nine::canonicalize(geometry,parent|m.removed)!=(m.key&FULL))
            throw std::runtime_error("plan member embedding mismatch");
        unsigned mask=0;
        for(unsigned j=0;j<q;++j)if(!(m.removed>>vertices[core+j]&1))mask|=1u<<j;
        p.masks.push_back(mask);
        unsigned bound=hafnian_matching_bound::matching_bound_power(geometry,m.key&FULL,unmatched);
        if(bound>=124)throw std::runtime_error("matching bound exceeds CRT schedule");
        unsigned count=0;unsigned __int128 modulus=1;
        while(count<4&&modulus<=(static_cast<unsigned __int128>(1)<<bound))modulus*=PRIMES[count++];
        if(modulus<=(static_cast<unsigned __int128>(1)<<bound))throw std::runtime_error("insufficient CRT range");
        bounds.push_back(bound);primes.push_back(count);
    }
    return p;
}

template<uint32_t P> std::vector<uint32_t> run(core_gpu::Input& in,uint64_t begin,
        uint64_t count,unsigned chunk,double& seconds){
    // Each field's allocations survive group and checkpoint boundaries.
    static core_gpu::ReducedRunner<P> runner;
    core_gpu::set_field(in,P);
#ifdef CORE_HOST_EMULATION
    constexpr unsigned threads=1;
#else
    constexpr unsigned threads=128;
#endif
    return runner.run(in,begin,count,chunk,threads,seconds);
}
} // namespace

int main(int argc,char** argv)try {
    if(argc>1)throw std::runtime_error("worker takes protocol on stdin; use common_core_campaign.py");
    (void)argv;
#ifdef CORE_HOST_EMULATION
    const char* backend="cpu-reference";
#else
    const char* backend="cuda";
#endif
    std::cout<<"HCCWORKER2 "<<backend<<" exp484-p11-o16-i1-l1-c1-w1-s1-h1-b1-a1-t128-tail1 "
             <<sha256_file("/proc/self/exe")<<std::endl;
    six_by_twenty_nine::Geometry geometry;
    std::unique_ptr<Problem> prepared;
    std::unique_ptr<core_gpu::Input> input;
    std::unique_ptr<TailQuery> tail;
    unsigned current_slack=0;
    std::vector<unsigned> required,bounds,last_active;
    std::string line;
    while(std::getline(std::cin,line)){
        std::istringstream fields(line);std::string command,trailing;fields>>command;
        if(command=="prepare"){
            unsigned slack,size;uint64_t parent,boundary;
            if(!(fields>>slack>>parent>>boundary>>size)||size<2||size>165)
                throw std::runtime_error("bad prepare request");
            std::vector<Member> members(size);
            for(auto& m:members)if(!(fields>>m.key>>m.removed))throw std::runtime_error("truncated members");
            if(fields>>trailing)throw std::runtime_error("trailing prepare fields");
            bounds.clear();required.clear();last_active.clear();input.reset();
            prepared=std::make_unique<Problem>(problem(geometry,slack,parent,boundary,members,bounds,required));
            tail.reset();current_slack=slack;
            uint64_t domain=prepared->core?UINT64_C(1)<<(prepared->core/2-1):1;
            std::cout<<"prepared "<<domain<<" "<<bounds.size();
            for(unsigned i=0;i<bounds.size();++i)std::cout<<" "<<bounds[i]<<" "<<required[i];
            std::cout<<std::endl;
        }else if(command=="prepare_single"){
            unsigned slack;uint64_t key;
            if(!(fields>>slack>>key)||(fields>>trailing)||slack<1||slack>3)
                throw std::runtime_error("invalid independent preparation");
            unsigned d=key>>60,used=__builtin_popcountll(key&FULL);
            if(d>2*slack||used<2*d||used-2*d>2*slack)
                throw std::runtime_error("invalid independent sector");
            tail=std::make_unique<TailQuery>();tail->occupied=key&FULL;
            tail->defect_count=d;tail->excess=used-2*d;tail->unmatched=2*slack-tail->excess;
            tail->matching_bound_power=hafnian_matching_bound::matching_bound_power(geometry,tail->occupied,tail->unmatched);
            unsigned bound=tail->matching_bound_power,count=0;unsigned __int128 modulus=1;
            if(bound>=124)throw std::runtime_error("matching bound outside CRT schedule");
            while(count<4&&modulus<=(static_cast<unsigned __int128>(1)<<bound))modulus*=PRIMES[count++];
            if(modulus<=(static_cast<unsigned __int128>(1)<<bound))throw std::runtime_error("insufficient CRT range");
            six_by_twenty_nine::build_query_graph(geometry,*tail,30-slack,true);
            prepared.reset();input.reset();last_active.clear();required={count};current_slack=slack;
            std::cout<<"prepared "<<(UINT64_C(1)<<(tail->vertices/2-1))<<" 1 "<<bound<<" "<<count<<std::endl;
        }else if(command=="check_tail"){
            auto checks=check_tail();std::cout<<"checked "<<checks<<std::endl;
        }else if(command=="run"){
            unsigned pi,chunk;uint64_t begin,count;
            if((!prepared&&!tail)||!(fields>>pi>>begin>>count>>chunk)||pi>=4||(fields>>trailing))
                throw std::runtime_error("bad run request");
            std::vector<unsigned> active;
            for(unsigned i=0;i<required.size();++i)if(required[i]>pi)active.push_back(i);
            if(active.empty())throw std::runtime_error("no active children for field");
            if(tail){double seconds=0;auto sum=run_tail(*tail,current_slack,pi,begin,count,chunk,seconds);
                std::cout.precision(17);std::cout<<"result "<<seconds<<" 1 "<<sum<<std::endl;continue;}
            if(active!=last_active){auto selected=*prepared;selected.masks.clear();
                for(auto i:active)selected.masks.push_back(prepared->masks[i]);
                input=std::make_unique<core_gpu::Input>(core_gpu::prepare(selected));last_active=active;}
            double seconds=0;std::vector<uint32_t> sums;
            switch(pi){
                case 0:sums=run<2147483647U>(*input,begin,count,chunk,seconds);break;
                case 1:sums=run<2147483629U>(*input,begin,count,chunk,seconds);break;
                case 2:sums=run<2147483587U>(*input,begin,count,chunk,seconds);break;
                case 3:sums=run<2147483579U>(*input,begin,count,chunk,seconds);break;
            }
            std::cout.precision(17);std::cout<<"result "<<seconds<<" "<<sums.size();
            for(auto value:sums)std::cout<<" "<<value;
            std::cout<<std::endl;
        }else if(command=="quit")return 0;
        else throw std::runtime_error("unknown worker command");
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"common-core worker: "<<e.what()<<'\n';return 1;}
