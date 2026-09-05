// Reuse the independent research finite-field Hessenberg/Lanczos controls.
// No maintained CUDA or production checkpoint code is changed by this probe.
#define main historical_gray_probe_main
#include "hafnian_gray_update_probe.cpp"
#undef main
#include "six_by_twenty_seven_common_core.hpp"
#include <fstream>
#include <sstream>
#include <set>
#include <omp.h>

namespace common_bench {
using six_by_common_core::bits;
using six_by_common_core::full;
using six_by_common_core::hash;

struct Input {
    unsigned e=0,d=0,cap=0,core=0;
    unsigned prime_index=UINT32_MAX;
    uint64_t root=0,parent=0,boundary=0;
    std::vector<std::pair<uint64_t,uint64_t>> members;
};

std::vector<Input> read_groups(const std::string& path) {
    std::ifstream file(path);
    if(!file)throw std::runtime_error("cannot read census log");
    std::vector<Input> out;
    for(std::string line;std::getline(file,line);) {
        if(line.rfind("CORE27_GROUP ",0))continue;
        std::istringstream fields(line);std::string word;fields>>word;
        Input v;
        unsigned reported=0;
        while(fields>>word) {
            const size_t split=word.find('=');
            const auto key=word.substr(0,split),value=word.substr(split+1);
            if(key=="e")v.e=std::stoul(value);
            else if(key=="d")v.d=std::stoul(value);
            else if(key=="cap")v.cap=std::stoul(value);
            else if(key=="root")v.root=std::stoull(value);
            else if(key=="parent")v.parent=std::stoull(value);
            else if(key=="boundary")v.boundary=std::stoull(value);
            else if(key=="core")v.core=std::stoul(value);
            else if(key=="queries")reported=std::stoul(value);
            else if(key=="prime_index")v.prime_index=std::stoul(value);
            else if(key=="members") {
                std::istringstream list(value);
                for(std::string item;std::getline(list,item,',');) {
                    if(item.empty())continue;
                    auto colon=item.find(':');
                    v.members.emplace_back(std::stoull(item.substr(0,colon)),
                                           std::stoull(item.substr(colon+1)));
                }
            }
        }
        if(reported!=v.members.size() || v.e>6 || (v.parent&v.boundary))
            throw std::runtime_error("invalid group record");
        out.push_back(std::move(v));
    }
    return out;
}

bool adjacent(const six_by_twenty_nine::Geometry& geometry,unsigned a,unsigned b) {
    if(a==b)return false;
    if(a>=60||b>=60)return (a<60)!=(b<60);
    if(a/15==b/15)return false;
    const auto [u,v]=geometry.pairs[a%15];const auto [x,y]=geometry.pairs[b%15];
    return u!=x&&u!=y&&v!=x&&v!=y;
}

struct Problem {
    Matrix adjacency;
    unsigned core=0,q=0;
    std::vector<unsigned> masks;
};

Problem make_problem(const six_by_twenty_nine::Geometry& geometry,const Input& in,unsigned slack=3) {
    if(in.e>2*slack)throw std::runtime_error("excess exceeds slack");
    std::vector<unsigned> vertices;
    uint64_t live=full&~(in.parent|in.boundary);
    for(unsigned i=0;i<60;++i)if(live&(UINT64_C(1)<<i))vertices.push_back(i);
    for(unsigned i=0;i<2*slack-in.e;++i)vertices.push_back(60+i);
    if(vertices.size()!=in.core || (in.core&1))throw std::runtime_error("bad core order");
    for(unsigned i=0;i<60;++i)if(in.boundary&(UINT64_C(1)<<i))vertices.push_back(i);
    Problem p{Matrix(unsigned(vertices.size())),in.core,bits(in.boundary),{}};
    for(unsigned i=0;i<vertices.size();++i)for(unsigned j=0;j<vertices.size();++j)
        p.adjacency.at(i,j)=adjacent(geometry,vertices[i],vertices[j]);
    std::set<uint64_t> ids;
    for(const auto& [key,removed]:in.members) {
        if(bits(removed)!=3 || (removed&~in.boundary) || !ids.insert(key).second ||
           six_by_twenty_nine::canonicalize(geometry,in.parent|removed)!=key)
            throw std::runtime_error("invalid canonical member");
        unsigned mask=0;
        for(unsigned j=0;j<p.q;++j)
            if(!(removed&(UINT64_C(1)<<vertices[p.core+j])))mask|=1u<<j;
        p.masks.push_back(mask);
    }
    return p;
}

struct Times { double core=0,moments=0,boundary=0; };
double seconds(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double>(b-a).count();}

// Persistent polynomial workspace: only reachable even boundary subsets are
// materialized. Its dependency graph is constructed once, outside sign loops.
struct Workspace {
    unsigned c,q,m,stride;
    std::vector<unsigned> plan;
    std::vector<uint32_t> memo,k,power,next,f,inverse;
    explicit Workspace(const Problem& p,const Mod& mod):c(p.core),q(p.q),m(c/2),stride(m+1),
        memo(size_t(1u<<q)*stride),k(size_t(q)*q*stride),power(c*q),next(c*q),
        f(stride),inverse(2*m+1) {
        if(q>11)throw std::runtime_error("boundary pool exceeds bounded gate");
        for(unsigned i=1;i<inverse.size();++i)inverse[i]=mod.inverse(i);
        std::vector<bool> seen(1u<<q);
        auto visit=[&](auto&& self,unsigned mask)->void {
            if(seen[mask])return;seen[mask]=true;
            if(mask) {
                unsigned first=unsigned(__builtin_ctz(mask)),rest=mask^(1u<<first);
                for(unsigned candidates=rest;candidates;candidates&=candidates-1)
                    self(self,rest^(candidates&-candidates));
            }
            plan.push_back(mask);
        };
        for(unsigned mask:p.masks)visit(visit,mask);
    }
};

std::vector<uint32_t> shared_term(const Problem& p,Workspace& w,uint64_t signs,
                                const Mod& mod,Times* times=nullptr) {
    auto started=Clock::now();
    auto sign=[&](unsigned row){return row/2==0 || (signs&(UINT64_C(1)<<(row/2-1)));};
    Matrix matrix(w.c);
    for(unsigned i=0;i<w.c;++i)for(unsigned j=0;j<w.c;++j)
        matrix.at(i,j)=p.adjacency.at(i^1,j)?(sign(i)?1:mod.p-1):0;
    const auto chi=hessenberg_coefficients(std::move(matrix),w.m,mod);
    w.f[0]=1;
    for(unsigned d=1;d<=w.m;++d) {
        uint32_t value=0;
        for(unsigned j=1;j<=d;++j)value=mod.add(value,
            mod.mul(2*d-j,mod.mul(chi[j],w.f[d-j])));
        w.f[d]=mod.neg(mod.mul(value,w.inverse[2*d]));
    }
    auto core_done=Clock::now();
    std::fill(w.k.begin(),w.k.end(),0);
    for(unsigned i=0;i<w.q;++i)for(unsigned j=0;j<w.q;++j)
        w.k[(size_t(i)*w.q+j)*w.stride]=p.adjacency.at(w.c+i,w.c+j);
    for(unsigned v=0;v<w.c;++v)for(unsigned j=0;j<w.q;++j)
        w.power[v*w.q+j]=p.adjacency.at(v^1,w.c+j)?(sign(v)?1:mod.p-1):0;
    for(unsigned d=1;d<=w.m;++d) {
        for(unsigned i=0;i<w.q;++i)for(unsigned j=i+1;j<w.q;++j) {
            uint32_t value=0;
            for(unsigned v=0;v<w.c;++v)if(p.adjacency.at(w.c+i,v))
                value=mod.add(value,w.power[v*w.q+j]);
            w.k[(size_t(i)*w.q+j)*w.stride+d]=
                w.k[(size_t(j)*w.q+i)*w.stride+d]=value;
        }
        if(d==w.m)break;
        std::fill(w.next.begin(),w.next.end(),0);
        for(unsigned v=0;v<w.c;++v) {
            for(unsigned u=0;u<w.c;++u)if(p.adjacency.at(v^1,u))
                for(unsigned j=0;j<w.q;++j)w.next[v*w.q+j]=
                    mod.add(w.next[v*w.q+j],w.power[u*w.q+j]);
            if(!sign(v))for(unsigned j=0;j<w.q;++j)w.next[v*w.q+j]=mod.neg(w.next[v*w.q+j]);
        }
        w.next.swap(w.power);
    }
    auto moments_done=Clock::now();
    for(unsigned mask:w.plan) {
        uint32_t* out=w.memo.data()+size_t(mask)*w.stride;
        std::fill(out,out+w.stride,0);
        if(!mask){out[0]=1;continue;}
        unsigned first=unsigned(__builtin_ctz(mask)),rest=mask^(1u<<first);
        for(unsigned candidates=rest;candidates;candidates&=candidates-1) {
            unsigned bit=candidates&-candidates,second=unsigned(__builtin_ctz(bit));
            const uint32_t* left=w.k.data()+(size_t(first)*w.q+second)*w.stride;
            const uint32_t* right=w.memo.data()+size_t(rest^bit)*w.stride;
            for(unsigned d=0;d<=w.m;++d) {
                uint32_t sum=0;uint64_t pending=0;unsigned count=0;
                for(unsigned j=0;j<=d;++j) {
                    pending+=uint64_t(left[j])*right[d-j];
                    // Four products of residues below 2^31 fit uint64_t.
                    if(++count==4){sum=mod.add(sum,uint32_t(pending%mod.p));pending=0;count=0;}
                }
                out[d]=mod.add(out[d],mod.add(sum,uint32_t(pending%mod.p)));
            }
        }
    }
    std::vector<uint32_t> result;
    const bool negative=w.m && ((w.m-1-unsigned(__builtin_popcountll(signs)))&1);
    for(unsigned mask:p.masks) {
        const uint32_t* value=w.memo.data()+size_t(mask)*w.stride;
        uint32_t sum=0;
        for(unsigned d=0;d<=w.m;++d)sum=mod.add(sum,mod.mul(w.f[d],value[w.m-d]));
        result.push_back(negative?mod.neg(sum):sum);
    }
    auto done=Clock::now();
    if(times){times->core+=seconds(started,core_done);times->moments+=seconds(core_done,moments_done);
              times->boundary+=seconds(moments_done,done);}
    return result;
}

Problem single_problem(const Problem& p,unsigned mask) {
    std::vector<unsigned> ids(p.core);std::iota(ids.begin(),ids.end(),0);
    for(unsigned i=0;i<p.q;++i)if(mask&(1u<<i))ids.push_back(p.core+i);
    Problem s{Matrix(unsigned(ids.size())),p.core,unsigned(ids.size())-p.core,{}};
    s.masks.push_back((1u<<s.q)-1);
    for(unsigned i=0;i<ids.size();++i)for(unsigned j=0;j<ids.size();++j)
        s.adjacency.at(i,j)=p.adjacency.at(ids[i],ids[j]);
    return s;
}

six_by_twenty_eight::Query reference_query(const Problem& p,unsigned mask) {
    auto single=single_problem(p,mask);
    unsigned n=single.adjacency.n;
    six_by_twenty_eight::Query query;query.vertices=n;query.adjacency.resize(n*n);
    // Existing control pairs first-half vertices with second-half vertices.
    for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j) {
        unsigned a=i<n/2?2*i:2*(i-n/2)+1,b=j<n/2?2*j:2*(j-n/2)+1;
        query.adjacency[i*n+j]=uint8_t(single.adjacency.at(a,b));
    }
    return query;
}

uint32_t brute(const Matrix& a,uint64_t mask,const Mod& mod) {
    if(!mask)return 1;
    unsigned first=unsigned(__builtin_ctzll(mask));uint64_t rest=mask&(mask-1);
    uint32_t sum=0;
    for(uint64_t candidates=rest;candidates;candidates&=candidates-1) {
        uint64_t bit=candidates&-candidates;
        if(a.at(first,unsigned(__builtin_ctzll(bit))))sum=mod.add(sum,brute(a,rest^bit,mod));
    }
    return sum;
}

void self_test() {
    uint64_t checks=0;std::mt19937_64 random(476);
    for(uint32_t prime:{1000003u,1000033u}) {
        Mod mod{prime};
        for(unsigned c:{0u,2u,4u,6u,8u})for(unsigned q:{5u,7u})for(unsigned mode=0;mode<3;++mode) {
            Problem p{Matrix(c+q),c,q,{}};
            for(unsigned mask=0;mask<(1u<<q);++mask)if(bits(mask)==q-3)p.masks.push_back(mask);
            for(unsigned i=0;i<c+q;++i)for(unsigned j=0;j<i;++j)
                p.adjacency.at(i,j)=p.adjacency.at(j,i)=
                    mode==0?1:(mode==1&&i<c?0:unsigned(random()%2));
            Workspace w(p,mod);std::vector<uint32_t> total(p.masks.size());
            uint64_t terms=UINT64_C(1)<<(c?c/2-1:0);
            for(uint64_t i=0;i<terms;++i) {
                auto value=shared_term(p,w,i^(i>>1),mod);
                for(unsigned j=0;j<value.size();++j)total[j]=mod.add(total[j],value[j]);
            }
            for(unsigned j=0;j<p.masks.size();++j) {
                auto s=single_problem(p,p.masks[j]);
                uint32_t want=brute(s.adjacency,(UINT64_C(1)<<s.adjacency.n)-1,mod);
                if(mod.mul(total[j],mod.inverse(uint32_t(terms)))!=want)
                    throw std::runtime_error("complete shared/brute mismatch");
                const auto query=reference_query(p,p.masks[j]);
                unsigned half=query.vertices/2;uint64_t count=UINT64_C(1)<<(half-1);
                uint32_t reference=0;
                for(uint64_t k=0;k<count;++k)reference=mod.add(reference,term_from_coefficients(
                    hessenberg_coefficients(build_signed_matrix(query,k,mod),half,mod),k,half,mod));
                if(mod.mul(reference,mod.inverse(uint32_t(count)))!=want)
                    throw std::runtime_error("complete independent/brute mismatch");
                ++checks;
            }
        }
    }
    std::printf("CORE_BENCH_SELF_TEST complete_minors=%llu primes=2 exact=OK\n",(unsigned long long)checks);
}

// Existing CPU four-term resolvent reference. Not the optimized CUDA
// eight-term hybrid; CPU ratios must not be reported as GPU speedups.
std::vector<uint32_t> resolver4(const SymmetricRankFactor& factor,unsigned half,
        uint64_t start,const Mod& mod,std::mt19937_64& random) {
    const auto reduced=build_reduced_matrix(factor,start^(start>>1),mod);
    auto base=generalized_lanczos(factor.rank,factor.metric,[&](const auto& x,auto& y){
        apply_dense(reduced,x,y,mod);},mod,random,4);
    RankMatrix factors(factor.rank,6);std::vector<uint32_t> deltas(3);
    for(unsigned stage=1;stage<4;++stage) {
        uint64_t index=start+stage,signs=index^(index>>1);
        unsigned edge=unsigned(__builtin_ctzll(index))+1;
        std::array<uint32_t,16> kernel{};
        auto v=make_reduced_update_factor(factor,edge,mod,kernel,signs&(UINT64_C(1)<<(edge-1)));
        auto transformed=inverse_basis_apply(base.basis,factor.metric,base.inverse_metric,v,mod);
        for(unsigned i=0;i<factor.rank;++i)for(unsigned j=0;j<2;++j)
            factors.at(i,2*(stage-1)+j)=transformed.at(i,j);
        deltas[stage-1]=kernel[1];
    }
    auto correction=resolvent_correction_matrix(base.tridiagonal,base.metric,factors,deltas,half,mod);
    auto determinants=leading_principal_determinants(std::move(correction),6,half,mod);
    auto chi=tridiagonal_coefficients(base.tridiagonal,half,mod);
    std::vector<uint32_t> out;
    for(unsigned stage=0;stage<4;++stage) {
        auto characteristic=stage?polynomial_multiply_truncated(chi,determinants[stage-1],half,mod):chi;
        uint64_t index=start+stage;
        out.push_back(term_from_coefficients(characteristic,index^(index>>1),half,mod));
    }
    return out;
}

void benchmark(const Problem& p,const Input& input,unsigned steps,unsigned repeats,const Mod& mod) {
    const unsigned n=p.core+bits(p.masks.front()),tail=n-p.core;
    const uint64_t common_domain=UINT64_C(1)<<(p.core/2-1),domain=UINT64_C(1)<<(n/2-1);
    if(!steps || steps%4 || steps>common_domain || steps>domain)throw std::runtime_error("invalid steps");
    const uint64_t start=(hash(input.root)%(common_domain-steps+1))&~UINT64_C(7);
    Workspace workspace(p,mod);
    std::vector<six_by_twenty_eight::Query> queries;
    std::vector<SymmetricRankFactor> factors;
    std::vector<uint64_t> reference_starts;
    std::vector<Problem> singles;std::vector<Workspace> single_workspaces;
    auto setup=Clock::now();
    for(unsigned mask:p.masks) {
        queries.push_back(reference_query(p,mask));
        reference_starts.push_back((hash(input.root^uint64_t(mask))%(domain-steps+1))&~UINT64_C(7));
        factors.push_back(factor_adjacency(queries.back(),mod));
        singles.push_back(single_problem(p,mask));
        single_workspaces.emplace_back(singles.back(),mod);
    }
    double setup_seconds=seconds(setup,Clock::now());
    std::vector<std::vector<uint32_t>> ref(p.masks.size(),std::vector<uint32_t>(steps));
    std::vector<std::vector<uint32_t>> common(steps);
    double shared_seconds=0,hess_seconds=0,resolver_seconds=0,single_seconds=0;
    unsigned fallback_blocks=0;Times times;
    for(unsigned rep=0;rep<repeats;++rep) {
        std::mt19937_64 random(476);
        auto run_shared=[&]{auto t=Clock::now();
            for(unsigned i=0;i<steps;++i){uint64_t k=start+i;
                common[i]=shared_term(p,workspace,k^(k>>1),mod,&times);}
            shared_seconds+=seconds(t,Clock::now());};
        auto run_reference=[&]{
            auto t=Clock::now();
            for(unsigned j=0;j<queries.size();++j)for(unsigned i=0;i<steps;++i) {
                uint64_t k=reference_starts[j]+i,signs=k^(k>>1);
                ref[j][i]=term_from_coefficients(hessenberg_coefficients(
                    build_signed_matrix(queries[j],signs,mod),n/2,mod),signs,n/2,mod);
            }
            hess_seconds+=seconds(t,Clock::now());
            for(unsigned j=0;j<queries.size();++j)for(unsigned i=0;i<steps;i+=4) {
                t=Clock::now();std::vector<uint32_t> actual;
                try{actual=resolver4(factors[j],n/2,reference_starts[j]+i,mod,random);}
                catch(const std::runtime_error& error){
                    if(std::string(error.what()).find("generalized Lanczos breakdown")!=0)throw;
                    ++fallback_blocks;
                    for(unsigned k=0;k<4;++k){uint64_t index=reference_starts[j]+i+k,signs=index^(index>>1);
                        actual.push_back(term_from_coefficients(hessenberg_coefficients(
                            build_signed_matrix(queries[j],signs,mod),n/2,mod),signs,n/2,mod));}
                }
                resolver_seconds+=seconds(t,Clock::now());
                for(unsigned k=0;k<4;++k)if(actual[k]!=ref[j][i+k])
                    throw std::runtime_error("resolvent/reference mismatch");
            }
        };
        if(rep&1){run_reference();run_shared();}else{run_shared();run_reference();}
        // Equal partial-core summands allow a real-order per-sign check.
        // They must NOT be compared to full Glynn summands above.
        auto t=Clock::now();
        for(unsigned j=0;j<singles.size();++j)for(unsigned i=0;i<steps;++i) {
            uint64_t index=start+i;
            auto got=shared_term(singles[j],single_workspaces[j],index^(index>>1),mod);
            if(got[0]!=common[i][j])throw std::runtime_error("shared/individual partial-core mismatch");
        }
        single_seconds+=seconds(t,Clock::now());
    }
    double conversion=std::ldexp(1.0,int(tail/2));
    std::printf("CORE_BENCH e=%u d=%u root=%llu order=%u core=%u tail=%u pool=%u queries=%zu"
                " core_samples=%u repeats=%u prime=%u setup_s=%.6f shared_s=%.6f"
                " char_s=%.6f moment_s=%.6f boundary_s=%.6f individual_partial_s=%.6f"
                " independent_hessenberg_s=%.6f cpu_resolver4_s=%.6f fallback_blocks=%u"
                " normalized_hess_ratio=%.6f normalized_resolver4_ratio=%.6f"
                " within_method_reuse=%.6f boundary_states=%zu exact_ranges=OK\n",
        input.e,input.d,(unsigned long long)input.root,n,p.core,tail,p.q,p.masks.size(),steps,repeats,mod.p,
        setup_seconds,shared_seconds,times.core,times.moments,times.boundary,single_seconds,hess_seconds,
        resolver_seconds,fallback_blocks,hess_seconds*conversion/shared_seconds,
        resolver_seconds*conversion/shared_seconds,single_seconds/shared_seconds,workspace.plan.size());
    std::fflush(stdout);
}
// Complete a real group, independently of production full-Glynn ranges.
void complete(const Input& in,unsigned threads,const Mod& mod) {
    six_by_twenty_nine::Geometry geometry;
    auto catalog=six_by_twenty_eight::build_catalog();
    auto p=make_problem(geometry,in,2);
    if(p.core>42)throw std::runtime_error("complete CPU gate exceeds bounded core order");
    uint64_t terms=p.core?UINT64_C(1)<<(p.core/2-1):1;
    std::vector<std::vector<uint32_t>> partial(threads,std::vector<uint32_t>(p.masks.size()));
    auto started=Clock::now();
    #pragma omp parallel num_threads(threads)
    {
        unsigned tid=unsigned(omp_get_thread_num());Workspace w(p,mod);
        #pragma omp for schedule(static)
        for(uint64_t i=0;i<terms;++i) {
            auto values=shared_term(p,w,i^(i>>1),mod);
            for(unsigned j=0;j<values.size();++j)partial[tid][j]=mod.add(partial[tid][j],values[j]);
        }
    }
    for(unsigned j=0;j<p.masks.size();++j) {
        uint32_t sum=0;for(const auto& v:partial)sum=mod.add(sum,v[j]);
        uint32_t residue=mod.mul(sum,mod.inverse(uint32_t(terms%mod.p)));
        auto it=std::find_if(catalog.queries.begin(),catalog.queries.end(),[&](const auto& q){
            return q.occupied==in.members[j].first&&q.excess==in.e&&q.defect_count==in.d;});
        if(it==catalog.queries.end())throw std::runtime_error("complete query absent from catalog");
        std::printf("CORE_COMPLETE query_id=%u occupied=%llu prime=%u augmented_hafnian=%u query_digest=%s\n",
            it->id,(unsigned long long)it->occupied,mod.p,residue,it->digest.c_str());
    }
    std::printf("CORE_COMPLETE_SUMMARY queries=%zu core=%u pool=%u terms=%llu threads=%u seconds=%.6f catalog=%s\n",
        p.masks.size(),p.core,p.q,(unsigned long long)terms,threads,seconds(started,Clock::now()),catalog.digest.c_str());
}

// Actual once-only assignment of a complete known sector, not independently
// picked best groups. Coefficients belong to queries and are never multiplied
// by the number of embeddings/families in which the query appears.
void coverage() {
    using namespace six_by_common_core;
    six_by_twenty_nine::Geometry geometry;
    auto catalog=six_by_twenty_eight::build_catalog();
    std::vector<uint64_t> triples;
    for(const auto& s:six_by_twenty_nine::weighted_supports(geometry))if(s.excess==1)triples.push_back(s.mask);
    std::sort(triples.begin(),triples.end());
    std::map<uint64_t,uint64_t> targets;
    std::vector<Family> families;
    auto started=Clock::now();
    for(const auto& q:catalog.queries) {
        if(q.excess==4&&q.defect_count==4)targets.emplace(q.occupied,q.defect_coefficient);
        if(q.excess==3&&q.defect_count==3)families.push_back(build_family(geometry,q.occupied,triples));
    }
    for(const auto& f:families)for(const auto& c:f.children)
        if(!targets.count(c.canonical))throw std::runtime_error("family child outside complete sector");
    if(targets.size()!=33077||families.size()!=664)throw std::runtime_error("coverage sector census mismatch");
    std::printf("CORE_COVER_SETUP targets=%zu parents=%zu seconds=%.6f\n",targets.size(),families.size(),seconds(started,Clock::now()));
    for(unsigned cap:{7u,9u,11u})for(unsigned seed:{476u,477u}) {
        auto order=families;
        std::sort(order.begin(),order.end(),[&](const auto& a,const auto& b){
            return std::make_pair(hash(a.parent^seed),a.parent)<std::make_pair(hash(b.parent^seed),b.parent);});
        started=Clock::now();std::set<uint64_t> owned;
        uint64_t coefficient=0;unsigned groups=0,largest=0;std::map<unsigned,unsigned> histogram;
        for(auto& f:order) {
            std::set<uint64_t> tried;
            for(;;) {
                f.children.erase(std::remove_if(f.children.begin(),f.children.end(),[&](const auto& c){return owned.count(c.canonical);}),f.children.end());
                auto anchor=std::find_if(f.children.begin(),f.children.end(),[&](const auto& c){return !tried.count(c.removed);});
                if(anchor==f.children.end())break;
                tried.insert(anchor->removed);
                auto g=grow(f,anchor->removed,cap);
                if(g.size()<2)continue;
                ++groups;largest=std::max(largest,g.size());++histogram[g.size()];
                validate(f,g,g.children.front().canonical,48,0);
                for(const auto& c:g.children) {
                    if(!owned.insert(c.canonical).second)throw std::runtime_error("duplicate query ownership");
                    coefficient+=targets.at(c.canonical);
                }
            }
        }
        for(const auto& [key,weight]:targets)if(!owned.count(key))coefficient+=weight;
        if(coefficient!=UINT64_C(8126516160))throw std::runtime_error("lost defect coefficient");
        std::printf("CORE_COVER cap=%u seed=%u targets=%zu grouped=%zu groups=%u singletons=%zu mean=%.6f max=%u coverage=%.6f seconds=%.6f coefficient=%llu exact_once=OK sizes=",
            cap,seed,targets.size(),owned.size(),groups,targets.size()-owned.size(),double(owned.size())/groups,largest,
            double(owned.size())/targets.size(),seconds(started,Clock::now()),(unsigned long long)coefficient);
        for(auto [size,count]:histogram)std::printf("%u:%u,",size,count);
        std::printf("\n");std::fflush(stdout);
    }
}
} // namespace common_bench

#ifndef COMMON_CORE_NO_MAIN
int main(int argc,char** argv)try {
    std::string path;unsigned steps=64,repeats=2,per_sector=1,query_limit=0;uint32_t prime=2147483647U;
    bool test=false,cover=false,full_group=false;unsigned threads=8;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--self-test")test=true;
        else if(a=="--coverage6x28")cover=true;
        else if(a=="--complete6x28")full_group=true;
        else if(a=="--threads"&&i+1<argc)threads=unsigned(number(argv[++i]));
        else if(a=="--groups"&&i+1<argc)path=argv[++i];
        else if(a=="--steps"&&i+1<argc)steps=unsigned(number(argv[++i]));
        else if(a=="--repeats"&&i+1<argc)repeats=unsigned(number(argv[++i]));
        else if(a=="--per-sector"&&i+1<argc)per_sector=unsigned(number(argv[++i]));
        else if(a=="--query-limit"&&i+1<argc)query_limit=unsigned(number(argv[++i]));
        else if(a=="--prime"&&i+1<argc)prime=uint32_t(number(argv[++i]));
        else throw std::runtime_error("usage: common-core-bench [--self-test] [--coverage6x28] [--groups LOG --steps N --repeats N --per-sector N --query-limit N --prime P] [--complete6x28 --groups LOG --threads N --prime P]");
    }
    if(prime!=2147483647U&&prime!=2147483629U)throw std::runtime_error("benchmark prime not certified");
    if(test)common_bench::self_test();
    if(cover){common_bench::coverage();return 0;}
    if(!threads||threads>16)throw std::runtime_error("threads must be 1..16");
    if(path.empty())return test?0:2;
    if(!repeats||!per_sector)throw std::runtime_error("empty benchmark");
    auto groups=common_bench::read_groups(path);
    if(full_group) {
        if(groups.empty())throw std::runtime_error("no complete groups");
        auto it=std::max_element(groups.begin(),groups.end(),[](const auto& a,const auto& b){return a.members.size()<b.members.size();});
        common_bench::complete(*it,threads,Mod{prime});return 0;
    }
    std::sort(groups.begin(),groups.end(),[](const auto& a,const auto& b){
        return std::make_pair(six_by_common_core::hash(a.root),a.cap)<
               std::make_pair(six_by_common_core::hash(b.root),b.cap);});
    std::map<std::tuple<unsigned,unsigned,unsigned>,unsigned> counts;
    six_by_twenty_nine::Geometry geometry;
    for(auto in:groups) {
        unsigned order=66-2*in.e-2*in.d;
        if(order<42||order>46||in.cap<7)continue;
        if(counts[{in.e,in.d,in.cap}]++>=per_sector)continue;
        // Sensitivity check for smaller assigned groups; this is a trimmed
        // sample, not a claim to be a full 6x27 ownership plan.
        if(query_limit&&in.members.size()>query_limit)in.members.resize(query_limit);
        common_bench::benchmark(common_bench::make_problem(geometry,in),in,steps,repeats,Mod{prime});
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"error: %s\n",e.what());return 1;}
#endif
