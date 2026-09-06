// Bounded-memory, exact-once research assignment. No production result IDs.
#include "common_core_catalog_io.hpp"
#include "six_by_twenty_seven_common_core.hpp"
#include "common_core_cost.hpp"
#include <omp.h>
#include <chrono>
#include <numeric>
#include <tuple>
#include <memory>

namespace {
using namespace six_by_common_core;
using Entry=common_catalog::Entry;
using U128=unsigned __int128;
using Clock=std::chrono::steady_clock;
std::string decimal(U128 x){if(!x)return "0";std::string s;while(x){s.push_back(char('0'+x%10));x/=10;}std::reverse(s.begin(),s.end());return s;}
unsigned defects(uint64_t key){return unsigned(key>>60);}
unsigned excess(uint64_t key){return bits(key&full)-2*defects(key);}
unsigned order(uint64_t key,unsigned slack){return 60+2*slack-2*excess(key)-2*defects(key);}
struct Index {
    std::vector<uint64_t> keys;std::vector<uint32_t> ids;size_t mask;
    explicit Index(const std::vector<Entry>& rows){size_t size=1;while(size<rows.size()*2)size*=2;
        mask=size-1;keys.assign(size,UINT64_MAX);ids.resize(size);
        for(size_t i=0;i<rows.size();++i){size_t slot=hash(rows[i].key)&mask;
            while(keys[slot]!=UINT64_MAX){if(keys[slot]==rows[i].key)throw std::runtime_error("duplicate catalog key");slot=(slot+1)&mask;}
            keys[slot]=rows[i].key;ids[slot]=uint32_t(i);}
    }
    uint32_t at(uint64_t key)const{size_t slot=hash(key)&mask;
        while(keys[slot]!=key){if(keys[slot]==UINT64_MAX)throw std::runtime_error("child missing from catalog");slot=(slot+1)&mask;}return ids[slot];}
};
struct Totals {uint64_t queries=0,groups=0;U128 coefficient=0,original=0,signs=0;};

void verify(const std::string& path,const std::vector<Entry>& rows,unsigned slack,const std::string& digest,bool maps) {
    common_catalog::File f(path,false);char magic[8],identity[64];f.get(magic,8);
    if(std::memcmp(magic,"HCPLAN01",8)||f.get64()!=slack)throw std::runtime_error("plan header mismatch");
    unsigned cap=unsigned(f.get64());f.get64(); // seed
    if(cap!=7&&cap!=9&&cap!=11)throw std::runtime_error("unsupported plan cap");
    if(f.get64()!=rows.size())throw std::runtime_error("plan query count mismatch");
    f.get(identity,64);if(std::string(identity,64)!=digest)throw std::runtime_error("plan/catalog digest mismatch");
    std::vector<uint8_t> seen(rows.size());uint64_t groups=0,queries=0;U128 coefficient=0;
    six_by_twenty_nine::Geometry geometry;
    std::vector<std::pair<uint64_t,uint64_t>> embeddings;embeddings.reserve(4096);
    auto flush=[&]{unsigned failures=0;
        #pragma omp parallel for reduction(+:failures) schedule(static)
        for(size_t i=0;i<embeddings.size();++i)
            failures+=six_by_twenty_nine::canonicalize(geometry,embeddings[i].first)!=embeddings[i].second;
        if(failures)throw std::runtime_error("incorrect canonical residual embedding");embeddings.clear();};
    for(;;){uint64_t parent=f.get64();if(parent==UINT64_MAX)break;
        uint64_t boundary=f.get64(),count=f.get64();
        if(!count||count>440||(parent>>60)||(boundary>>60)||(parent&boundary)||bits(boundary)>cap)
            throw std::runtime_error("invalid plan group");
        if(count>1&&(!(bits(boundary)&1)||!boundary))throw std::runtime_error("invalid multi-query boundary");
        unsigned sector=UINT32_MAX;
        for(unsigned i=0;i<count;++i){uint64_t id=f.get64(),removed=f.get64();
            if(id>=rows.size()||seen[id]++)throw std::runtime_error("duplicate/out-of-range plan ownership");
            const auto& r=rows[id];unsigned tag=excess(r.key)*16+defects(r.key);
            if(i&&tag!=sector)throw std::runtime_error("mixed sector group");sector=tag;
            if(count>1){
                if(bits(removed)!=3||(removed&~boundary)||(removed&parent)||bits(parent|removed)!=bits(r.key&full))
                    throw std::runtime_error("invalid residual embedding");
                unsigned c=60-bits(parent|boundary)+2*slack-excess(r.key);
                if(c&1||c>48)throw std::runtime_error("unsupported core parity/order");
                if(maps){embeddings.emplace_back(parent|removed,r.key&full);if(embeddings.size()==4096)flush();}
            }else if(parent||boundary||removed)throw std::runtime_error("invalid singleton encoding");
            ++queries;coefficient+=r.coefficient;
        }++groups;
    }
    if(f.get64()!=groups||queries!=rows.size())throw std::runtime_error("incomplete group coverage");
    if(maps&&!embeddings.empty())flush();
    auto plan_digest=f.finish_read();U128 expected=0;for(const auto& r:rows)expected+=r.coefficient;
    if(coefficient!=expected)throw std::runtime_error("coefficient coverage mismatch");
    std::printf("CORE_PLAN_VERIFY queries=%llu groups=%llu coefficient=%s maps=%s digest=%s exact_once=OK\n",
        (unsigned long long)queries,(unsigned long long)groups,decimal(coefficient).c_str(),maps?"all":"structural",plan_digest.c_str());
}

void build(const std::string& path,const std::vector<Entry>& rows,unsigned slack,const std::string& digest,
           unsigned cap,unsigned threads,uint64_t seed,const std::string& source,
           const std::string& costs,unsigned anchors) {
    auto started=Clock::now();Index index(rows);std::vector<uint8_t> owned(rows.size());
    std::unique_ptr<common_cost::Model> model;
    if(!source.empty())model=std::make_unique<common_cost::Model>(costs);
    six_by_twenty_nine::Geometry geometry;std::vector<uint64_t> triples;
    for(const auto& s:six_by_twenty_nine::weighted_supports(geometry))if(s.excess==1)triples.push_back(s.mask);
    std::sort(triples.begin(),triples.end());
    std::vector<uint32_t> parents;
    if(source.empty())for(size_t i=0;i<rows.size();++i){auto key=rows[i].key;unsigned e=excess(key),d=defects(key);
        if(e+1<=2*slack&&d+1<=2*slack){unsigned child_order=60+2*slack-2*(e+1)-2*(d+1);
            // Deliberately leave the larger, unbenchmarked orders as exact
            // independent fallbacks rather than inventing their GPU cost.
            if(child_order>=42&&child_order<=48)parents.push_back(uint32_t(i));}
    }
    std::sort(parents.begin(),parents.end(),[&](uint32_t a,uint32_t b){return std::make_pair(hash(rows[a].key^seed),rows[a].key)<std::make_pair(hash(rows[b].key^seed),rows[b].key);});
    common_catalog::File out(path,true);out.put("HCPLAN01",8);out.put64(slack);out.put64(cap);out.put64(seed);out.put64(rows.size());out.put(digest.data(),64);
    std::map<std::pair<unsigned,unsigned>,Totals> totals;
    // Histograms count actual field images, after dropping children that do
    // not need a later prime. They do not treat max-prime work as all children.
    using Bin=std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned>;
    std::map<Bin,uint64_t> histogram;
    std::map<Bin,std::vector<std::pair<uint64_t,Group>>> samples;
    uint64_t group_count=0,singletons=0;
    auto emit=[&](const Group& g,const std::vector<uint32_t>& ids){
        out.put64(g.parent);out.put64(g.boundary);out.put64(ids.size());++group_count;
        const auto& first=rows[ids[0]];unsigned e=excess(first.key),d=defects(first.key),n=order(first.key,slack);
        unsigned c=ids.size()>1?60-bits(g.parent|g.boundary)+2*slack-e:n;
        unsigned max_prime=0;
        auto& t=totals[{e,d}];++t.groups;
        for(size_t i=0;i<ids.size();++i){uint32_t id=ids[i];const auto& r=rows[id];
            if(owned[id]++)throw std::runtime_error("duplicate assignment");
            out.put64(id);out.put64(ids.size()>1?g.children[i].removed:0);
            ++t.queries;t.coefficient+=r.coefficient;t.original+=U128(r.primes)<<(n/2-1);
            max_prime=std::max(max_prime,unsigned(r.primes));}
        for(unsigned prime=0;prime<max_prime;++prime){unsigned active=0;for(auto id:ids)active+=rows[id].primes>prime;
            Bin bin{n,c,bits(g.boundary),active,prime};
            ++histogram[bin];t.signs+=U128(1)<<(c/2-1);
            if(ids.size()>1){uint64_t score=hash(g.parent^g.boundary^group_count^seed);
                auto& selected=samples[bin];
                if(selected.size()<3||score<selected.back().first){
                    Group sample;sample.parent=g.parent;sample.boundary=g.boundary;
                    for(size_t i=0;i<ids.size();++i)if(rows[ids[i]].primes>prime)sample.children.push_back(g.children[i]);
                    selected.emplace_back(score,std::move(sample));
                    std::sort(selected.begin(),selected.end(),[](const auto& a,const auto& b){return a.first<b.first;});
                    if(selected.size()>3)selected.pop_back();
                }
            }
        }
        if(ids.size()==1)++singletons;
    };
    if(!source.empty()){
        using Owned=common_cost::OwnedGroup;
        common_catalog::File input(source,false);char magic[8],identity[64];input.get(magic,8);
        if(std::memcmp(magic,"HCPLAN01",8)||input.get64()!=slack)throw std::runtime_error("source header mismatch");
        if(input.get64()>cap)throw std::runtime_error("cannot lower the source plan boundary cap");
        input.get64(); // source seed, independently verified by main
        if(input.get64()!=rows.size())throw std::runtime_error("source count mismatch");
        input.get(identity,64);if(std::string(identity,64)!=digest)throw std::runtime_error("source catalog mismatch");
        std::vector<std::vector<Owned>> batch;std::vector<Owned> family;
        uint64_t families=0,improved=0,read_groups=0,estimated_groups=0;
        U128 before=0,after=0;
        auto flush=[&]{if(batch.empty())return;
            std::vector<std::vector<Owned>> repaired(batch.size());std::vector<std::string> errors(batch.size());
            #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
            for(size_t i=0;i<batch.size();++i)try{
                repaired[i]=common_cost::repair(batch[i],*model,rows,slack,cap,anchors);
            }catch(const std::exception& e){errors[i]=e.what();}
            for(size_t i=0;i<batch.size();++i){if(!errors[i].empty())throw std::runtime_error(errors[i]);
                auto old=common_cost::total_cost(batch[i],*model,rows,slack),now=common_cost::total_cost(repaired[i],*model,rows,slack);
                before+=old;after+=now;improved+=now<old;++families;
                std::vector<uint32_t> want,got;
                for(const auto& g:batch[i])want.insert(want.end(),g.ids.begin(),g.ids.end());
                for(const auto& g:repaired[i])got.insert(got.end(),g.ids.begin(),g.ids.end());
                std::sort(want.begin(),want.end());std::sort(got.begin(),got.end());
                if(want!=got)throw std::runtime_error("repair changed owned query set");
                for(const auto& g:repaired[i]){bool estimated=false;model->cost(g,rows,slack,&estimated);
                    estimated_groups+=estimated;emit(g.group,g.ids);}
            }
            batch.clear();
            if(families%8192==0){std::printf("CORE_REPLAN_PROGRESS families=%llu improved=%llu model_before_h=%.6f model_after_h=%.6f seconds=%.3f\n",
                (unsigned long long)families,(unsigned long long)improved,double(before)/3.6e15,double(after)/3.6e15,
                std::chrono::duration<double>(Clock::now()-started).count());std::fflush(stdout);}
        };
        auto finish_family=[&]{if(!family.empty()){batch.push_back(std::move(family));family.clear();if(batch.size()==128)flush();}};
        std::printf("CORE_REPLAN_START model_sha256=%s anchors=%u cap=%u scope=model_not_measured\n",model->digest.c_str(),anchors,cap);std::fflush(stdout);
        for(;;){uint64_t parent=input.get64();if(parent==UINT64_MAX)break;
            uint64_t boundary=input.get64(),count=input.get64();
            if(!count||count>440)throw std::runtime_error("invalid source group count");
            Owned x;x.group.parent=parent;x.group.boundary=boundary;
            for(unsigned i=0;i<count;++i){uint64_t id=input.get64(),removed=input.get64();
                if(id>=rows.size())throw std::runtime_error("source query out of range");
                x.ids.push_back(uint32_t(id));x.group.children.push_back({removed,rows[id].key&full,uint32_t(id)});}
            ++read_groups;
            if(count==1){finish_family();flush();emit(x.group,x.ids);continue;}
            if(!family.empty()&&(family.front().group.parent!=parent||
                defects(rows[family.front().ids.front()].key)!=defects(rows[x.ids.front()].key)))finish_family();
            family.push_back(std::move(x));
        }
        finish_family();flush();
        if(input.get64()!=read_groups)throw std::runtime_error("source group count mismatch");
        auto source_digest=input.finish_read();
        std::printf("CORE_REPLAN_DONE families=%llu improved=%llu source_groups=%llu output_groups=%llu estimated_groups=%llu model_before_h=%.9f model_after_h=%.9f source_sha256=%s model_sha256=%s scope=model_not_measured\n",
            (unsigned long long)families,(unsigned long long)improved,(unsigned long long)read_groups,
            (unsigned long long)group_count,(unsigned long long)estimated_groups,double(before)/3.6e15,double(after)/3.6e15,
            source_digest.c_str(),model->digest.c_str());
    }
    constexpr size_t CHUNK=1024;
    std::printf("CORE_PLAN_START queries=%zu parents=%zu cap=%u threads=%u\n",rows.size(),parents.size(),cap,threads);std::fflush(stdout);
    for(size_t begin=0;begin<parents.size();begin+=CHUNK){size_t count=std::min(CHUNK,parents.size()-begin);
        std::vector<Family> families(count);std::vector<std::string> errors(count);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,8)
        for(size_t i=0;i<count;++i)try{families[i]=build_family(geometry,rows[parents[begin+i]].key&full,triples);}catch(const std::exception& e){errors[i]=e.what();}
        for(size_t i=0;i<count;++i){if(!errors[i].empty())throw std::runtime_error(errors[i]);
            auto& f=families[i];unsigned child_defects=defects(rows[parents[begin+i]].key)+1;
            std::vector<uint32_t> child_ids(f.distinct,UINT32_MAX);
            for(const auto& child:f.children)child_ids[child.id]=index.at(child.canonical|(uint64_t(child_defects)<<60));
            std::set<uint64_t> tried;
            for(;;){f.children.erase(std::remove_if(f.children.begin(),f.children.end(),[&](const Child& c){return owned[child_ids[c.id]];}),f.children.end());
                auto anchor=std::find_if(f.children.begin(),f.children.end(),[&](const Child& c){return !tried.count(c.removed);});
                if(anchor==f.children.end())break;tried.insert(anchor->removed);
                Group g=grow(f,anchor->removed,cap);if(g.size()<2)continue;
                unsigned e=excess(rows[child_ids[g.children[0].id]].key),n=order(rows[child_ids[g.children[0].id]].key,slack);
                validate(f,g,g.children[0].canonical,n,2*slack-e);
                std::vector<uint32_t> ids;for(const auto& c:g.children)ids.push_back(child_ids[c.id]);emit(g,ids);
            }
        }
        if(begin%(CHUNK*32)==0||begin+count==parents.size()){
            std::printf("CORE_PLAN_PROGRESS parents=%zu/%zu groups=%llu seconds=%.3f\n",begin+count,parents.size(),(unsigned long long)group_count,std::chrono::duration<double>(Clock::now()-started).count());std::fflush(stdout);}
    }
    for(size_t i=0;i<rows.size();++i)if(!owned[i])emit(Group{},std::vector<uint32_t>{uint32_t(i)});
    out.put64(UINT64_MAX);out.put64(group_count);auto plan_digest=out.finish_write();
    U128 original=0,planned=0;uint64_t covered=0;
    for(const auto& [key,t]:totals){auto [e,d]=key;original+=t.original;planned+=t.signs;covered+=t.queries;
        std::printf("CORE_PLAN_SECTOR e=%u d=%u order=%u queries=%llu groups=%llu coefficient=%s original_signs=%s assigned_signs=%s\n",
            e,d,60+2*slack-2*e-2*d,(unsigned long long)t.queries,(unsigned long long)t.groups,decimal(t.coefficient).c_str(),decimal(t.original).c_str(),decimal(t.signs).c_str());}
    for(auto [key,count]:histogram){auto [n,c,pool,g,prime]=key;
        std::printf("CORE_PLAN_BIN order=%u core=%u pool=%u active_queries=%u prime_index=%u groups=%llu\n",n,c,pool,g,prime,(unsigned long long)count);}
    for(const auto& [key,selected]:samples)for(const auto& sample:selected){const auto& g=sample.second;auto [n,core,pool,active,prime]=key;
        unsigned occupied=bits(g.parent)+3;unsigned d=0;
        uint32_t id=0;bool found=false;
        for(unsigned candidate=1;candidate<=6;++candidate){unsigned e=occupied-2*candidate;
            if(e<=2*slack&&60+2*slack-2*candidate-2*e==n){id=index.at(g.children[0].canonical|(uint64_t(candidate)<<60));found=true;break;}}
        if(!found)throw std::runtime_error("sample sector lost");d=defects(rows[id].key);unsigned e=excess(rows[id].key);
        std::printf("CORE27_GROUP e=%u d=%u root=%llu cap=%u parent=%llu boundary=%llu core=%u tail=%u queries=%u prime_index=%u members=",e,d,
            (unsigned long long)g.children[0].canonical,pool,(unsigned long long)g.parent,(unsigned long long)g.boundary,
            core_order(g,2*slack-e),pool-3,g.size(),prime);
        for(const auto& c:g.children)std::printf("%llu:%llu,",(unsigned long long)c.canonical,(unsigned long long)c.removed);std::printf("\n");}
    if(covered!=rows.size())throw std::runtime_error("coverage lost");
    std::printf("CORE_PLAN_DONE queries=%llu groups=%llu singletons=%llu original_signs=%s assigned_signs=%s digest=%s seconds=%.3f exact_once=OK\n",
        (unsigned long long)covered,(unsigned long long)group_count,(unsigned long long)singletons,decimal(original).c_str(),decimal(planned).c_str(),plan_digest.c_str(),std::chrono::duration<double>(Clock::now()-started).count());
}
}
int main(int argc,char** argv)try {
    std::string catalog,path,source,costs;bool audit=false,maps=false;unsigned cap=11,threads=8,anchors=0;uint64_t seed=478;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--catalog"&&i+1<argc)catalog=argv[++i];
        else if(a=="--output"&&i+1<argc)path=argv[++i];
        else if(a=="--verify"&&i+1<argc){path=argv[++i];audit=true;}
        else if(a=="--all-maps")maps=true;
        else if(a=="--cap"&&i+1<argc)cap=std::stoul(argv[++i]);
        else if(a=="--threads"&&i+1<argc)threads=std::stoul(argv[++i]);
        else if(a=="--seed"&&i+1<argc)seed=std::stoull(argv[++i]);
        else if(a=="--replan"&&i+1<argc)source=argv[++i];
        else if(a=="--costs"&&i+1<argc)costs=argv[++i];
        else if(a=="--repair-anchors"&&i+1<argc)anchors=std::stoul(argv[++i]);
        else throw std::runtime_error("usage: --catalog FILE --output FILE|--verify FILE [--all-maps --cap 7|9|11 --threads N --seed N] [--replan PLAN --costs TABLE --repair-anchors 0..8]");}
    if(path.empty()||catalog.empty()||!threads||threads>16||(cap!=7&&cap!=9&&cap!=11))throw std::runtime_error("invalid plan options");
    if(source.empty()!=costs.empty()||anchors>8||(audit&&!source.empty()))throw std::runtime_error("invalid replan options");
    unsigned slack;std::string digest;auto rows=common_catalog::read(catalog,slack,digest);
    omp_set_dynamic(0);omp_set_num_threads(int(threads));
    if(audit)verify(path,rows,slack,digest,maps);else {
        if(!source.empty())verify(source,rows,slack,digest,false);
        build(path,rows,slack,digest,cap,threads,seed,source,costs,anchors);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"error: %s\n",e.what());return 1;}
