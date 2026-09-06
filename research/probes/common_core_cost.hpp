#pragma once
// Local, exact-ownership plan repair. Timings guide selection; they do not
// prove speedups. Missing group-size samples are interpolated/clamped and
// every new plan needs its own GPU sweep before a runtime claim.
#include "common_core_catalog_io.hpp"
#include "six_by_twenty_seven_common_core.hpp"
#include <fstream>
#include <numeric>
#include <sstream>
#include <tuple>

namespace common_cost {
using namespace six_by_common_core;
using U128=unsigned __int128;
constexpr uint64_t unknown=UINT64_MAX;
struct OwnedGroup {Group group;std::vector<uint32_t> ids;};

struct Model {
    using Shape=std::tuple<unsigned,unsigned,unsigned,unsigned>;
    std::map<Shape,std::map<unsigned,uint64_t>> table;
    std::string digest;
    explicit Model(const std::string& path) {
        std::ifstream in(path);if(!in)throw std::runtime_error("cannot read cost model");
        std::string line;Sha256 sha;
        if(!std::getline(in,line)||line!="HCCOST01 hess=1 boundary=1 scratch=1")
            throw std::runtime_error("unsupported cost model/kernel");
        sha.update(line+"\n");
        while(std::getline(in,line)){sha.update(line+"\n");if(line.empty()||line[0]=='#')continue;
            std::istringstream fields(line);unsigned n,c,q,g,p,count;uint64_t ps;std::string extra;
            if(!(fields>>n>>c>>q>>g>>p>>ps>>count)||(fields>>extra)||
               n<42||n>48||(n&1)||q<5||q>11||!(q&1)||n!=c+q-3||
               !g||g>256||p>3||!ps||ps>1000000000||!count||
               !table[{n,c,q,p}].emplace(g,ps).second)
                throw std::runtime_error("invalid/duplicate cost model row");
        }
        if(table.empty())throw std::runtime_error("empty cost model");digest=sha.finish_hex();
    }
    uint64_t rate(unsigned n,unsigned q,unsigned g,unsigned prime,bool* estimated=nullptr)const {
        unsigned c=n-q+3;auto it=table.find({n,c,q,prime});
        if(it==table.end()||!g)return unknown;
        const auto& curve=it->second;auto hi=curve.lower_bound(g);
        if(hi!=curve.end()&&hi->first==g)return hi->second;
        if(estimated)*estimated=true;
        // Clamp below the measured range to its smallest group. This is a
        // heuristic, not a certified timing bound. Do not extrapolate larger
        // groups beyond the measured gate.
        if(hi==curve.begin())return hi->second;
        if(hi==curve.end())return unknown;
        auto lo=std::prev(hi);unsigned width=hi->first-lo->first;
        return uint64_t((U128(lo->second)*(hi->first-g)+U128(hi->second)*(g-lo->first)+width/2)/width);
    }
    uint64_t cost(const OwnedGroup& x,const std::vector<common_catalog::Entry>& rows,
                  unsigned slack,bool* estimated=nullptr)const {
        if(x.ids.size()<2||x.ids.size()>256)return unknown;
        auto key=rows[x.ids.front()].key;
        unsigned d=unsigned(key>>60),e=bits(key&full)-2*d;
        unsigned n=60+2*slack-2*e-2*d,q=bits(x.group.boundary),c=n-q+3;
        if(q<5||q>11||!(q&1)||c>48||c<2||(c&1))return unknown;
        uint64_t total=0;
        for(unsigned p=0;p<4;++p){unsigned active=0;for(auto id:x.ids)active+=rows[id].primes>p;
            if(!active)continue;uint64_t value=rate(n,q,active,p,estimated);if(value==unknown)return unknown;
            total+=value*(UINT64_C(1)<<(c/2-1));}
        return total;
    }
};

inline OwnedGroup cheapest(OwnedGroup x,const Model& model,
        const std::vector<common_catalog::Entry>& rows,unsigned slack,unsigned cap) {
    uint64_t used=0;for(const auto& c:x.group.children)used|=c.removed;
    uint64_t best=model.cost(x,rows,slack),boundary=used;
    for(unsigned q=bits(used);q<=cap;++q){
        if(q>=5&&(q&1)){
            auto trial=x;trial.group.boundary=boundary;
            uint64_t cost=model.cost(trial,rows,slack);
            if(cost<best){x.group.boundary=boundary;best=cost;}
        }
        uint64_t spare=full&~(x.group.parent|boundary);if(!spare)break;
        boundary|=spare&-spare;
    }
    return x;
}

inline uint64_t total_cost(const std::vector<OwnedGroup>& groups,const Model& model,
        const std::vector<common_catalog::Entry>& rows,unsigned slack) {
    uint64_t total=0;for(const auto& g:groups){auto cost=model.cost(g,rows,slack);
        if(cost==unknown)return unknown;total+=cost;}return total;
}

inline std::vector<OwnedGroup> repair(std::vector<OwnedGroup> groups,const Model& model,
        const std::vector<common_catalog::Entry>& rows,unsigned slack,unsigned cap,unsigned anchors) {
    uint64_t original=total_cost(groups,model,rows,slack);
    if(original==unknown)throw std::runtime_error("incumbent shape absent from measured model");
    for(auto& g:groups)g=cheapest(std::move(g),model,rows,slack,cap);
    // Bounded pair exchanges retain the original parent's owned query set.
    // Each accepted step strictly reduces the integer model objective.
    for(unsigned pass=0;pass<2;++pass){bool changed=false;
        for(size_t a=0;a<groups.size();++a)for(size_t b=a+1;b<groups.size();++b){
            OwnedGroup joined=groups[a];joined.group.boundary|=groups[b].group.boundary;
            joined.ids.insert(joined.ids.end(),groups[b].ids.begin(),groups[b].ids.end());
            joined.group.children.insert(joined.group.children.end(),groups[b].group.children.begin(),groups[b].group.children.end());
            uint64_t best=model.cost(groups[a],rows,slack)+model.cost(groups[b],rows,slack);
            std::vector<OwnedGroup> replacement;
            auto merge=cheapest(joined,model,rows,slack,cap);
            uint64_t merge_cost=bits(merge.group.boundary)<=cap?model.cost(merge,rows,slack):unknown;
            if(merge_cost<best){best=merge_cost;replacement={std::move(merge)};}
            if(anchors&&joined.ids.size()<=440){
                Family f;f.parent=joined.group.parent;f.distinct=unsigned(joined.ids.size());f.children=joined.group.children;
                for(unsigned i=0;i<f.children.size();++i)f.children[i].id=i;
                std::vector<unsigned> order(f.children.size());std::iota(order.begin(),order.end(),0);
                std::sort(order.begin(),order.end(),[&](unsigned i,unsigned j){return std::make_pair(hash(f.children[i].canonical),i)<std::make_pair(hash(f.children[j].canonical),j);});
                if(order.size()>anchors)order.resize(anchors);
                for(unsigned anchor:order)for(unsigned q:{7u,9u,11u})if(q<=cap){
                    Group grown=grow(f,f.children[anchor].removed,q);
                    if(grown.size()<2||grown.size()+2>joined.ids.size())continue;
                    bool selected[440]{};OwnedGroup left{grown,{}};
                    for(const auto& c:grown.children){selected[c.id]=true;left.ids.push_back(joined.ids[c.id]);}
                    OwnedGroup right;right.group.parent=f.parent;
                    for(unsigned i=0;i<f.children.size();++i)if(!selected[i]){
                        right.ids.push_back(joined.ids[i]);right.group.children.push_back(f.children[i]);right.group.boundary|=f.children[i].removed;}
                    if(bits(right.group.boundary)>cap)continue;
                    left=cheapest(std::move(left),model,rows,slack,cap);
                    right=cheapest(std::move(right),model,rows,slack,cap);
                    auto lc=model.cost(left,rows,slack),rc=model.cost(right,rows,slack);
                    if(lc!=unknown&&rc!=unknown&&lc+rc<best){best=lc+rc;replacement={std::move(left),std::move(right)};}
                }
            }
            if(!replacement.empty()){
                groups[a]=std::move(replacement[0]);
                if(replacement.size()==2)groups[b]=std::move(replacement[1]);
                else {groups.erase(groups.begin()+b);--b;}
                changed=true;
            }
        }
        if(!changed)break;
    }
    if(total_cost(groups,model,rows,slack)>original)throw std::runtime_error("cost repair worsened its own objective");
    return groups;
}
} // namespace common_cost
