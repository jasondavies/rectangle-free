#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace column_flip {
struct Parity {
    std::array<uint8_t,8> parent{},offset{};
    unsigned components;
    explicit Parity(unsigned n):components(n) {
        for(unsigned i=0;i<8;++i)parent[i]=uint8_t(i);
    }
    std::pair<unsigned,unsigned> find(unsigned i) const {
        unsigned sign=0;
        while(parent[i]!=i){sign^=offset[i];i=parent[i];}
        return {i,sign};
    }
    bool add(unsigned a,unsigned b,unsigned parity) {
        auto [ra,pa]=find(a); auto [rb,pb]=find(b);
        if(ra==rb)return (pa^pb)==parity;
        parent[ra]=uint8_t(rb);offset[ra]=uint8_t(pa^pb^parity);
        --components;return true;
    }
};

struct Split {uint32_t zero=0,one=0;};
struct Key {
    std::array<uint32_t,8> planes{};
    uint32_t fixed=0,count=0;
    bool operator==(const Key& b)const {
        return fixed==b.fixed&&count==b.count&&planes==b.planes;
    }
};
struct Hash {
    size_t operator()(const Key& k)const {
        uint64_t h=k.fixed^(uint64_t(k.count)<<32);
        for(uint32_t x:k.planes) {
            h^=x+UINT64_C(0x9e3779b97f4a7c15)+(h<<6)+(h>>2);
        }
        return size_t(h);
    }
};
struct Template {
    Key key;
    uint64_t weight=0;
    uint32_t occupied=0;
};
struct Family {
    std::vector<Template> entries;
    uint64_t raw_templates=1,valid_templates=0;
    bool capped=false;
};

inline std::vector<Split> choices(uint8_t active) {
    std::vector<Split> out;
    unsigned assignment=active;
    unsigned anchor=active&(~unsigned(active)+1);
    for(;;) {
        if(!(assignment&anchor)) {
            Split split;
            unsigned p=0;
            for(unsigned i=0;i<8;++i)for(unsigned j=i+1;j<8;++j,++p) {
                if(!(active>>i&1)||!(active>>j&1))continue;
                if((assignment>>i&1)!=(assignment>>j&1))continue;
                ((assignment>>i&1)?split.one:split.zero)|=1U<<p;
            }
            out.push_back(split);
        }
        if(!assignment)break;
        assignment=(assignment-1)&active;
    }
    return out;
}

inline Family build(uint32_t prefix,uint64_t cap) {
    std::vector<std::vector<Split>> options;
    Family out;
    for(unsigned c=0;c<4;++c) {
        uint8_t active=0;
        for(unsigned row=0;row<8;++row)
            if(prefix>>(4*(7-row)+c)&1)active|=1U<<row;
        if(!active)continue; // Empty columns have no orientation variable.
        options.push_back(choices(active));
        out.raw_templates*=options.back().size();
    }
    if(out.raw_templates>cap){out.capped=true;return out;}
    std::unordered_map<Key,uint64_t,Hash> merged;
    std::array<Split,4> selected{};
    auto visit=[&](auto&& self,unsigned column,Parity parity)->void {
        if(column==options.size()) {
            ++out.valid_templates;
            std::array<Split,4> components{};
            for(unsigned i=0;i<column;++i) {
                auto [root,sign]=parity.find(i);
                components[root].zero|=sign?selected[i].one:selected[i].zero;
                components[root].one|=sign?selected[i].zero:selected[i].one;
            }
            Key key;
            uint64_t weight=1;
            std::vector<std::pair<uint32_t,uint32_t>> free;
            for(unsigned i=0;i<column;++i)if(parity.parent[i]==i) {
                auto [a,b]=components[i];
                uint32_t fixed=a&b;key.fixed|=fixed;
                a^=fixed;b^=fixed;
                if(!(a|b))weight*=2; // Invisible/fixed orientation multiplicity.
                else free.emplace_back(std::min(a,b),std::max(a,b));
            }
            std::sort(free.begin(),free.end());
            key.count=unsigned(free.size());
            for(unsigned i=0;i<free.size();++i) {
                key.planes[2*i]=free[i].first;key.planes[2*i+1]=free[i].second;
            }
            merged[key]+=weight;
            return;
        }
        for(Split current:options[column]) {
            Parity next=parity;
            bool valid=true;
            for(unsigned j=0;j<column&&valid;++j) {
                const auto previous=selected[j];
                uint32_t same=(current.zero&previous.zero)|(current.one&previous.one);
                uint32_t opposite=(current.zero&previous.one)|(current.one&previous.zero);
                if(same&&opposite)valid=false;
                else if(same||opposite)valid=next.add(column,j,bool(same));
            }
            if(valid){selected[column]=current;self(self,column+1,next);}
        }
    };
    visit(visit,0,Parity(unsigned(options.size())));
    for(const auto& [key,weight]:merged) {
        uint32_t occupied=key.fixed;
        for(uint32_t p:key.planes)occupied|=p;
        out.entries.push_back(Template{key,weight,occupied});
    }
    return out;
}

inline std::vector<uint64_t> expand(const Template& t) {
    std::vector<uint64_t> masks;
    for(unsigned signs=0;signs<(1U<<t.key.count);++signs) {
        uint32_t a=t.key.fixed,b=t.key.fixed;
        for(unsigned i=0;i<t.key.count;++i) {
            unsigned flip=signs>>i&1;
            a|=t.key.planes[2*i+flip];b|=t.key.planes[2*i+(flip^1)];
        }
        masks.push_back(uint64_t(a)|(uint64_t(b)<<28));
    }
    return masks;
}

struct Work {
    uint64_t pairs=0,fixed_reject=0,conflict_reject=0,cycle_reject=0;
    uint64_t component_tests=0,equations=0,accepted=0;
};

inline unsigned orientations(const Template& a,const Template& b,Work& work) {
    ++work.pairs;
    if((a.key.fixed&b.occupied)||(b.key.fixed&a.occupied)) {
        ++work.fixed_reject;return 0;
    }
    Parity parity(a.key.count+b.key.count);
    // A bipartite graph with at most one vertex on either side is a forest:
    // every noncontradictory edge removes exactly one free sign. Avoid DSU
    // searches for this common case without changing the general cycle test.
    const bool star=a.key.count<=1||b.key.count<=1;
    for(unsigned i=0;i<a.key.count;++i)for(unsigned j=0;j<b.key.count;++j) {
        ++work.component_tests;
        uint32_t a0=a.key.planes[2*i],a1=a.key.planes[2*i+1];
        uint32_t b0=b.key.planes[2*j],b1=b.key.planes[2*j+1];
        uint32_t same=(a0&b0)|(a1&b1),opposite=(a0&b1)|(a1&b0);
        if(same&&opposite){++work.conflict_reject;return 0;}
        if(same||opposite) {
            ++work.equations;
            if(star)--parity.components;
            else if(!parity.add(i,a.key.count+j,bool(same))) {
                ++work.cycle_reject;return 0;
            }
        }
    }
    ++work.accepted;
    return 1U<<parity.components;
}
} // namespace column_flip
