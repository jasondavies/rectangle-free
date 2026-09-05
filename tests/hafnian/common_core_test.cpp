#include "../../research/probes/six_by_twenty_seven_common_core.hpp"
#include <cassert>
#include <iostream>
#include <numeric>

using namespace six_by_common_core;

static bool adjacent(const six_by_twenty_nine::Geometry& geometry,
                     unsigned a, unsigned b) {
    if (a == b) return false;
    if (a >= 60 || b >= 60) return (a < 60) != (b < 60);
    if (a/15 == b/15) return false;
    auto [u,v] = geometry.pairs[a%15];
    auto [x,y] = geometry.pairs[b%15];
    return u != x && u != y && v != x && v != y;
}

int main() {
    Sample a{31,475}, b{31,475};
    std::vector<std::pair<uint64_t,uint64_t>> all;
    for (uint64_t i=0;i<10000;++i) {
        a.add(i); b.add(9999-i); all.emplace_back(hash(i ^ 475),i);
    }
    std::sort(all.begin(),all.end()); all.resize(31);
    std::vector<uint64_t> expected;
    for(auto x:all)expected.push_back(x.second);
    std::sort(expected.begin(),expected.end());
    assert(a.population==10000 && a.values()==b.values() && a.values()==expected);

    six_by_twenty_nine::Geometry geometry;
    std::vector<uint64_t> triples;
    for (const auto& s:six_by_twenty_nine::weighted_supports(geometry))
        if(s.excess==1)triples.push_back(s.mask);
    std::sort(triples.begin(),triples.end());
    uint64_t parent=0;
    for(uint64_t s:triples)if(!(s&parent)) {
        parent|=s; if(bits(parent)==9)break;
    }
    assert(bits(parent)==9);
    parent=six_by_twenty_nine::canonicalize(geometry,parent);
    Family f=build_family(geometry,parent,triples);
    assert(f.distinct>1 && f.children.size()<=440);
    unsigned checked=0;
    for(unsigned anchor=0;anchor<std::min<size_t>(32,f.children.size());++anchor)
        for(unsigned cap:{5u,7u,9u,11u}) for(unsigned dummy:{0u,2u}) {
            Group g=grow(f,f.children[anchor].removed,cap);
            unsigned order=48+dummy;
            validate(f,g,f.children[anchor].canonical,order,dummy);
            assert(bits(g.boundary)<=cap && g.size()==contained(f,g.boundary));
            std::vector<unsigned> common;
            uint64_t mask=full&~(g.parent|g.boundary);
            for(unsigned v=0;v<60;++v)if(mask&(UINT64_C(1)<<v))common.push_back(v);
            for(unsigned i=0;i<dummy;++i)common.push_back(60+i);
            assert(common.size()==core_order(g,dummy) && !(common.size()&1));
            for(const Child& child:g.children) {
                assert(child.canonical==six_by_twenty_nine::canonicalize(
                    geometry,g.parent|child.removed));
                auto vertices=common;
                uint64_t tail=g.boundary&~child.removed;
                for(unsigned v=0;v<60;++v)if(tail&(UINT64_C(1)<<v))vertices.push_back(v);
                assert(vertices.size()==order);
                assert(std::set<unsigned>(vertices.begin(),vertices.end()).size()==order);
                // Matrix M=A*X*diag(signs): compare the common block under
                // several sign schedules. Its paired columns never enter tail.
                for(unsigned sign=0;sign<4;++sign)
                    for(unsigned i=0;i<common.size();++i)
                        for(unsigned j=0;j<common.size();++j) {
                            int factor=((hash(sign+j/2)&1)?-1:1);
                            int actual=factor*int(adjacent(geometry,vertices[i],vertices[j^1]));
                            int want=factor*int(adjacent(geometry,common[i],common[j^1]));
                            assert(actual==want);
                        }
                ++checked;
            }
        }
    bool rejected=false;
    try{grow(f,f.children.front().removed,6);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
    std::cout<<"COMMON_CORE_TEST members="<<checked
             <<" sampling=OK dedup=OK pairing=OK signed_core=OK exact=OK\n";
}
