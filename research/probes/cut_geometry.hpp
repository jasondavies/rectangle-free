#pragma once
// Exact 8x8 cut mappings shared by research drivers.
#include <map>
#include <set>
#include <sstream>
#define GRID_ROWS 8
#define GRID_COLUMNS 8
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 4
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "R8ORB01"
#include "../../src/gpu/twocolour_gpu_common.cuh"

static uint64_t transpose(uint64_t key) {
    uint64_t result=0;
    for(unsigned r=0;r<8;++r)for(unsigned c=0;c<8;++c)
        result|=((key>>((7-r)*8+c))&1)<<((7-c)*8+r);
    return result;
}

namespace reuse_cut {
static uint32_t half(uint64_t key,unsigned columns) {
    if(__builtin_popcount(columns)!=4||columns>255)throw std::runtime_error("invalid cut");
    uint32_t out=0;
    for(unsigned r=0;r<8;++r) {
        unsigned pattern=0,pos=0;
        for(unsigned c=0;c<8;++c)if(columns>>c&1)
            pattern|=unsigned(key>>((7-r)*8+c)&1)<<pos++;
        out=(out<<4)|pattern;
    }
    return out;
}
static uint64_t assemble(uint32_t left,uint32_t right) {
    uint64_t out=0;
    for(unsigned r=0;r<8;++r)out=(out<<8)|((left>>((7-r)*4))&15)|
        (uint64_t((right>>((7-r)*4))&15)<<4);
    return out;
}
static void self_test() {
    for(unsigned t=0;t<64;++t)for(unsigned cut=0;cut<256;++cut) {
        if(__builtin_popcount(cut)!=4)continue;
        uint64_t key=mix64(t),out=assemble(half(key,cut),half(key,255^cut)),restored=0;
        for(unsigned r=0;r<8;++r) {
            unsigned low=0,high=4,row=0;
            for(unsigned c=0;c<8;++c)row|=unsigned(out>>((7-r)*8+((cut>>c&1)?low++:high++))&1)<<c;
            restored=(restored<<8)|row;
        }
        if(restored!=key||half(~key,cut)!=~half(key,cut)||
           transpose(transpose(out))!=out)throw std::runtime_error("cut inverse/complement mismatch");
    }
    std::cout<<"REUSE_BUDGET_CUT_TEST exact=OK\n";
}
static std::vector<std::pair<bool,unsigned>> enumerate(const std::string& menu) {
    std::vector<std::pair<bool,unsigned>> cuts;
    if(menu=="four")cuts={{false,15},{true,15},{true,23},{false,23}};
    else if(menu=="all") {
        cuts.emplace_back(false,15);
        for(unsigned t=0;t<2;++t)for(unsigned c=0;c<256;++c)
            if((c&1)&&__builtin_popcount(c)==4&&(t||c!=15))cuts.emplace_back(bool(t),c);
    } else throw std::runtime_error("menu must be four or all");
    return cuts;
}
}
