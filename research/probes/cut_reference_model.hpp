#pragma once
// Independent reference scorer for exact regression tests.
#include "cut_geometry.hpp"
#include "response_model.hpp"
struct CostModel {
    response::Engine engine{8};
    std::unordered_map<uint64_t,std::vector<Entry>> cache;
    uint64_t cached_entries=0,builds=0,hits=0;
    response::Dist layout(uint64_t raw) {
        const CanonicalForm form=canonical_prefix(raw,4);
        auto it=cache.find(form.key);
        if(it==cache.end()) {
            auto d=quotient_token_planes(build_distribution(form.key,4,false));
            if(cached_entries+d.entries.size()>2000000){cache.clear();cached_entries=0;}
            cached_entries+=d.entries.size();++builds;
            it=cache.emplace(form.key,std::move(d.entries)).first;
        } else ++hits;
        std::map<unsigned,std::map<std::pair<uint64_t,unsigned>,uint64_t>> sizes;
        for(auto entry:it->second) {
            uint64_t mask=transform_pair_mask(entry.mask,form.row_map);
            // Preserve the representative chosen in canonical coordinates.
            ++sizes[engine.prefix(mask)][{entry.weight,engine.orbit(mask)}];
        }
        response::Dist out;
        for(const auto& [p,classes]:sizes) {
            response::Bucket b{p,{},{}};
            for(auto [key,n]:classes)b.classes.push_back({n,key.second});
            out.push_back(std::move(b));
        }
        return out;
    }
    uint64_t cost(uint64_t key) {
        uint64_t total=0;
        for(unsigned side=0;side<2;++side) {
            uint64_t k=side?~key:key,left=0,right=0;
            for(unsigned r=0;r<8;++r){unsigned row=unsigned(k>>((7-r)*8))&255;
                left=(left<<4)|(row&15);right=(right<<4)|(row>>4);}
            auto a=layout(left),b=layout(right);
            response::Budget budget(100000000000ull,300);
            engine.tile_model(a,b,budget);total+=budget.modeled_tiles;
        }
        return total;
    }
};
