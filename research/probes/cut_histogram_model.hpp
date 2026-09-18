// Research-only exact prefix/class summaries. Included after the common model.
// Never re-choose a token-plane representative after applying the row map.
#include "cut_tile_index.hpp"
struct CutHistogramModel {
    response::Engine engine{8};
    struct Source {
        std::vector<uint64_t> entries; // 56-bit canonical mask + class ordinal
        std::vector<std::pair<uint64_t,unsigned>> classes;
    };
    std::unordered_map<uint64_t,Source> cache;
    std::vector<uint32_t> counts = std::vector<uint32_t>(16384*32);
    std::vector<uint32_t> touched;
    bool projected=true;
    bool indexed=false;
    bool grouped=false;
    bool planned=false;
    CutTileScorer scorer;
    uint64_t builds=0,hits=0,entries=0;
    double build_seconds=0,histogram_seconds=0,canonical_seconds=0;
    static constexpr uint64_t mask_bits=(uint64_t(1)<<56)-1;

    response::Dist layout(uint64_t raw) {
        double start=response::now();
        auto form=canonical_prefix(raw,4);
        canonical_seconds+=response::now()-start;
        auto it=cache.find(form.key);
        if(it==cache.end()) {
            start=response::now();
            auto d=quotient_token_planes(build_distribution(form.key,4,false));
            Source source;
            std::set<std::pair<uint64_t,unsigned>> alphabet;
            for(auto e:d.entries)alphabet.emplace(e.weight,engine.orbit(e.mask));
            source.classes.assign(alphabet.begin(),alphabet.end());
            if(source.classes.size()>32)throw std::runtime_error("weight alphabet too large");
            source.entries.reserve(d.entries.size());
            for(auto e:d.entries) {
                auto klass=std::make_pair(e.weight,engine.orbit(e.mask));
                unsigned ordinal=std::lower_bound(source.classes.begin(),source.classes.end(),klass)-source.classes.begin();
                if(e.mask&~mask_bits)throw std::runtime_error("non-8-row support");
                source.entries.push_back(e.mask|(uint64_t(ordinal)<<56));
            }
            // Bounded research cache: never silently evict/rebuild sources.
            if(entries+source.entries.size()>300000000)
                throw std::runtime_error("canonical cache exceeds 300M-entry research cap");
            entries+=source.entries.size();++builds;
            it=cache.emplace(form.key,std::move(source)).first;
            build_seconds+=response::now()-start;
        } else ++hits;
        start=response::now();
        const auto& source=it->second;
        response::Dist out;
        if(!projected) {
            std::map<unsigned,std::map<std::pair<uint64_t,unsigned>,uint64_t>> sizes;
            for(uint64_t e:source.entries) {
                uint64_t mask=transform_pair_mask(e&mask_bits,form.row_map);
                ++sizes[engine.prefix(mask)][source.classes[e>>56]];
            }
            for(const auto& [p,classes]:sizes) {
                response::Bucket b{p,{},{}};
                for(auto [key,n]:classes)b.classes.push_back({n,key.second});
                out.push_back(std::move(b));
            }
        } else if(!source.entries.empty()) {
            // The final histogram only needs the 14 prefix bits. Compose the
            // row permutation with projection once, then use seven byte LUTs.
            uint16_t table[7][256]{};
            for(unsigned byte=0;byte<7;++byte) {
                uint16_t image[8];
                for(unsigned bit=0;bit<8;++bit)
                    image[bit]=engine.prefix(transform_pair_mask(uint64_t(1)<<(8*byte+bit),form.row_map));
                for(unsigned x=1;x<256;++x)
                    table[byte][x]=table[byte][x&(x-1)]|image[__builtin_ctz(x)];
            }
            const unsigned n=source.classes.size();
            for(uint64_t e:source.entries) {
                unsigned prefix=0;
                for(unsigned byte=0;byte<7;++byte)prefix|=table[byte][(e>>(8*byte))&255];
                unsigned slot=prefix*n+unsigned(e>>56);
                if(!counts[slot])touched.push_back(slot);
                if(counts[slot]==UINT32_MAX)throw std::runtime_error("histogram count overflow");
                ++counts[slot];
            }
            std::sort(touched.begin(),touched.end());
            for(unsigned slot:touched) {
                unsigned p=slot/n,c=slot%n;
                if(out.empty()||out.back().prefix!=p)out.push_back(response::Bucket{p,{},{}});
                out.back().classes.push_back({counts[slot],source.classes[c].second});
                counts[slot]=0;
            }
            touched.clear();
        }
        histogram_seconds+=response::now()-start;
        return out;
    }
};
