// Exact CPU cost-model acceleration, not a distribution join implementation.
struct CutTileIndex {
    struct Class {uint64_t n8,n16;unsigned orbit;};
    struct Bucket {unsigned prefix,begin,end;};
    std::vector<Bucket> buckets;
    std::vector<Class> classes;
    std::array<uint64_t,256> occupied{};
    std::array<unsigned,256> base{};
    std::array<uint64_t,4> words{};
    explicit CutTileIndex(const response::Dist& d) {
        for(const auto& b:d) {
            if(b.prefix>=16384||(!buckets.empty()&&buckets.back().prefix>=b.prefix))
                throw std::runtime_error("invalid sorted prefix metadata");
            Bucket out{b.prefix,unsigned(classes.size()),0};
            for(auto c:b.classes) {
                if(!c.count||c.count>UINT32_MAX||(c.orbit!=1&&c.orbit!=2))
                    throw std::runtime_error("invalid class count/orbit");
                classes.push_back({(c.count+7)/8,(c.count+15)/16,c.orbit});
            }
            out.end=classes.size();buckets.push_back(out);
            occupied[b.prefix/64]|=uint64_t(1)<<(b.prefix%64);
        }
        unsigned n=0;
        for(unsigned w=0;w<256;++w) {
            base[w]=n;n+=__builtin_popcountll(occupied[w]);
            if(occupied[w])words[w/64]|=uint64_t(1)<<(w%64);
        }
    }
    CutTileIndex()=default;
};

struct CutTileScorer {
    std::array<uint64_t,64> low{};
    std::array<std::array<uint64_t,4>,256> high{};
    uint64_t calls=0,word_visits=0,pair_visits=0;
    bool zeta=false;
    bool demand_only=false; // Relax join-count gate only with a complete plan.
    struct Memo {uint64_t queries=0;std::vector<uint64_t> table;};
    struct SourceMemo {uint64_t joins=0;bool planned=false;std::unordered_map<uint64_t,Memo> sizes;};
    std::unordered_map<const CutTileIndex*,SourceMemo> memo;
    uint64_t tables=0,lookup_classes=0,table_builds=0,peak_tables=0,groups=0,short_group_tables=0;
    double table_seconds=0;
    static constexpr uint64_t table_cap=2048; // 256 MiB of table payload
    void begin_group() {memo.clear();tables=0;++groups;}
    void begin_pass() {memo.clear();tables=0;lookup_classes=0;table_seconds=0;table_builds=0;peak_tables=0;groups=0;short_group_tables=0;}
    void plan_group(const CutTileIndex& a,const std::vector<const CutTileIndex*>& rights) {
        if(a.buckets.empty())return;
        auto& source=memo[&a];
        if(source.joins||source.planned)throw std::runtime_error("group already started");
        source.planned=true;
        for(auto* b:rights)if(!b->buckets.empty()) {
            ++source.joins;
            for(auto r:b->classes)++source.sizes[r.n8].queries;
        }
    }
    CutTileScorer() {
        for(unsigned p=0;p<64;++p)for(unsigned q=0;q<64;++q)
            if(!(p&q))low[p]|=uint64_t(1)<<q;
        for(unsigned p=0;p<256;++p)for(unsigned q=0;q<256;++q)
            if(!(p&q))high[p][q/64]|=uint64_t(1)<<(q%64);
    }
    uint64_t score(const CutTileIndex& a,const CutTileIndex& b) {
        ++calls;uint64_t total=0;
        std::vector<bool> covered;
        if(zeta&&!a.buckets.empty()&&!b.buckets.empty()) {
            auto& source=memo[&a];
            auto& by_size=source.sizes;
            if(!source.planned) {
                ++source.joins;
                for(auto r:b.classes)++by_size[r.n8].queries;
            }
            // Charge table construction only after enough repeated class
            // queries have accrued to plausibly amortize a 14-bit transform.
            for(auto& [n8,m]:by_size)if(m.table.empty()&&tables<table_cap&&
                    (source.joins>=4||(demand_only&&source.planned))&&
                    m.queries>=8&&m.queries*a.classes.size()>=32768) {
                double start=response::now();
                m.table.resize(16384);uint64_t n16=(n8+1)/2;
                for(auto x:a.buckets)for(unsigned i=x.begin;i<x.end;++i) {
                    auto l=a.classes[i];
                    m.table[x.prefix]+=std::min(l.n16*n8,n16*l.n8);
                }
                for(unsigned bit=1;bit<16384;bit<<=1)
                    for(unsigned start=0;start<16384;start+=2*bit)
                        for(unsigned j=0;j<bit;++j)m.table[start+bit+j]+=m.table[start+j];
                ++tables;++table_builds;peak_tables=std::max(peak_tables,tables);
                short_group_tables+=source.joins<4;
                table_seconds+=response::now()-start;
            }
            covered.resize(b.classes.size());size_t done=0;
            for(auto y:b.buckets)for(unsigned k=y.begin;k<y.end;++k) {
                auto r=b.classes[k];const auto& t=by_size[r.n8].table;
                if(t.empty())continue;
                covered[k]=true;++done;++lookup_classes;
                total+=t[16383^y.prefix];
                if(r.orbit==2) {
                    unsigned sw=((y.prefix&127)<<7)|(y.prefix>>7);
                    total+=t[16383^sw];
                }
            }
            if(done==b.classes.size())return total;
            if(!done)covered.clear(); // No indexed bucket needs filtering.
        }
        std::array<uint64_t,256> active_occupied{};
        std::array<uint64_t,4> active_words{};
        const auto* occupied=&b.occupied;
        const auto* words=&b.words;
        if(!covered.empty()) {
            for(auto y:b.buckets) {
                bool remaining=false;
                for(unsigned k=y.begin;k<y.end;++k)remaining|=!covered[k];
                if(remaining) {
                    active_occupied[y.prefix/64]|=uint64_t(1)<<(y.prefix%64);
                    active_words[y.prefix/4096]|=uint64_t(1)<<((y.prefix/64)%64);
                }
            }
            occupied=&active_occupied;words=&active_words;
        }
        for(auto x:a.buckets) {
            unsigned p=x.prefix,swapped=((p&127)<<7)|(p>>7);
            for(unsigned block=0;block<4;++block) {
                uint64_t live=(*words)[block]&(high[p>>6][block]|high[swapped>>6][block]);
                while(live) {
                    unsigned w=64*block+__builtin_ctzll(live);live&=live-1;++word_visits;
                    uint64_t forward=(p>>6)&w?0:low[p&63];
                    uint64_t reverse=(swapped>>6)&w?0:low[swapped&63];
                    uint64_t candidates=(*occupied)[w]&(forward|reverse);
                    while(candidates) {
                        unsigned bit=__builtin_ctzll(candidates);candidates&=candidates-1;++pair_visits;
                        unsigned j=b.base[w]+__builtin_popcountll(b.occupied[w]&((uint64_t(1)<<bit)-1));
                        auto y=b.buckets[j];
                        unsigned f=(forward>>bit)&1,s=(reverse>>bit)&1;
                        for(unsigned i=x.begin;i<x.end;++i)for(unsigned k=y.begin;k<y.end;++k) {
                            if(!covered.empty()&&covered[k])continue;
                            auto l=a.classes[i],r=b.classes[k];
                            uint64_t tiles=std::min(l.n16*r.n8,r.n16*l.n8);
                            total+=tiles*(f+(s&&r.orbit==2));
                        }
                    }
                }
            }
        }
        return total;
    }
};
