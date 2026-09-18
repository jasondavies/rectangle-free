// Cold/warm A/B exporter; each pass rebuilds labelled metadata and scores tiles.
#include "cut_export.hpp"

static void histogram_self_test() {
    initialise_tables();CostModel reference;CutHistogramModel candidate;
    for(unsigned i=0;i<130;++i) {
        uint32_t key=i==0?0:i==1?UINT32_MAX:uint32_t(mix64(i));
        auto a=reference.layout(key);
        for(bool projected:{false,true}) {
            candidate.projected=projected;
            auto b=candidate.layout(key);
            if(a.size()!=b.size())throw std::runtime_error("bucket count differs");
            for(size_t p=0;p<a.size();++p) {
                if(a[p].prefix!=b[p].prefix||a[p].classes.size()!=b[p].classes.size())
                    throw std::runtime_error("prefix/classes differ");
                for(size_t c=0;c<a[p].classes.size();++c)
                    if(a[p].classes[c].count!=b[p].classes[c].count||a[p].classes[c].orbit!=b[p].classes[c].orbit)
                        throw std::runtime_error("class count/orbit differs");
            }
        }
    }
    std::cout<<"HISTOGRAM_TEST exact=OK layouts=130 backends=2\n";
}

static void tile_self_test() {
    response::Engine engine{8};CutTileScorer indexed,zeta;zeta.zeta=true;
    CutTileScorer planned;planned.zeta=true;
    const uint64_t counts[]={1,7,8,9,15,16,17,31,32,33,127,128,129,1023};
    for(unsigned trial=0;trial<12;++trial) {
        response::Dist a,b;
        for(unsigned side=0;side<2;++side) {
            auto& d=side?b:a;std::set<unsigned> prefixes;
            for(unsigned p:{0,1,63,64,127,128,4095,4096,8192,16383})prefixes.insert(p);
            for(unsigned i=0;i<300;++i)prefixes.insert(mix64(i+1000*trial+100000*side)&16383);
            for(unsigned p:prefixes) {
                response::Bucket bucket{p,{},{}};
                for(unsigned c=0;c<3;++c)bucket.classes.push_back({counts[(p+c+trial)%14],1+((p+c+side)%2)});
                d.push_back(std::move(bucket));
            }
        }
        if(trial==0)a.clear();if(trial==1)b.clear();
        CutTileIndex x(a),y(b);
        response::Budget ref;engine.tile_model(a,b,ref);
        zeta.begin_pass();
        planned.begin_pass();planned.begin_group();
        planned.plan_group(x,std::vector<const CutTileIndex*>(4,&y));
        for(unsigned repeat=0;repeat<4;++repeat)
            if(indexed.score(x,y)!=ref.modeled_tiles||zeta.score(x,y)!=ref.modeled_tiles||
               planned.score(x,y)!=ref.modeled_tiles)
                throw std::runtime_error("indexed/zeta tile mismatch");
        if(trial>1&&!zeta.tables)throw std::runtime_error("test failed to exercise zeta construction");
        if(trial>1&&!planned.tables)throw std::runtime_error("test failed to exercise planned tables");
        uint64_t built=planned.table_builds;
        planned.begin_group();
        if(planned.tables||!planned.memo.empty()||planned.table_builds!=built)
            throw std::runtime_error("group cache not released or accounting reset");
        zeta.begin_pass();zeta.tables=CutTileScorer::table_cap; // Force capped fallback.
        if(zeta.score(x,y)!=ref.modeled_tiles)throw std::runtime_error("capped tile mismatch");
    }
    // Largest supported class size; all 14 coordinates and a self-swapped prefix.
    response::Dist a{{0,{},{{UINT32_MAX,1}}}},b{{16383,{},{{UINT32_MAX,2}}}};
    response::Budget ref;engine.tile_model(a,b,ref);
    if(indexed.score(CutTileIndex(a),CutTileIndex(b))!=ref.modeled_tiles)
        throw std::runtime_error("wide-count tile mismatch");
    std::cout<<"TILE_INDEX_TEST exact=OK cases=12 empty=fixed=swapped=ceilings=cap=OK\n";
}

static void demand_self_test() {
    response::Engine engine{8};
    for(auto [left_count,right_count]:std::vector<std::pair<unsigned,unsigned>>{{4095,8},{4096,8},{8192,7}}) {
        response::Dist a,b;
        for(unsigned p=0;p<left_count;++p)a.push_back({p,{},{{1,1+(p%2)}}});
        for(unsigned p=0;p<right_count;++p)b.push_back({p,{},{{8,1+(p%2)}}});
        CutTileIndex x(a),y(b);response::Budget expected;engine.tile_model(a,b,expected);
        for(bool plan:{false,true})for(bool demand:{false,true})for(bool capped:{false,true}) {
            CutTileScorer scorer;scorer.zeta=true;scorer.demand_only=demand;
            scorer.begin_pass();scorer.begin_group();
            if(plan)scorer.plan_group(x,{&y});
            if(capped)scorer.tables=CutTileScorer::table_cap;
            if(scorer.score(x,y)!=expected.modeled_tiles)throw std::runtime_error("demand cost mismatch");
            bool builds=plan&&demand&&!capped&&right_count>=8&&uint64_t(left_count)*right_count>=32768;
            if(scorer.table_builds!=unsigned(builds)||scorer.short_group_tables!=unsigned(builds))
                throw std::runtime_error("demand eligibility boundary mismatch");
        }
    }
    std::cout<<"DEMAND_TEST exact=OK eligibility=work=query=plan=cap=OK\n";
}

int main(int argc,char** argv) try {
    if(argc==2&&std::string(argv[1])=="--self-test") {histogram_self_test();return 0;}
    if(argc==2&&std::string(argv[1])=="--tile-self-test") {tile_self_test();return 0;}
    if(argc==2&&std::string(argv[1])=="--demand-self-test") {demand_self_test();return 0;}
    if(argc!=5)throw std::runtime_error("usage: INPUT_TSV SHORTLIST_TSV OUTPUT_PREFIX cached|projected|indexed|zeta|grouped|planned|demand");
    std::string mode=argv[4];
    if(mode!="cached"&&mode!="projected"&&mode!="indexed"&&mode!="zeta"&&mode!="grouped"&&mode!="planned"&&mode!="demand")throw std::runtime_error("invalid backend");
    std::string paths[2]={std::string(argv[3])+".cold.jsonl",std::string(argv[3])+".warm.jsonl"};
    for(auto& p:paths)if(std::ifstream(p).good())throw std::runtime_error("output already exists");
    initialise_tables();CutHistogramModel model;model.projected=mode!="cached";model.indexed=mode!="cached"&&mode!="projected";
    model.grouped=mode=="grouped"||mode=="planned"||mode=="demand";model.planned=mode=="planned"||mode=="demand";
    model.scorer.zeta=mode=="zeta"||model.grouped;
    model.scorer.demand_only=mode=="demand";
    for(unsigned pass=0;pass<2;++pass) {
        model.scorer.begin_pass(); // No tile table survives into the warm pass.
        std::ofstream out(paths[pass]);if(!out)throw std::runtime_error("cannot open output");
        uint64_t builds=model.builds;
        auto words=model.scorer.word_visits,pairs=model.scorer.pair_visits;
        double build=model.build_seconds,hist=model.histogram_seconds,canon=model.canonical_seconds;
        auto* previous=std::cout.rdbuf(out.rdbuf());
        try {reuse_cut::run(argv[1],"all",argv[2],&model);}
        catch(...) {std::cout.rdbuf(previous);throw;}
        std::cout.rdbuf(previous);out.close();
        if(!out)throw std::runtime_error("output write failed");
        std::cout<<"{\"pass\":"<<pass<<",\"builds\":"<<model.builds-builds
            <<",\"cached_sources\":"<<model.cache.size()<<",\"cached_entries\":"<<model.entries
            <<",\"build_seconds\":"<<model.build_seconds-build<<",\"histogram_seconds\":"<<model.histogram_seconds-hist
            <<",\"canonical_seconds\":"<<model.canonical_seconds-canon
            <<",\"word_visits\":"<<model.scorer.word_visits-words<<",\"pair_visits\":"<<model.scorer.pair_visits-pairs
            <<",\"tables\":"<<model.scorer.table_builds<<",\"peak_tables\":"<<model.scorer.peak_tables
            <<",\"groups\":"<<model.scorer.groups<<",\"table_seconds\":"<<model.scorer.table_seconds
            <<",\"short_group_tables\":"<<model.scorer.short_group_tables
            <<",\"lookup_classes\":"<<model.scorer.lookup_classes<<"}"<<std::endl;
    }
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
