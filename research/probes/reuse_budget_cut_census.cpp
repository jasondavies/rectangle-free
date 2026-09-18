// Research-only candidate exporter: actual production quotient/layout model.
#define SHARED_COLUMN_FAMILY_CENSUS_NO_MAIN
#include "shared_column_family_census.cpp"
#ifdef REUSE_CUT_HISTOGRAM_MODEL
#include "cut_histogram_model.hpp"
using CutCostModel=CutHistogramModel;
#else
using CutCostModel=CostModel;
#endif

namespace reuse_cut {
struct Pair {
    response::Dist selected,complement;
#ifdef REUSE_CUT_HISTOGRAM_MODEL
    std::unique_ptr<CutTileIndex> selected_index,complement_index;
#endif
    uint64_t entries=0,bytes=48;
};
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
static void run(const std::string& input,const std::string& menu,const std::string& filter_path="",CutCostModel* shared=nullptr) {
    auto cuts=enumerate(menu);
    std::map<std::pair<unsigned,uint64_t>,std::set<unsigned>> filter;
    if(!filter_path.empty()) {
        if(menu!="all")throw std::runtime_error("shortlisting requires the all-cut index convention");
        std::ifstream in(filter_path);std::string line;
        while(std::getline(in,line)) {
            std::istringstream f(line);unsigned source,slot;uint64_t index;
            if(!(f>>source>>index))throw std::runtime_error("invalid shortlist header");
            std::set<unsigned> slots;
            while(f>>slot)if(slot>=140||!slots.insert(slot).second)throw std::runtime_error("invalid/duplicate shortlist slot");
            if(!f.eof()||!slots.count(0)||slots.size()>9||!filter.emplace(std::make_pair(source,index),slots).second)
                throw std::runtime_error("invalid shortlist row");
        }
        if(!in.eof()||filter.empty())throw std::runtime_error("invalid shortlist file");
    }
    CutCostModel local_model;
    auto& model=shared?*shared:local_model;
    bool grouped=false;
#ifdef REUSE_CUT_HISTOGRAM_MODEL
    grouped=model.grouped;
#endif
    struct Choice {uint32_t left,right;uint64_t tiles;unsigned cut;bool transposed,reversed;};
    struct Record {unsigned source;uint64_t index,key,weight;std::vector<Choice> choices;};
    std::vector<Record> deferred;
    auto emit=[](const Record& r) {
        std::cout<<"{\"type\":\"record\",\"source\":"<<r.source<<",\"index\":"<<r.index
            <<",\"key\":\""<<r.key<<"\",\"weight\":\""<<r.weight<<"\",\"choices\":[";
        for(size_t i=0;i<r.choices.size();++i) {
            const auto& c=r.choices[i];if(i)std::cout<<',';
            std::cout<<"{\"left\":"<<c.left<<",\"right\":"<<c.right<<",\"tiles\":"<<c.tiles
                <<",\"columns\":"<<c.cut<<",\"transpose\":"<<int(c.transposed)<<",\"reverse\":"<<int(c.reversed)<<"}";
        }
        std::cout<<"]}"<<std::endl;
    };
    std::unordered_map<uint32_t,Pair> layouts;
    double layout_seconds=0,tile_seconds=0,group_schedule_seconds=0;
    auto get=[&](uint32_t raw)->const Pair& {
        auto found=layouts.find(raw);if(found!=layouts.end())return found->second;
        double start=response::now();
        Pair p;p.selected=model.layout(raw);p.complement=model.layout(~raw);
#ifdef REUSE_CUT_HISTOGRAM_MODEL
        if(model.indexed) {
            p.selected_index=std::make_unique<CutTileIndex>(p.selected);
            p.complement_index=std::make_unique<CutTileIndex>(p.complement);
        }
#endif
        for(const auto* d:{&p.selected,&p.complement})for(const auto& b:*d) {
            p.bytes+=12;
            for(auto c:b.classes){p.entries+=c.count;p.bytes+=8*c.count+16;}
        }
        std::cout<<"{\"type\":\"layout\",\"key\":"<<raw<<",\"entries\":"<<p.entries<<",\"bytes\":"<<p.bytes<<"}\n";
        auto it=layouts.emplace(raw,std::move(p)).first;
        layout_seconds+=response::now()-start;
        return it->second;
    };
    std::ifstream file(input);std::string line;unsigned records=0;
    double started=response::now();
    while(std::getline(file,line)) {
        uint64_t key,weight,index;unsigned source;std::string extra;
        std::istringstream fields(line);
        if(!(fields>>source>>index>>key>>weight)||fields>>extra||!weight||__builtin_popcountll(key)>32||++records>16384)
            throw std::runtime_error("invalid candidate input TSV");
        std::vector<Choice> choices;
        const std::set<unsigned>* slots=nullptr;
        if(!filter_path.empty()) {
            auto it=filter.find({source,index});
            if(it==filter.end())throw std::runtime_error("record missing from shortlist");
            slots=&it->second;
        }
        unsigned slot=0;
        for(auto [tr,cut]:cuts) {
            uint64_t oriented=tr?transpose(key):key;
            uint32_t l=half(oriented,cut),r=half(oriented,255^cut);
            for(unsigned rev=0;rev<2;++rev) {
                unsigned current=slot++;
                if(slots&&!slots->count(current))continue;
                uint32_t left=rev?r:l,right=rev?l:r;
                const Pair& a=get(left);const Pair& b=get(right);
                response::Budget budget(100000000000ull,300);
                double start=response::now();
#ifdef REUSE_CUT_HISTOGRAM_MODEL
                if(grouped) {
                    // Every histogram is built above in original input order.
                    // Costs are filled after grouping, without changing outputs.
                } else if(model.indexed) {
                    budget.modeled_tiles=model.scorer.score(*a.selected_index,*b.selected_index)+
                        model.scorer.score(*a.complement_index,*b.complement_index);
                } else
#endif
                {
                    model.engine.tile_model(a.selected,b.selected,budget);
                    model.engine.tile_model(a.complement,b.complement,budget);
                }
                tile_seconds+=response::now()-start;
                choices.push_back({left,right,budget.modeled_tiles,cut,tr,bool(rev)});
            }
        }
        // Candidate zero is the literal production split, with no row/column
        // normalization or side reorientation. Keep it even when another ties.
        if(assemble(choices[0].left,choices[0].right)!=key)
            throw std::runtime_error("baseline gauge changed");
        Record row{source,index,key,weight,std::move(choices)};
        if(grouped)deferred.push_back(std::move(row));else emit(row);
        if(records%64==0)std::cerr<<"records="<<records<<" layouts="<<layouts.size()<<" seconds="<<response::now()-started<<std::endl;
    }
    if(!file.eof()||!records)throw std::runtime_error("empty/invalid input");
    if(!filter_path.empty()&&filter.size()!=records)throw std::runtime_error("unused shortlist records");
#ifdef REUSE_CUT_HISTOGRAM_MODEL
    if(grouped) {
        double start=response::now();
        std::map<uint32_t,std::vector<Choice*>> jobs;
        for(auto& row:deferred)for(auto& choice:row.choices)jobs[choice.left].push_back(&choice);
        group_schedule_seconds=response::now()-start;
        for(auto& [left,choices]:jobs)for(unsigned side=0;side<2;++side) {
            start=response::now();
            model.scorer.begin_group();
            const auto& pair=layouts.at(left);
            const auto& a=side?*pair.complement_index:*pair.selected_index;
            if(model.planned) {
                std::vector<const CutTileIndex*> rights;rights.reserve(choices.size());
                for(auto* choice:choices) {
                    const auto& right=layouts.at(choice->right);
                    rights.push_back(side?right.complement_index.get():right.selected_index.get());
                }
                model.scorer.plan_group(a,rights);
            }
            for(auto* choice:choices) {
                const auto& right=layouts.at(choice->right);
                const auto& b=side?*right.complement_index:*right.selected_index;
                choice->tiles+=model.scorer.score(a,b);
            }
            tile_seconds+=response::now()-start;
        }
        // Drop the final group's tables too; free time is part of the pass.
        model.scorer.memo.clear();model.scorer.tables=0;
        for(const auto& row:deferred)emit(row);
    }
#endif
    std::cout<<"{\"type\":\"complete\",\"records\":"<<records<<",\"layouts\":"<<layouts.size()
        <<",\"cuts\":"<<cuts.size()<<",\"seconds\":"<<response::now()-started
        <<",\"layout_seconds\":"<<layout_seconds<<",\"tile_seconds\":"<<tile_seconds
        <<",\"group_schedule_seconds\":"<<group_schedule_seconds<<"}"<<std::endl;
}
}
#ifndef REUSE_BUDGET_CUT_NO_MAIN
int main(int argc,char** argv) try {
    initialise_tables();
    if(argc==2&&std::string(argv[1])=="--self-test"){reuse_cut::self_test();return 0;}
    if(argc!=3&&argc!=4)throw std::runtime_error("usage: INPUT_TSV four|all [SHORTLIST_TSV] | --self-test");
    reuse_cut::run(argv[1],argv[2],argc==4?argv[3]:"");
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
#endif
