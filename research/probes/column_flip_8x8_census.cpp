// Isolated CPU research gate; does not change the production solver or corpus.
#define PAIR_PROJECTION_8X8_CENSUS_NO_MAIN
#include "pair_projection_8x8_census.cpp"
#include "column_flip_parity.hpp"

namespace {
struct HalfModel {
    column_flip::Family family;
    ProjectionDistribution baseline;
    uint64_t representatives=0;
};

static HalfModel make_model(uint32_t prefix,uint64_t cap) {
    HalfModel model;
    model.family=column_flip::build(prefix,cap);
    model.baseline=build_distribution(prefix,false,production_mask(7));
    for(const auto& c:model.baseline.classes)model.representatives+=c.suffixes.size();
    if(!model.family.capped) {
        std::unordered_map<uint64_t,uint64_t> expanded;
        for(const auto& t:model.family.entries)
            for(uint64_t mask:column_flip::expand(t))expanded[mask]+=t.weight;
        const auto reference=build_full_weighted_distribution(prefix,false);
        if(expanded.size()!=reference.size())throw std::runtime_error("support census mismatch");
        for(const auto& entry:reference)
            if(expanded.at(entry.mask)!=entry.weight)
                throw std::runtime_error("exact distribution weight mismatch");
    }
    return model;
}

static void check_pair(const column_flip::Template& a,const column_flip::Template& b) {
    unsigned expected=0;
    for(uint64_t u:column_flip::expand(a))for(uint64_t v:column_flip::expand(b))
        expected+=!(u&v);
    column_flip::Work work;
    if(column_flip::orientations(a,b,work)!=expected)
        throw std::runtime_error("parity join differs from Cartesian expansion");
}

static U128 baseline_join(const ProjectionDistribution& a,const ProjectionDistribution& b) {
    U128 result=0;
    for(const auto& ba:a.buckets)for(const auto& bb:b.buckets) {
        bool forward=!(ba.prefix&bb.prefix);
        bool swapped=!(ba.prefix&swap_prefix_planes(bb.prefix));
        if(!forward&&!swapped)continue;
        for(unsigned i=0;i<ba.class_count;++i)for(unsigned j=0;j<bb.class_count;++j) {
            const auto& ca=a.classes[ba.class_offset+i];
            const auto& cb=b.classes[bb.class_offset+j];
            uint64_t count=0;
            for(uint64_t u:ca.suffixes)for(uint64_t v:cb.suffixes) {
                if(forward)count+=!(u&v);
                if(swapped&&cb.orbit_size==2)count+=!(u&swap_suffix_planes(v));
            }
            result+=U128(count)*ca.orbit_size*ca.weight*cb.weight;
        }
    }
    return result;
}

struct RecordStats {
    U128 tiles=0,covered_tiles=0,template_pairs=0;
    long double estimated_tests=0,estimated_equations=0,estimated_fixed_survivors=0;
    uint64_t representatives=0,templates=0,capped=0,verified_joins=0;
    double parity_seconds=0,reference_seconds=0;
    std::array<uint64_t,5> component_hist{};
    column_flip::Work sample;
};

static RecordStats inspect(uint64_t key,uint64_t cap,uint64_t samples,uint64_t exact_cap) {
    RecordStats out;
    for(unsigned complement=0;complement<2;++complement) {
        uint64_t mask=complement?~key:key;
        auto a=make_model(half_prefix(mask,0),cap);
        auto b=make_model(half_prefix(mask,4),cap);
        Stats baseline;
        std::priority_queue<HeavyTask,std::vector<HeavyTask>,MinTiles> heap;
        distribution_stats(a.baseline,b.baseline,baseline,heap,0);
        out.tiles+=baseline.tiles;
        if(a.family.capped||b.family.capped){++out.capped;continue;}
        out.covered_tiles+=baseline.tiles;
        out.representatives+=a.representatives+b.representatives;
        out.templates+=a.family.entries.size()+b.family.entries.size();
        for(const auto& t:a.family.entries)++out.component_hist[t.key.count];
        for(const auto& t:b.family.entries)++out.component_hist[t.key.count];
        uint64_t products=uint64_t(a.family.entries.size())*b.family.entries.size();
        out.template_pairs+=products;
        column_flip::Work work;
        uint64_t draws=std::min(samples,products);
        for(uint64_t i=0;i<draws;++i) {
            // Uniform deterministic pair sampling with replacement; never
            // present these sampled rejection frequencies as exact censuses.
            uint64_t slot=mix64(key^mix64(i+1)^uint64_t(complement))%products;
            const auto& lhs=a.family.entries[slot/b.family.entries.size()];
            const auto& rhs=b.family.entries[slot%b.family.entries.size()];
            column_flip::orientations(lhs,rhs,work);
            if(i<32)check_pair(lhs,rhs);
        }
        if(draws) {
            long double scale=static_cast<long double>(products)/draws;
            out.estimated_tests+=scale*work.component_tests;
            out.estimated_equations+=scale*work.equations;
            out.estimated_fixed_survivors+=scale*(work.pairs-work.fixed_reject);
        }
        out.sample.pairs+=work.pairs;
        out.sample.fixed_reject+=work.fixed_reject;
        out.sample.conflict_reject+=work.conflict_reject;
        out.sample.cycle_reject+=work.cycle_reject;
        out.sample.accepted+=work.accepted;
        if(products<=exact_cap) {
            U128 answer=0;
            double start=omp_get_wtime();
            for(const auto& lhs:a.family.entries)for(const auto& rhs:b.family.entries) {
                column_flip::Work ignored;
                answer+=U128(column_flip::orientations(lhs,rhs,ignored))*lhs.weight*rhs.weight;
            }
            out.parity_seconds+=omp_get_wtime()-start;
            start=omp_get_wtime();
            U128 reference=baseline_join(a.baseline,b.baseline);
            out.reference_seconds+=omp_get_wtime()-start;
            if(answer!=reference)
                throw std::runtime_error("complete weighted join mismatch");
            ++out.verified_joins;
        }
    }
    return out;
}

static void self_test() {
    // Exhaust all 3-row, 4-column active masks (embedded in eight rows).
    // This includes empty/singleton columns, fixed-plane supports and
    // constraints which connect several column-orientation variables.
    for(uint32_t prefix=0;prefix<(1U<<12);++prefix)
        make_model(prefix,1U<<20);
    if(!column_flip::build(0xffffffffU,1).capped)
        throw std::runtime_error("raw-template cap not enforced");
    column_flip::Parity odd_cycle(3);
    if(!odd_cycle.add(0,1,1)||!odd_cycle.add(1,2,1)||odd_cycle.add(0,2,1))
        throw std::runtime_error("odd parity cycle not rejected");
    column_flip::Parity even_cycle(3);
    if(!even_cycle.add(0,1,1)||!even_cycle.add(1,2,1)||!even_cycle.add(0,2,0)
       ||even_cycle.components!=1)
        throw std::runtime_error("consistent parity cycle rejected");
    for(uint32_t prefix:{0U,0x11111111U,0x01234567U,0x00000123U,0x12345678U}) {
        auto model=make_model(prefix,1U<<20);
        if(model.family.capped)throw std::runtime_error("self-test unexpectedly capped");
        for(size_t i=0;i<std::min<size_t>(model.family.entries.size(),32);++i)
            for(size_t j=0;j<std::min<size_t>(model.family.entries.size(),32);++j)
                check_pair(model.family.entries[i],model.family.entries[j]);
    }
    // A sparse outer-mask fixture supplies a complete weighted join check.
    auto result=inspect(0x0102030401020304ULL,1U<<20,64,100000);
    if(!result.verified_joins)throw std::runtime_error("no complete self-test join");
    std::cout<<"COLUMN_FLIP_SELF_TEST exact=OK\n";
}
} // namespace

int main(int argc,char** argv) try {
    initialise_tables();initialise_weighted_increments();
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc<2||argc>6)throw std::runtime_error(
        "usage: column_flip_8x8_census SHARD[,SHARD...] [RECORDS_PER_SHARD=4] "
        "[RAW_TEMPLATE_CAP=1048576] [PAIR_SAMPLES=4096] [EXACT_PAIR_CAP=100000]");
    uint64_t count=argc>2?std::stoull(argv[2]):4;
    uint64_t cap=argc>3?std::stoull(argv[3]):1U<<20;
    uint64_t samples=argc>4?std::stoull(argv[4]):4096;
    uint64_t exact_cap=argc>5?std::stoull(argv[5]):100000;
    if(!count||!cap||!samples)throw std::runtime_error("positive limits required");
    std::vector<SampleRecord> records;
    for(const auto& path:split_paths(argv[1])) {
        auto part=read_stride_sample(path,count);records.insert(records.end(),part.begin(),part.end());
    }
    std::vector<RecordStats> results(records.size());
    std::vector<std::string> errors(records.size());
    double start=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,1)
    for(size_t i=0;i<records.size();++i)try {
        results[i]=inspect(records[i].key,cap,samples,exact_cap);
    }catch(const std::exception& e){errors[i]=e.what();}
    RecordStats total;
    for(size_t i=0;i<results.size();++i) {
        if(!errors[i].empty())throw std::runtime_error(errors[i]);
        const auto& s=results[i];
        total.tiles+=s.tiles;total.covered_tiles+=s.covered_tiles;
        total.template_pairs+=s.template_pairs;
        total.estimated_tests+=s.estimated_tests;total.estimated_equations+=s.estimated_equations;
        total.estimated_fixed_survivors+=s.estimated_fixed_survivors;
        total.representatives+=s.representatives;total.templates+=s.templates;
        total.capped+=s.capped;total.verified_joins+=s.verified_joins;
        total.parity_seconds+=s.parity_seconds;total.reference_seconds+=s.reference_seconds;
        for(unsigned c=0;c<5;++c)total.component_hist[c]+=s.component_hist[c];
        total.sample.pairs+=s.sample.pairs;total.sample.fixed_reject+=s.sample.fixed_reject;
        total.sample.conflict_reject+=s.sample.conflict_reject;
        total.sample.cycle_reject+=s.sample.cycle_reject;total.sample.accepted+=s.sample.accepted;
        std::cout<<"COLUMN_FLIP_RECORD index="<<i<<" key="<<std::hex<<records[i].key<<std::dec
                 <<" tiles="<<u128_string(s.tiles)
                 <<" covered_tiles="<<u128_string(s.covered_tiles)
                 <<" templates="<<s.templates<<" representatives="<<s.representatives
                 <<" pairs="<<u128_string(s.template_pairs)
                 <<" component_tests_est="<<s.estimated_tests<<" capped="<<s.capped<<"\n";
    }
    std::cout<<std::setprecision(12)<<"COLUMN_FLIP_TOTAL records="<<records.size()
             <<" tiles="<<u128_string(total.tiles)<<" covered_tiles="<<u128_string(total.covered_tiles)
             <<" pairs="<<u128_string(total.template_pairs)
             <<" component_tests_est="<<total.estimated_tests
             <<" equations_est="<<total.estimated_equations
             <<" fixed_survivors_est="<<total.estimated_fixed_survivors
             <<" templates="<<total.templates<<" representatives="<<total.representatives
             <<" template_bytes="<<sizeof(column_flip::Template)
             <<" sampled_pairs="<<total.sample.pairs<<" sampled_fixed_reject="<<total.sample.fixed_reject
             <<" sampled_conflict_reject="<<total.sample.conflict_reject
             <<" sampled_cycle_reject="<<total.sample.cycle_reject
             <<" sampled_accepted="<<total.sample.accepted
             <<" capped_joins="<<total.capped<<" complete_join_checks="<<total.verified_joins
             <<" parity_cpu_seconds="<<total.parity_seconds
             <<" reference_cpu_seconds="<<total.reference_seconds
             <<" components_0="<<total.component_hist[0]
             <<" components_1="<<total.component_hist[1]
             <<" components_2="<<total.component_hist[2]
             <<" components_3="<<total.component_hist[3]
             <<" components_4="<<total.component_hist[4]
             <<" seconds="<<omp_get_wtime()-start<<" exact_distributions=OK\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
