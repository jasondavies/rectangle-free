#include <limits>
#include <unordered_map>

#define PREFIX_BMMA_COST_CENSUS_NO_MAIN
#include "prefix_bmma_cost_census.cpp"

namespace {

struct PortfolioMetric {
    long double tiles = 0;
    long double tested_bucket_pairs = 0;
    long double class_pairs = 0;
    long double suffix_cells = 0;
};

struct PortfolioProfile {
    size_t dataset = 0;
    uint32_t right = 0;
    std::vector<PortfolioMetric> costs;
};

static PortfolioMetric metric(const Cost& cost) {
    return PortfolioMetric{u128_long_double(cost.tiles),
                           static_cast<long double>(cost.tested_bucket_pairs),
                           u128_long_double(cost.class_pairs),
                           u128_long_double(cost.suffix_cells)};
}

static void add(PortfolioMetric& destination,
                const PortfolioMetric& source) {
    destination.tiles += source.tiles;
    destination.tested_bucket_pairs += source.tested_bucket_pairs;
    destination.class_pairs += source.class_pairs;
    destination.suffix_cells += source.suffix_cells;
}

static std::vector<uint32_t> enumerate_k4_disjoint_edge() {
    std::vector<uint32_t> result;
    for (unsigned a = 0; a < ROWS; a++)
        for (unsigned b = a + 1; b < ROWS; b++)
            for (unsigned c = b + 1; c < ROWS; c++)
                for (unsigned d = c + 1; d < ROWS; d++) {
                    std::array<unsigned, 4> clique{a, b, c, d};
                    std::array<bool, ROWS> inside{};
                    uint32_t mask = 0;
                    for (unsigned row : clique) inside[row] = true;
                    for (unsigned i = 0; i < clique.size(); i++)
                        for (unsigned j = i + 1; j < clique.size(); j++)
                            mask |= uint32_t(1) <<
                                    g_pair_index[clique[i]][clique[j]];
                    for (unsigned first = 0; first < ROWS; first++) {
                        if (inside[first]) continue;
                        for (unsigned second = first + 1; second < ROWS;
                             second++) {
                            if (inside[second]) continue;
                            result.push_back(
                                mask | (uint32_t(1) <<
                                        g_pair_index[first][second]));
                        }
                    }
                }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.size() != 420)
        throw std::logic_error("K4-disjoint-edge orbit is incomplete");
    return result;
}

static uint32_t attached_edge_control_mask() {
    return (uint32_t(1) << g_pair_index[0][1]) |
           (uint32_t(1) << g_pair_index[0][2]) |
           (uint32_t(1) << g_pair_index[0][3]) |
           (uint32_t(1) << g_pair_index[1][2]) |
           (uint32_t(1) << g_pair_index[1][3]) |
           (uint32_t(1) << g_pair_index[2][3]) |
           (uint32_t(1) << g_pair_index[3][4]);
}

static std::vector<Cost> evaluate_records(const Dataset& dataset,
                                          uint32_t pair_mask) {
    std::vector<CandidatePair> layouts(dataset.pairs.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)dataset.pairs.size(); index++) {
        layouts[size_t(index)] = CandidatePair{
            bucket_distribution(dataset.pairs[size_t(index)].selected,
                                pair_mask),
            bucket_distribution(dataset.pairs[size_t(index)].complement,
                                pair_mask)};
    }
    std::vector<Cost> result(dataset.records.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long long record_index = 0;
         record_index < (long long)dataset.records.size(); record_index++) {
        const CensusRecord& record = dataset.records[size_t(record_index)];
        Cost local;
        for (unsigned component = 0; component < 2; component++) {
            const CandidateDistribution& left = component
                ? layouts[record.left].complement
                : layouts[record.left].selected;
            const CandidateDistribution& right = component
                ? layouts[record.right].complement
                : layouts[record.right].selected;
            local.tested_bucket_pairs +=
                uint64_t(left.buckets.size()) * right.buckets.size();
            for (const ClassBucket& lhs : left.buckets) {
                for (const ClassBucket& rhs : right.buckets) {
                    if (lhs.prefix & rhs.prefix) continue;
                    local.compatible_bucket_pairs++;
                    local.suffix_cells += U128(lhs.count) * rhs.count;
                    local.class_pairs +=
                        U128(lhs.classes.size()) * rhs.classes.size();
                    for (uint32_t left_count : lhs.classes)
                        for (uint32_t right_count : rhs.classes)
                            local.tiles += predicate_bmma_tiles(
                                left_count, right_count);
                }
            }
        }
        result[size_t(record_index)] = local;
    }
    return result;
}

static long double score(const PortfolioMetric& value, long double tile_cost,
                         long double bucket_cost) {
    return tile_cost * value.tiles +
           bucket_cost * value.tested_bucket_pairs;
}

static PortfolioMetric sum_candidate(
    const std::vector<PortfolioProfile>& profiles,
    const std::vector<size_t>& indices, size_t candidate) {
    PortfolioMetric result;
    for (size_t index : indices) add(result, profiles[index].costs[candidate]);
    return result;
}

static std::vector<size_t> greedy_portfolio(
    const std::vector<PortfolioProfile>& profiles,
    const std::vector<size_t>& training, size_t wanted,
    long double tile_cost, long double bucket_cost) {
    const size_t candidate_count = profiles.front().costs.size();
    std::vector<long double> best(training.size(),
                                  std::numeric_limits<long double>::infinity());
    std::vector<bool> used(candidate_count);
    std::vector<size_t> result;
    while (result.size() < wanted) {
        size_t winner = 0;
        long double winner_cost =
            std::numeric_limits<long double>::infinity();
        for (size_t candidate = 0; candidate < candidate_count; candidate++) {
            if (used[candidate]) continue;
            long double total = 0;
            for (size_t group = 0; group < training.size(); group++) {
                total += std::min(
                    best[group],
                    score(profiles[training[group]].costs[candidate],
                          tile_cost, bucket_cost));
            }
            if (total < winner_cost) {
                winner = candidate;
                winner_cost = total;
            }
        }
        used[winner] = true;
        result.push_back(winner);
        for (size_t group = 0; group < training.size(); group++) {
            best[group] = std::min(
                best[group],
                score(profiles[training[group]].costs[winner],
                      tile_cost, bucket_cost));
        }
    }
    return result;
}

static void report(const char* split,
                   const std::vector<PortfolioProfile>& profiles,
                   const std::vector<size_t>& indices,
                   const std::vector<size_t>& portfolio,
                   size_t production_candidate, long double tile_cost,
                   long double bucket_cost) {
    PortfolioMetric current;
    PortfolioMetric adaptive;
    PortfolioMetric oracle;
    long double current_score = 0;
    long double adaptive_score = 0;
    long double oracle_score = 0;
    for (size_t index : indices) {
        const PortfolioProfile& profile = profiles[index];
        const PortfolioMetric& fixed = profile.costs[production_candidate];
        current_score += score(fixed, tile_cost, bucket_cost);
        add(current, fixed);

        size_t choice = portfolio.front();
        for (size_t candidate : portfolio)
            if (score(profile.costs[candidate], tile_cost, bucket_cost) <
                score(profile.costs[choice], tile_cost, bucket_cost))
                choice = candidate;
        adaptive_score += score(profile.costs[choice], tile_cost, bucket_cost);
        add(adaptive, profile.costs[choice]);

        size_t best = 0;
        for (size_t candidate = 1; candidate < profile.costs.size(); candidate++)
            if (score(profile.costs[candidate], tile_cost, bucket_cost) <
                score(profile.costs[best], tile_cost, bucket_cost))
                best = candidate;
        oracle_score += score(profile.costs[best], tile_cost, bucket_cost);
        add(oracle, profile.costs[best]);
    }
    std::cout << std::setprecision(12)
              << "BMMA_PORTFOLIO split=" << split
              << " size=" << portfolio.size()
              << " groups=" << indices.size()
              << " score_reduction="
              << double(1 - adaptive_score / current_score)
              << " oracle_score_reduction="
              << double(1 - oracle_score / current_score)
              << " tile_reduction="
              << double(1 - adaptive.tiles / current.tiles)
              << " tested_bucket_reduction="
              << double(1 - adaptive.tested_bucket_pairs /
                                  current.tested_bucket_pairs)
              << " class_pair_reduction="
              << double(1 - adaptive.class_pairs / current.class_pairs)
              << " suffix_cell_reduction="
              << double(1 - adaptive.suffix_cells / current.suffix_cells)
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: " << argv[0]
                      << " TRAIN.orbits[,TRAIN.orbits] TEST.orbits "
                         "[SAMPLE_RECORDS_PER_SHARD=128] [PORTFOLIO=16]\n";
            return 2;
        }
        uint64_t sample_records = argc > 3 ? std::stoull(argv[3]) : 128;
        size_t maximum_portfolio = argc > 4 ? std::stoull(argv[4]) : 16;
        if (!sample_records || !maximum_portfolio || maximum_portfolio > 32)
            return 2;
        initialise_tables();
        initialise_weighted_increments();
        std::vector<std::string> paths = split(argv[1], ',');
        paths.push_back(argv[2]);
        std::vector<Dataset> datasets;
        for (const std::string& path : paths) {
            double start = seconds_now();
            datasets.push_back(build_dataset(path, sample_records));
            std::cout << "BMMA_PORTFOLIO_INPUT path=" << path
                      << " records=" << datasets.back().records.size()
                      << " distributions=" << datasets.back().pairs.size() * 2
                      << " entries=" << datasets.back().entries
                      << " seconds=" << seconds_now() - start << '\n';
        }

        std::vector<uint32_t> candidates = enumerate_k4_disjoint_edge();
        const uint32_t production_mask = ::production_mask(7);
        auto production =
            std::find(candidates.begin(), candidates.end(), production_mask);
        if (production == candidates.end())
            throw std::logic_error("production gauge is outside its orbit");
        size_t production_candidate = size_t(production - candidates.begin());

        std::vector<PortfolioProfile> profiles;
        std::vector<std::vector<size_t>> dataset_profiles(datasets.size());
        for (size_t dataset = 0; dataset < datasets.size(); dataset++) {
            std::unordered_map<uint32_t, size_t> groups;
            for (const CensusRecord& record : datasets[dataset].records) {
                if (groups.count(record.right)) continue;
                size_t index = profiles.size();
                profiles.push_back(PortfolioProfile{
                    dataset, record.right,
                    std::vector<PortfolioMetric>(candidates.size())});
                groups.emplace(record.right, index);
                dataset_profiles[dataset].push_back(index);
            }
        }

        double evaluation_start = seconds_now();
        for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
            for (size_t dataset = 0; dataset < datasets.size(); dataset++) {
                std::vector<Cost> costs =
                    evaluate_records(datasets[dataset], candidates[candidate]);
                std::unordered_map<uint32_t, size_t> group_index;
                for (size_t index : dataset_profiles[dataset])
                    group_index.emplace(profiles[index].right, index);
                for (size_t record = 0; record < costs.size(); record++) {
                    size_t group = group_index.at(
                        datasets[dataset].records[record].right);
                    add(profiles[group].costs[candidate], metric(costs[record]));
                }
            }
            if (!((candidate + 1) % 20))
                std::cerr << "BMMA_PORTFOLIO_PROGRESS candidates="
                          << candidate + 1
                          << " seconds=" << seconds_now() - evaluation_start
                          << '\n';
        }

        std::vector<size_t> training;
        for (size_t dataset = 0; dataset + 1 < datasets.size(); dataset++)
            training.insert(training.end(), dataset_profiles[dataset].begin(),
                            dataset_profiles[dataset].end());
        const std::vector<size_t>& test = dataset_profiles.back();
        std::vector<size_t> all = training;
        all.insert(all.end(), test.begin(), test.end());

        PortfolioMetric fixed =
            sum_candidate(profiles, all, production_candidate);
        std::vector<Cost> attached_records;
        PortfolioMetric attached;
        for (const Dataset& dataset : datasets) {
            attached_records =
                evaluate_records(dataset, attached_edge_control_mask());
            for (const Cost& cost : attached_records) add(attached, metric(cost));
        }
        constexpr long double fixed_gpu_seconds = 12.211704L;
        constexpr long double attached_gpu_seconds = 15.503433L;
        long double determinant =
            fixed.tiles * attached.tested_bucket_pairs -
            attached.tiles * fixed.tested_bucket_pairs;
        long double tile_cost =
            (fixed_gpu_seconds * attached.tested_bucket_pairs -
             attached_gpu_seconds * fixed.tested_bucket_pairs) /
            determinant;
        long double bucket_cost =
            (fixed.tiles * attached_gpu_seconds -
             attached.tiles * fixed_gpu_seconds) /
            determinant;
        if (!(tile_cost > 0) || !(bucket_cost > 0))
            throw std::runtime_error("BMMA calibration is not positive");
        std::cout << std::setprecision(12)
                  << "BMMA_PORTFOLIO_CALIBRATION candidates="
                  << candidates.size()
                  << " production_mask=0x" << std::hex << production_mask
                  << std::dec << " tile_cost=" << double(tile_cost)
                  << " tested_bucket_cost=" << double(bucket_cost)
                  << " fixed_tiles=" << double(fixed.tiles)
                  << " fixed_tested_buckets="
                  << double(fixed.tested_bucket_pairs) << '\n';

        for (size_t size = 1; size <= maximum_portfolio; size *= 2) {
            std::vector<size_t> portfolio = greedy_portfolio(
                profiles, training, size, tile_cost, bucket_cost);
            report("train", profiles, training, portfolio,
                   production_candidate, tile_cost, bucket_cost);
            report("test", profiles, test, portfolio,
                   production_candidate, tile_cost, bucket_cost);
            report("all", profiles, all, portfolio,
                   production_candidate, tile_cost, bucket_cost);
            std::cout << "BMMA_PORTFOLIO_MEMBERS size=" << size;
            for (size_t candidate : portfolio)
                std::cout << " 0x" << std::hex << candidates[candidate]
                          << std::dec;
            std::cout << '\n';
        }
        std::cout << "BMMA_PORTFOLIO_DONE seconds="
                  << seconds_now() - evaluation_start << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
