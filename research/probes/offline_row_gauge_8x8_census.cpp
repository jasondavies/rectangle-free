#include <array>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_map>

#define PAIR_PROJECTION_8X8_CENSUS_NO_MAIN
#include "pair_projection_8x8_census.cpp"

namespace {

constexpr uint32_t TASK_CHUNK = 16;

struct ScheduleModel {
    const char* name;
    long double pair_cost;
    long double class_cost;
    long double tile_cost;
    bool warp_schedule;
};

constexpr std::array<ScheduleModel, 15> MODELS = {{
    {"total_tiles", 0, 0, 1, false},
    {"total_c1_t8", 0, 1, 8, false},
    {"total_p1_c4_t32", 1, 4, 32, false},
    {"warp_p1_c0_t1", 1, 0, 1, true},
    {"warp_p1_c0_t4", 1, 0, 4, true},
    {"warp_p1_c0_t16", 1, 0, 16, true},
    {"warp_p1_c0_t64", 1, 0, 64, true},
    {"warp_p1_c2_t1", 1, 2, 1, true},
    {"warp_p1_c2_t4", 1, 2, 4, true},
    {"warp_p1_c2_t16", 1, 2, 16, true},
    {"warp_p1_c2_t64", 1, 2, 64, true},
    {"warp_p1_c8_t1", 1, 8, 1, true},
    {"warp_p1_c8_t4", 1, 8, 4, true},
    {"warp_p1_c8_t16", 1, 8, 16, true},
    {"warp_p1_c8_t64", 1, 8, 64, true},
}};

struct ChunkWork {
    uint64_t pairs = 0;
    uint64_t class_pairs = 0;
    uint64_t tiles = 0;
};

struct GaugeMetric {
    U128 pairs = 0;
    U128 compatible_orientations = 0;
    U128 class_pairs = 0;
    U128 tiles = 0;
    uint64_t chunks = 0;
    uint64_t blocks = 0;
    std::array<long double, MODELS.size()> schedule{};
};

struct InputRecord {
    uint32_t left = 0;
    uint32_t right = 0;
    uint16_t file = 0;
};

struct GroupKey {
    uint32_t right = 0;
    uint16_t file = 0;
    bool operator==(const GroupKey& other) const {
        return right == other.right && file == other.file;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& value) const noexcept {
        return size_t(mix64(uint64_t(value.right) |
                            (uint64_t(value.file) << 32)));
    }
};

struct GroupProfile {
    GroupKey key;
    uint64_t records = 0;
    long double sampling_weight = 1;
    std::vector<std::array<long double, MODELS.size()>> costs;
};

struct CompleteGroupSample {
    std::vector<SampleRecord> records;
    std::unordered_map<uint32_t, long double> sampling_weights;
};

static CompleteGroupSample read_complete_group_sample(
    const std::string& path, uint64_t wanted_groups,
    uint64_t wanted_top_groups, uint64_t salt) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    const bool recognised = !std::memcmp(magic, "R8ORB01", 7) ||
                            !std::memcmp(magic, "R8SQT01", 7);
    if (!input || !recognised || columns != COLUMNS)
        throw std::runtime_error("invalid 8x8 solve file " + path);

    std::vector<SampleRecord> retained;
    retained.reserve(size_t(count / 2 + 1));
    std::unordered_map<uint32_t, uint64_t> observations;
    for (uint64_t ordinal = 0; ordinal < count; ordinal++) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated 8x8 solve file");
        if (__builtin_popcountll(record.key) * 2 > ROWS * COLUMNS) continue;
        observations[half_prefix(record.key, HALF_COLUMNS)]++;
        retained.push_back(SampleRecord{ordinal, record.key});
    }

    struct GroupCount { uint32_t right; uint64_t observations; };
    std::vector<GroupCount> groups;
    groups.reserve(observations.size());
    for (const auto& item : observations)
        groups.push_back(GroupCount{item.first, item.second});
    auto hot_first = [](const GroupCount& a, const GroupCount& b) {
        return a.observations != b.observations
            ? a.observations > b.observations : a.right < b.right;
    };
    wanted_groups = std::min<uint64_t>(wanted_groups, groups.size());
    const size_t top_count = size_t(std::min(wanted_groups,
                                             wanted_top_groups));
    const size_t random_count = size_t(wanted_groups) - top_count;
    std::partial_sort(groups.begin(), groups.begin() + top_count, groups.end(),
                      hot_first);

    CompleteGroupSample result;
    result.sampling_weights.reserve(size_t(wanted_groups * 2 + 1));
    for (size_t index = 0; index < top_count; index++)
        result.sampling_weights.emplace(groups[index].right, 1);
    auto hash_first = [salt](const GroupCount& a, const GroupCount& b) {
        uint64_t ah = mix64(uint64_t(a.right) ^ salt);
        uint64_t bh = mix64(uint64_t(b.right) ^ salt);
        return ah != bh ? ah < bh : a.right < b.right;
    };
    std::partial_sort(groups.begin() + top_count,
                      groups.begin() + top_count + random_count,
                      groups.end(), hash_first);
    const uint64_t random_population = groups.size() - top_count;
    const long double random_weight = random_count
        ? static_cast<long double>(random_population) / random_count : 1;
    for (size_t index = 0; index < random_count; index++)
        result.sampling_weights.emplace(groups[top_count + index].right,
                                        random_weight);
    result.records.reserve(size_t(wanted_groups * 4));
    for (const SampleRecord& record : retained) {
        uint32_t right = half_prefix(record.key, HALF_COLUMNS);
        if (result.sampling_weights.count(right))
            result.records.push_back(record);
    }
    std::cerr << "COMPLETE_GROUP_SAMPLE path=" << path
              << " file_records=" << count
              << " retained_records=" << retained.size()
              << " population_groups=" << observations.size()
              << " selected_groups=" << result.sampling_weights.size()
              << " selected_records=" << result.records.size()
              << " top_groups=" << top_count
              << " random_groups=" << random_count
              << " random_weight=" << double(random_weight) << '\n';
    return result;
}

static void add_metric(GaugeMetric& destination, const GaugeMetric& source) {
    destination.pairs += source.pairs;
    destination.compatible_orientations += source.compatible_orientations;
    destination.class_pairs += source.class_pairs;
    destination.tiles += source.tiles;
    destination.chunks += source.chunks;
    destination.blocks += source.blocks;
    for (size_t model = 0; model < MODELS.size(); model++)
        destination.schedule[model] += source.schedule[model];
}

static std::vector<uint32_t> enumerate_gauges() {
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
                            result.push_back(mask | (uint32_t(1) <<
                                g_pair_index[first][second]));
                        }
                    }
                }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.size() != 420)
        throw std::logic_error("K4-disjoint-edge gauge orbit is incomplete");
    return result;
}

static uint64_t class_tiles(const ProjectionClass& left,
                            const ProjectionClass& right) {
    return predicate_bmma_tiles(uint32_t(left.suffixes.size()),
                                uint32_t(right.suffixes.size()));
}

static GaugeMetric distribution_metric(const ProjectionDistribution& left,
                                       const ProjectionDistribution& right) {
    GaugeMetric result;
    result.blocks = 1;
    std::vector<ChunkWork> chunks;
    const uint64_t tasks = uint64_t(left.buckets.size()) * right.buckets.size();
    chunks.reserve(size_t((tasks + TASK_CHUNK - 1) / TASK_CHUNK));
    for (uint64_t begin = 0; begin < tasks; begin += TASK_CHUNK) {
        ChunkWork chunk;
        uint64_t end = std::min(tasks, begin + TASK_CHUNK);
        chunk.pairs = end - begin;
        for (uint64_t task = begin; task < end; task++) {
            uint32_t li = uint32_t(task / right.buckets.size());
            uint32_t ri = uint32_t(task % right.buckets.size());
            const ProjectionBucket& lhs = left.buckets[li];
            const ProjectionBucket& rhs = right.buckets[ri];
            bool forward = !(lhs.prefix & rhs.prefix);
            bool swapped = !(lhs.prefix & swap_prefix_planes(rhs.prefix));
            if (!forward && !swapped) continue;
            for (uint32_t ai = 0; ai < lhs.class_count; ai++) {
                const ProjectionClass& a = left.classes[lhs.class_offset + ai];
                for (uint32_t bi = 0; bi < rhs.class_count; bi++) {
                    const ProjectionClass& b =
                        right.classes[rhs.class_offset + bi];
                    uint64_t orientations = uint64_t(forward) +
                        uint64_t(swapped && b.orbit_size == 2);
                    if (!orientations) continue;
                    chunk.class_pairs++;
                    chunk.tiles += orientations * class_tiles(a, b);
                    result.compatible_orientations += orientations;
                }
            }
        }
        chunks.push_back(chunk);
        result.pairs += chunk.pairs;
        result.class_pairs += chunk.class_pairs;
        result.tiles += chunk.tiles;
    }
    result.chunks += chunks.size();
    for (size_t model = 0; model < MODELS.size(); model++) {
        if (!MODELS[model].warp_schedule) {
            result.schedule[model] =
                MODELS[model].pair_cost * static_cast<long double>(result.pairs) +
                MODELS[model].class_cost *
                    static_cast<long double>(result.class_pairs) +
                MODELS[model].tile_cost * static_cast<long double>(result.tiles);
            continue;
        }
        std::array<long double, 8> warp_load{};
        for (const ChunkWork& chunk : chunks) {
            size_t warp = size_t(std::min_element(warp_load.begin(),
                                                  warp_load.end()) -
                                 warp_load.begin());
            warp_load[warp] += MODELS[model].pair_cost * chunk.pairs +
                               MODELS[model].class_cost * chunk.class_pairs +
                               MODELS[model].tile_cost * chunk.tiles;
        }
        result.schedule[model] =
            *std::max_element(warp_load.begin(), warp_load.end());
    }
    return result;
}

static std::string mask_string(uint32_t mask) {
    std::ostringstream output;
    bool comma = false;
    for (unsigned first = 0; first < ROWS; first++)
        for (unsigned second = first + 1; second < ROWS; second++) {
            if (!(mask & (uint32_t(1) << g_pair_index[first][second])))
                continue;
            if (comma) output << ',';
            output << first << second;
            comma = true;
        }
    return output.str();
}

static std::vector<size_t> greedy_portfolio(
    const std::vector<GroupProfile>& groups,
    const std::vector<size_t>& training, size_t candidate_count,
    size_t model, size_t wanted) {
    std::vector<long double> best(training.size(),
                                  std::numeric_limits<long double>::infinity());
    std::vector<bool> used(candidate_count);
    std::vector<size_t> result;
    while (result.size() < wanted) {
        size_t winner = 0;
        long double winner_cost = std::numeric_limits<long double>::infinity();
        for (size_t candidate = 0; candidate < candidate_count; candidate++) {
            if (used[candidate]) continue;
            long double total = 0;
            for (size_t group = 0; group < training.size(); group++) {
                const GroupProfile& profile = groups[training[group]];
                total += profile.sampling_weight * std::min(
                    best[group], profile.costs[candidate][model]);
            }
            if (total < winner_cost) {
                winner = candidate;
                winner_cost = total;
            }
        }
        used[winner] = true;
        result.push_back(winner);
        for (size_t group = 0; group < training.size(); group++) {
            best[group] = std::min(best[group],
                groups[training[group]].costs[winner][model]);
        }
    }
    return result;
}

static void report_portfolio(const std::vector<GroupProfile>& groups,
                             const std::vector<size_t>& indices,
                             const std::vector<size_t>& portfolio,
                             size_t production, size_t model,
                             const char* split) {
    long double fixed = 0, adaptive = 0, oracle = 0;
    for (size_t index : indices) {
        const GroupProfile& group = groups[index];
        fixed += group.sampling_weight * group.costs[production][model];
        long double best = std::numeric_limits<long double>::infinity();
        for (size_t candidate : portfolio)
            best = std::min(best, group.costs[candidate][model]);
        adaptive += group.sampling_weight * best;
        long double all_best = std::numeric_limits<long double>::infinity();
        for (const auto& cost : group.costs)
            all_best = std::min(all_best, cost[model]);
        oracle += group.sampling_weight * all_best;
    }
    std::cout << "OFFLINE_GAUGE_PORTFOLIO model=" << MODELS[model].name
              << " split=" << split << " size=" << portfolio.size()
              << " groups=" << indices.size()
              << " reduction=" << double(1 - adaptive / fixed)
              << " oracle_reduction=" << double(1 - oracle / fixed) << '\n';
}

static void report_edge_policy(
    const std::vector<GroupProfile>& groups,
    const std::vector<size_t>& indices,
    const std::vector<size_t>& portfolio, size_t production,
    const std::vector<size_t>& record_group,
    const std::vector<std::vector<long double>>& record_tiles,
    const char* split) {
    std::vector<bool> included(groups.size());
    for (size_t index : indices) included[index] = true;
    std::vector<std::vector<size_t>> used_by_group(groups.size());
    long double fixed = 0, adaptive = 0;
    for (size_t record = 0; record < record_group.size(); record++) {
        size_t group_index = record_group[record];
        if (!included[group_index]) continue;
        const GroupProfile& group = groups[group_index];
        fixed += group.sampling_weight * record_tiles[record][production];
        size_t winner = portfolio.front();
        for (size_t candidate : portfolio) {
            if (record_tiles[record][candidate] < record_tiles[record][winner])
                winner = candidate;
        }
        adaptive += group.sampling_weight * record_tiles[record][winner];
        std::vector<size_t>& used = used_by_group[group_index];
        if (std::find(used.begin(), used.end(), winner) == used.end())
            used.push_back(winner);
    }
    long double base_layouts = 0, selected_layouts = 0;
    for (size_t index : indices) {
        base_layouts += groups[index].sampling_weight;
        selected_layouts += groups[index].sampling_weight *
                            used_by_group[index].size();
    }
    std::cout << "OFFLINE_GAUGE_EDGE_POLICY split=" << split
              << " size=" << portfolio.size()
              << " reduction=" << double(1 - adaptive / fixed)
              << " right_layout_multiplier="
              << double(selected_layouts / base_layouts) << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " SHARD [SHARD...] [--samples N] [--all]"
                     " [--gauges MASK,...] [--complete-groups N]"
                     " [--top-groups N] [--portfolio N]\n";
        return 2;
    }
    uint64_t samples = 128;
    uint64_t complete_groups = 0;
    uint64_t top_groups = std::numeric_limits<uint64_t>::max();
    bool all_gauges = false;
    size_t maximum_portfolio = 16;
    std::vector<uint32_t> explicit_gauges;
    std::vector<std::string> paths;
    for (int argument = 1; argument < argc; argument++) {
        std::string value = argv[argument];
        if (value == "--all") {
            all_gauges = true;
        } else if (value == "--gauges") {
            if (++argument >= argc) throw std::runtime_error("missing value");
            std::stringstream masks(argv[argument]);
            std::string mask;
            while (std::getline(masks, mask, ',')) {
                if (mask.empty()) throw std::runtime_error("empty gauge");
                explicit_gauges.push_back(uint32_t(std::stoul(mask, nullptr, 0)));
            }
        } else if (value == "--samples" || value == "--complete-groups" ||
                   value == "--top-groups" || value == "--portfolio") {
            if (++argument >= argc) throw std::runtime_error("missing value");
            uint64_t parsed = std::stoull(argv[argument]);
            if (value == "--samples") samples = parsed;
            else if (value == "--complete-groups") complete_groups = parsed;
            else if (value == "--top-groups") top_groups = parsed;
            else maximum_portfolio = size_t(parsed);
        } else if (value.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option " + value);
        } else {
            paths.push_back(value);
        }
    }
    if (paths.empty() || !samples || !maximum_portfolio ||
        maximum_portfolio > 32 || (all_gauges && !explicit_gauges.empty()))
        throw std::runtime_error("invalid arguments");

    initialise_tables();
    initialise_weighted_increments();
    std::vector<SampleRecord> sampled;
    std::vector<uint16_t> sampled_file;
    std::vector<long double> sampled_group_weight;
    for (size_t file = 0; file < paths.size(); file++) {
        CompleteGroupSample complete;
        std::vector<SampleRecord> part;
        if (complete_groups) {
            uint64_t selected_top = top_groups ==
                std::numeric_limits<uint64_t>::max()
                ? complete_groups / 2 : top_groups;
            if (selected_top > complete_groups)
                throw std::runtime_error("top groups exceed complete groups");
            complete = read_complete_group_sample(
                paths[file], complete_groups, selected_top,
                UINT64_C(0x6a09e667f3bcc909) ^ file);
            part = std::move(complete.records);
        } else {
            part = read_stride_sample(paths[file], samples);
        }
        sampled.insert(sampled.end(), part.begin(), part.end());
        sampled_file.insert(sampled_file.end(), part.size(), uint16_t(file));
        for (const SampleRecord& record : part) {
            uint32_t right = half_prefix(record.key, HALF_COLUMNS);
            sampled_group_weight.push_back(complete_groups
                ? complete.sampling_weights.at(right) : 1);
        }
    }
    std::vector<uint32_t> prefixes;
    for (const SampleRecord& record : sampled) {
        prefixes.push_back(half_prefix(record.key, 0));
        prefixes.push_back(half_prefix(record.key, HALF_COLUMNS));
    }
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()),
                   prefixes.end());
    std::unordered_map<uint32_t, uint32_t> prefix_id;
    for (size_t index = 0; index < prefixes.size(); index++)
        prefix_id.emplace(prefixes[index], uint32_t(index));
    std::vector<InputRecord> records;
    for (size_t index = 0; index < sampled.size(); index++) {
        records.push_back(InputRecord{
            prefix_id.at(half_prefix(sampled[index].key, 0)),
            prefix_id.at(half_prefix(sampled[index].key, HALF_COLUMNS)),
            sampled_file[index]});
    }

    std::vector<uint32_t> gauges = enumerate_gauges();
    const std::vector<uint32_t> valid_gauges = gauges;
    const uint32_t production_pair_mask =
        ::production_mask(PREFIX_COORDINATES);
    auto production_it = std::find(gauges.begin(), gauges.end(),
                                   production_pair_mask);
    if (production_it == gauges.end())
        throw std::runtime_error("production gauge missing");
    size_t production = size_t(production_it - gauges.begin());
    if (!explicit_gauges.empty()) {
        gauges = explicit_gauges;
        std::sort(gauges.begin(), gauges.end());
        gauges.erase(std::unique(gauges.begin(), gauges.end()), gauges.end());
        for (uint32_t mask : gauges) {
            if (std::find(valid_gauges.begin(), valid_gauges.end(), mask) ==
                valid_gauges.end())
                throw std::runtime_error("explicit gauge is outside orbit");
        }
        production_it = std::find(gauges.begin(), gauges.end(),
                                  production_pair_mask);
        if (production_it == gauges.end())
            throw std::runtime_error("explicit gauges omit production gauge");
        production = size_t(production_it - gauges.begin());
    } else if (!all_gauges) {
        std::vector<uint32_t> diagnostic = {
            production_pair_mask, UINT32_C(0x804600e), UINT32_C(0x400428b)};
        gauges.clear();
        for (uint32_t mask : diagnostic) {
            if (std::find(valid_gauges.begin(), valid_gauges.end(), mask) ==
                valid_gauges.end())
                throw std::runtime_error("diagnostic gauge is outside orbit");
            gauges.push_back(mask);
        }
        production = 0;
    }

    std::unordered_map<GroupKey, size_t, GroupKeyHash> group_index;
    std::vector<GroupProfile> groups;
    std::vector<size_t> record_group(records.size());
    for (size_t index = 0; index < records.size(); index++) {
        GroupKey key{prefixes[records[index].right], records[index].file};
        auto inserted = group_index.emplace(key, groups.size());
        if (inserted.second) {
            GroupProfile group;
            group.key = key;
            group.sampling_weight = sampled_group_weight[index];
            group.costs.resize(gauges.size());
            groups.push_back(std::move(group));
        } else if (groups[inserted.first->second].sampling_weight !=
                   sampled_group_weight[index]) {
            throw std::logic_error("inconsistent group sampling weight");
        }
        size_t group = inserted.first->second;
        groups[group].records++;
        record_group[index] = group;
    }
    std::vector<std::vector<long double>> record_tiles(
        records.size(), std::vector<long double>(gauges.size()));

    double start = seconds_now();
    for (size_t gauge = 0; gauge < gauges.size(); gauge++) {
        std::vector<ProjectionPair> pairs(prefixes.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)prefixes.size(); index++)
            pairs[size_t(index)] = build_pair_local(prefixes[size_t(index)],
                                                    gauges[gauge]);
        GaugeMetric total;
        for (size_t record_index = 0; record_index < records.size();
             record_index++) {
            const InputRecord& record = records[record_index];
            GaugeMetric metric;
            add_metric(metric, distribution_metric(
                pairs[record.left].selected, pairs[record.right].selected));
            add_metric(metric, distribution_metric(
                pairs[record.left].complement,
                pairs[record.right].complement));
            add_metric(total, metric);
            for (size_t model = 0; model < MODELS.size(); model++) {
                groups[record_group[record_index]].costs[gauge][model] +=
                    metric.schedule[model];
            }
            record_tiles[record_index][gauge] = metric.schedule[0];
        }
        std::cout << std::setprecision(12)
                  << "OFFLINE_GAUGE mask=0x" << std::hex << gauges[gauge]
                  << std::dec << " edges=" << mask_string(gauges[gauge])
                  << " records=" << records.size()
                  << " blocks=" << total.blocks
                  << " pairs=" << u128_string(total.pairs)
                  << " class_pairs=" << u128_string(total.class_pairs)
                  << " tiles=" << u128_string(total.tiles);
        for (size_t model = 0; model < MODELS.size(); model++) {
            std::cout << ' ' << MODELS[model].name << '='
                      << double(total.schedule[model]);
        }
        std::cout << " elapsed=" << seconds_now() - start << '\n';
    }

    if (all_gauges || !explicit_gauges.empty()) {
        std::vector<size_t> training, test;
        for (size_t index = 0; index < groups.size(); index++) {
            bool train = paths.size() > 1
                ? size_t(groups[index].key.file) + 1 < paths.size()
                : (mix64(groups[index].key.right) & 1) == 0;
            (train ? training : test).push_back(index);
        }
        if (training.empty() || test.empty())
            throw std::runtime_error("empty train/test split");
        for (size_t model = 0; model < MODELS.size(); model++) {
            for (size_t size = 1;
                 size <= maximum_portfolio && size <= gauges.size();
                 size *= 2) {
                std::vector<size_t> portfolio = greedy_portfolio(
                    groups, training, gauges.size(), model, size);
                report_portfolio(groups, training, portfolio, production,
                                 model, "train");
                report_portfolio(groups, test, portfolio, production,
                                 model, "test");
                if (model == 0) {
                    report_edge_policy(groups, training, portfolio,
                                       production, record_group, record_tiles,
                                       "train");
                    report_edge_policy(groups, test, portfolio, production,
                                       record_group, record_tiles, "test");
                }
                std::cout << "OFFLINE_GAUGE_MEMBERS model="
                          << MODELS[model].name << " size=" << size;
                for (size_t candidate : portfolio)
                    std::cout << " 0x" << std::hex << gauges[candidate]
                              << std::dec;
                std::cout << '\n';
            }
        }
    }
    std::cout << "OFFLINE_GAUGE_DONE gauges=" << gauges.size()
              << " records=" << records.size() << " groups=" << groups.size()
              << " seconds=" << seconds_now() - start << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
