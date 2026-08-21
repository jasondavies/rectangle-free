#include <iomanip>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>

// Reuse the exact 8x4 support recurrence and checked R8ORB01 reader helpers.
// Renaming its diagnostic entry point keeps this oracle independent of the
// production solver while avoiding a second copy of the mathematics.
#define main prefix_hierarchy_8x8_census_unused_main
#include "prefix_hierarchy_8x8_census.cpp"
#undef main

namespace {

constexpr unsigned RESERVOIR_SIZE = 4;

struct PortfolioGroup {
    uint32_t right = 0;
    uint64_t observations = 0;
    std::vector<uint32_t> left_reservoir;
    unsigned file = 0;
};

struct SelectedGroup {
    size_t index = 0;
    double sampling_weight = 1;
};

struct GroupProfile {
    unsigned file = 0;
    uint32_t right = 0;
    double direct = 0;
    std::vector<double> costs;
    std::vector<double> uniform_right_scores;
    std::vector<double> selector_costs;
};

static std::vector<PortfolioGroup> read_portfolio_groups(
    const std::vector<std::string>& paths,
    std::vector<std::vector<size_t>>& by_file) {
    std::vector<PortfolioGroup> groups;
    by_file.resize(paths.size());
    for (size_t file = 0; file < paths.size(); file++) {
        double start = seconds_now();
        std::ifstream input(paths[file], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open " + paths[file]);
        char magic[8];
        uint32_t columns = 0;
        uint64_t count = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (!input || std::memcmp(magic, "R8ORB01", 7) || columns != COLUMNS)
            throw std::runtime_error("invalid 8x8 solve file: " + paths[file]);
        std::unordered_map<uint32_t, size_t> indices;
        indices.reserve(size_t(count / 2));
        uint64_t retained = 0;
        for (uint64_t ordinal = 0; ordinal < count; ordinal++) {
            OrbitRecord record{};
            input.read(reinterpret_cast<char*>(&record), sizeof(record));
            if (!input) throw std::runtime_error("truncated 8x8 solve file");
            unsigned cells = __builtin_popcountll(record.key);
            if (cells * 2 > ROWS * COLUMNS) continue;
            retained++;
            uint32_t right = half_prefix(record.key, HALF_COLUMNS);
            uint32_t left = half_prefix(record.key, 0);
            auto found = indices.find(right);
            if (found == indices.end()) {
                size_t index = groups.size();
                groups.push_back(PortfolioGroup{right, 0, {}, unsigned(file)});
                by_file[file].push_back(index);
                found = indices.emplace(right, index).first;
            }
            PortfolioGroup& group = groups[found->second];
            group.observations++;
            if (group.left_reservoir.size() < RESERVOIR_SIZE) {
                group.left_reservoir.push_back(left);
            } else {
                uint64_t slot = mix64(uint64_t(right) ^ group.observations) %
                                group.observations;
                if (slot < RESERVOIR_SIZE)
                    group.left_reservoir[size_t(slot)] = left;
            }
        }
        std::cerr << "GROUP_READ file=" << file << " records=" << count
                  << " retained=" << retained
                  << " groups=" << by_file[file].size()
                  << " seconds=" << seconds_now() - start << '\n';
    }
    return groups;
}

static std::vector<SelectedGroup> select_groups(
    const std::vector<PortfolioGroup>& groups,
    const std::vector<size_t>& source, size_t wanted) {
    wanted = std::min(wanted, source.size());
    const size_t top_count = wanted / 2;
    std::vector<size_t> ordered = source;
    std::partial_sort(ordered.begin(), ordered.begin() + top_count,
                      ordered.end(), [&](size_t a, size_t b) {
        if (groups[a].observations != groups[b].observations)
            return groups[a].observations > groups[b].observations;
        return mix64(groups[a].right) < mix64(groups[b].right);
    });
    std::vector<SelectedGroup> result;
    std::unordered_set<size_t> used;
    for (size_t index = 0; index < top_count; index++) {
        result.push_back(SelectedGroup{ordered[index], 1});
        used.insert(ordered[index]);
    }
    std::vector<size_t> remainder;
    for (size_t index : source) if (!used.count(index)) remainder.push_back(index);
    const size_t random_count = wanted - top_count;
    std::partial_sort(remainder.begin(), remainder.begin() + random_count,
                      remainder.end(), [&](size_t a, size_t b) {
        return mix64(groups[a].right) < mix64(groups[b].right);
    });
    const double weight = random_count
        ? double(source.size() - top_count) / random_count : 1;
    for (size_t index = 0; index < random_count; index++)
        result.push_back(SelectedGroup{remainder[index], weight});
    return result;
}

// The accepted seven-coordinate prefix is a K4 plus one edge from the clique
// to an outside vertex.  Its S8 orbit has 70 * 4 * 4 = 1,120 members.
static std::vector<uint32_t> enumerate_candidates() {
    std::vector<uint32_t> result;
    result.reserve(1120);
    for (unsigned a = 0; a < ROWS; a++)
        for (unsigned b = a + 1; b < ROWS; b++)
            for (unsigned c = b + 1; c < ROWS; c++)
                for (unsigned d = c + 1; d < ROWS; d++) {
                    std::array<unsigned, 4> clique{a, b, c, d};
                    uint32_t base = 0;
                    for (unsigned i = 0; i < 4; i++)
                        for (unsigned j = i + 1; j < 4; j++)
                            base |= uint32_t(1) <<
                                    g_pair_index[clique[i]][clique[j]];
                    for (unsigned inside : clique)
                        for (unsigned outside = 0; outside < ROWS; outside++) {
                            if (outside == a || outside == b ||
                                outside == c || outside == d) continue;
                            unsigned first = std::min(inside, outside);
                            unsigned second = std::max(inside, outside);
                            result.push_back(base | (uint32_t(1) <<
                                g_pair_index[first][second]));
                        }
                }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.size() != 1120)
        throw std::logic_error("K4+edge candidate enumeration is incomplete");
    return result;
}

static std::string candidate_string(uint32_t candidate) {
    std::ostringstream output;
    bool comma = false;
    for (unsigned first = 0; first < ROWS; first++)
        for (unsigned second = first + 1; second < ROWS; second++) {
            if (!(candidate & (uint32_t(1) << g_pair_index[first][second])))
                continue;
            if (comma) output << ',';
            output << first << second;
            comma = true;
        }
    return output.str();
}

static uint64_t random_bounded(uint64_t& state, uint64_t bound) {
    state = mix64(state + UINT64_C(0x9e3779b97f4a7c15));
    return bound ? state % bound : 0;
}

static void sample_component(const std::vector<uint64_t>& left,
                             const std::vector<uint64_t>& right,
                             double observation_scale, uint64_t wanted,
                             const std::vector<uint32_t>& candidates,
                             std::vector<double>& costs, double& direct,
                             uint64_t& state) {
    if (left.empty() || right.empty()) return;
    const long double product = (long double)left.size() * right.size();
    const uint64_t samples = uint64_t(std::min<long double>(product, wanted));
    const double weight = double(product * observation_scale / samples);
    direct += double(product * observation_scale);
    for (uint64_t sample = 0; sample < samples; sample++) {
        uint64_t lhs = left[size_t(random_bounded(state, left.size()))];
        uint64_t rhs = right[size_t(random_bounded(state, right.size()))];
        uint64_t overlap = lhs & rhs;
        uint32_t conflicts = uint32_t(overlap) | uint32_t(overlap >> PAIRS);
#pragma omp simd
        for (size_t candidate = 0; candidate < candidates.size(); candidate++)
            costs[candidate] +=
                (conflicts & candidates[candidate]) ? 0.0 : weight;
    }
}

static GroupProfile profile_group(
    const PortfolioGroup& group, double selection_weight, uint64_t samples,
    uint64_t selector_samples,
    const std::vector<uint32_t>& candidates,
    const std::unordered_map<uint32_t, size_t>& prefix_index,
    const std::vector<DistributionPair>& distributions) {
    GroupProfile result;
    result.file = group.file;
    result.right = group.right;
    result.costs.assign(candidates.size(), 0);
    result.uniform_right_scores.assign(candidates.size(), 0);
    result.selector_costs.assign(candidates.size(), 0);
    const DistributionPair& right = distributions.at(prefix_index.at(group.right));
    auto score_right = [&](const std::vector<uint64_t>& entries) {
        const size_t wanted = std::min<size_t>(entries.size(), 4096);
        if (!wanted) return;
        for (size_t sample = 0; sample < wanted; sample++) {
            uint64_t mask = entries[(sample * entries.size()) / wanted];
#pragma omp simd
            for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
                uint64_t both = uint64_t(candidates[candidate]) |
                                (uint64_t(candidates[candidate]) << PAIRS);
                result.uniform_right_scores[candidate] +=
                    std::ldexp(1.0, -__builtin_popcountll(mask & both));
            }
        }
    };
    score_right(right.selected);
    score_right(right.complement);
    const double observation_scale = selection_weight *
        double(group.observations) / group.left_reservoir.size();
    uint64_t state = mix64(uint64_t(group.right) ^ (uint64_t(group.file) << 32));
    for (uint32_t left_key : group.left_reservoir) {
        const DistributionPair& left =
            distributions.at(prefix_index.at(left_key));
        sample_component(left.selected, right.selected, observation_scale,
                         samples, candidates, result.costs, result.direct, state);
        sample_component(left.complement, right.complement, observation_scale,
                         samples, candidates, result.costs, result.direct, state);
        double ignored_direct = 0;
        uint64_t selector_state = mix64(state ^ UINT64_C(0xa0761d6478bd642f));
        sample_component(left.selected, right.selected, observation_scale,
                         selector_samples, candidates, result.selector_costs,
                         ignored_direct, selector_state);
        sample_component(left.complement, right.complement, observation_scale,
                         selector_samples, candidates, result.selector_costs,
                         ignored_direct, selector_state);
    }
    return result;
}

static void report_sample_selector(
    const std::vector<GroupProfile>& profiles, const std::vector<size_t>& test,
    const std::vector<size_t>& portfolio, size_t current_candidate) {
    double current = 0, selected = 0;
    for (size_t index : test) {
        current += profiles[index].costs[current_candidate];
        size_t choice = portfolio.front();
        for (size_t candidate : portfolio)
            if (profiles[index].selector_costs[candidate] <
                profiles[index].selector_costs[choice]) choice = candidate;
        selected += profiles[index].costs[choice];
    }
    std::cout << std::setprecision(12)
              << "SAMPLE_SELECTOR size=" << portfolio.size()
              << " test_groups=" << test.size()
              << " current=" << current << " selected=" << selected
              << " reduction_vs_current=" << 1.0 - selected / current << '\n';
}

static void report_right_selector(
    const std::vector<GroupProfile>& profiles,
    const std::vector<size_t>& test, const std::vector<size_t>& portfolio,
    size_t current_candidate) {
    double current = 0, selected = 0;
    for (size_t index : test) {
        current += profiles[index].costs[current_candidate];
        size_t choice = portfolio.front();
        for (size_t candidate : portfolio)
            if (profiles[index].uniform_right_scores[candidate] <
                profiles[index].uniform_right_scores[choice]) choice = candidate;
        selected += profiles[index].costs[choice];
    }
    std::cout << std::setprecision(12)
              << "RIGHT_SELECTOR size=" << portfolio.size()
              << " test_groups=" << test.size()
              << " current=" << current << " selected=" << selected
              << " reduction_vs_current=" << 1.0 - selected / current << '\n';
}

static double portfolio_cost(const std::vector<GroupProfile>& profiles,
                             const std::vector<size_t>& indices,
                             const std::vector<size_t>& portfolio) {
    double total = 0;
    for (size_t index : indices) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t candidate : portfolio)
            best = std::min(best, profiles[index].costs[candidate]);
        total += best;
    }
    return total;
}

static std::vector<size_t> greedy_portfolio(
    const std::vector<GroupProfile>& profiles,
    const std::vector<size_t>& training, size_t wanted) {
    const size_t candidate_count = profiles.front().costs.size();
    std::vector<double> best(training.size(),
                             std::numeric_limits<double>::infinity());
    std::vector<bool> used(candidate_count, false);
    std::vector<size_t> result;
    while (result.size() < wanted) {
        size_t winner = 0;
        double winner_cost = std::numeric_limits<double>::infinity();
        for (size_t candidate = 0; candidate < candidate_count; candidate++) {
            if (used[candidate]) continue;
            double total = 0;
            for (size_t i = 0; i < training.size(); i++)
                total += std::min(best[i], profiles[training[i]].costs[candidate]);
            if (total < winner_cost) {
                winner_cost = total;
                winner = candidate;
            }
        }
        used[winner] = true;
        result.push_back(winner);
        for (size_t i = 0; i < training.size(); i++)
            best[i] = std::min(best[i], profiles[training[i]].costs[winner]);
    }
    return result;
}

static void report(const char* split,
                   const std::vector<GroupProfile>& profiles,
                   const std::vector<size_t>& indices,
                   const std::vector<size_t>& portfolio,
                   size_t current_candidate) {
    double direct = 0, current = 0, oracle = 0;
    for (size_t index : indices) {
        direct += profiles[index].direct;
        current += profiles[index].costs[current_candidate];
        oracle += *std::min_element(profiles[index].costs.begin(),
                                    profiles[index].costs.end());
    }
    const double adaptive = portfolio_cost(profiles, indices, portfolio);
    constexpr double measured_join = 668.680089;
    constexpr double measured_total = 769.346226;
    constexpr double measured_left_layout = 0.662854;
    const double projected = measured_total - measured_join +
        measured_join * adaptive / current +
        measured_left_layout * double(portfolio.size() - 1);
    std::cout << std::setprecision(12)
              << "PORTFOLIO split=" << split
              << " size=" << portfolio.size()
              << " groups=" << indices.size()
              << " direct=" << direct
              << " current=" << current
              << " adaptive=" << adaptive
              << " oracle=" << oracle
              << " current_retained=" << current / direct
              << " adaptive_retained=" << adaptive / direct
              << " reduction_vs_current=" << 1.0 - adaptive / current
              << " oracle_reduction_vs_current=" << 1.0 - oracle / current
              << " projected_seconds=" << projected
              << " projected_speedup=" << measured_total / projected << '\n';
}

static void report_lookup_selector(
    const std::vector<GroupProfile>& profiles,
    const std::vector<size_t>& training, const std::vector<size_t>& test,
    const std::vector<size_t>& portfolio, size_t current_candidate) {
    const size_t candidates = portfolio.size();
    std::unordered_map<uint32_t, std::vector<double>> learned;
    learned.reserve(training.size() * 2);
    std::vector<double> global(candidates, 0);
    for (size_t index : training) {
        auto& costs = learned[profiles[index].right];
        if (costs.empty()) costs.assign(candidates, 0);
        for (size_t candidate = 0; candidate < candidates; candidate++) {
            costs[candidate] += profiles[index].costs[portfolio[candidate]];
            global[candidate] += profiles[index].costs[portfolio[candidate]];
        }
    }
    size_t global_choice = size_t(std::min_element(global.begin(), global.end()) -
                                  global.begin());
    double current = 0, selected = 0, oracle = 0;
    size_t hits = 0;
    for (size_t index : test) {
        current += profiles[index].costs[current_candidate];
        auto found = learned.find(profiles[index].right);
        size_t choice = global_choice;
        if (found != learned.end()) {
            hits++;
            choice = size_t(std::min_element(found->second.begin(),
                                             found->second.end()) -
                            found->second.begin());
        }
        selected += profiles[index].costs[portfolio[choice]];
        oracle += *std::min_element(profiles[index].costs.begin(),
                                    profiles[index].costs.end());
    }
    std::cout << std::setprecision(12)
              << "LOOKUP_SELECTOR size=" << portfolio.size()
              << " test_groups=" << test.size()
              << " hits=" << hits << " hit_fraction=" << double(hits) / test.size()
              << " current=" << current << " selected=" << selected
              << " oracle=" << oracle
              << " reduction_vs_current=" << 1.0 - selected / current
              << " oracle_reduction_vs_current=" << 1.0 - oracle / current
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: " << argv[0]
                      << " ORBITS [ORBITS ...] [--groups N] [--samples N]"
                         " [--selector-samples N] [--portfolio N]\n";
            return 2;
        }
        size_t wanted_groups = 48;
        uint64_t samples = 2048;
        uint64_t selector_samples = 32;
        size_t maximum_portfolio = 16;
        std::vector<std::string> paths;
        for (int argument = 1; argument < argc; argument++) {
            std::string value = argv[argument];
            auto take = [&](auto& output) {
                if (++argument >= argc)
                    throw std::runtime_error("missing option value");
                std::istringstream input(argv[argument]);
                input >> output;
                if (!input || !input.eof())
                    throw std::runtime_error("invalid numeric option");
            };
            if (value == "--groups") take(wanted_groups);
            else if (value == "--samples") take(samples);
            else if (value == "--selector-samples") take(selector_samples);
            else if (value == "--portfolio") take(maximum_portfolio);
            else if (value.rfind("--", 0) == 0)
                throw std::runtime_error("unknown option " + value);
            else paths.push_back(value);
        }
        if (paths.empty() || !wanted_groups || !samples || !selector_samples || !maximum_portfolio ||
            maximum_portfolio > 16)
            throw std::runtime_error("invalid oracle arguments");

        const double start = seconds_now();
        initialise_tables();
        std::vector<std::vector<size_t>> by_file;
        std::vector<PortfolioGroup> groups =
            read_portfolio_groups(paths, by_file);
        if (paths.size() > 1) {
            std::unordered_set<uint32_t> training_rights;
            size_t reserve = 0;
            for (size_t file = 0; file + 1 < by_file.size(); file++)
                reserve += by_file[file].size();
            training_rights.reserve(reserve);
            for (size_t file = 0; file + 1 < by_file.size(); file++)
                for (size_t index : by_file[file])
                    training_rights.insert(groups[index].right);
            size_t hits = 0;
            for (size_t index : by_file.back())
                hits += training_rights.count(groups[index].right);
            std::cerr << "FULL_LOOKUP_CENSUS training_unique="
                      << training_rights.size()
                      << " test_groups=" << by_file.back().size()
                      << " hits=" << hits
                      << " hit_fraction="
                      << double(hits) / by_file.back().size() << '\n';
        }
        std::vector<SelectedGroup> selected;
        for (const auto& file_groups : by_file) {
            std::vector<SelectedGroup> part =
                select_groups(groups, file_groups, wanted_groups);
            selected.insert(selected.end(), part.begin(), part.end());
        }

        std::vector<uint32_t> prefixes;
        for (const SelectedGroup& item : selected) {
            const PortfolioGroup& group = groups[item.index];
            prefixes.push_back(group.right);
            prefixes.insert(prefixes.end(), group.left_reservoir.begin(),
                            group.left_reservoir.end());
        }
        std::sort(prefixes.begin(), prefixes.end());
        prefixes.erase(std::unique(prefixes.begin(), prefixes.end()),
                       prefixes.end());
        std::vector<DistributionPair> distributions(prefixes.size());
        const double build_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long index = 0; index < (long long)prefixes.size(); index++)
            distributions[size_t(index)] = build_pair(prefixes[size_t(index)]);
        std::unordered_map<uint32_t, size_t> prefix_index;
        prefix_index.reserve(prefixes.size() * 2);
        for (size_t index = 0; index < prefixes.size(); index++)
            prefix_index.emplace(prefixes[index], index);
        const double build_seconds = seconds_now() - build_start;

        const std::vector<uint32_t> candidates = enumerate_candidates();
        uint32_t current_mask = 0;
        for (unsigned pair : {0U, 1U, 7U, 2U, 8U, 13U, 18U})
            current_mask |= uint32_t(1) << pair;
        auto found = std::lower_bound(candidates.begin(), candidates.end(),
                                      current_mask);
        if (found == candidates.end() || *found != current_mask)
            throw std::logic_error("production configuration is absent");
        const size_t current_candidate = size_t(found - candidates.begin());
        std::cerr << "ORACLE_SETUP files=" << paths.size()
                  << " selected_groups=" << selected.size()
                  << " unique_prefixes=" << prefixes.size()
                  << " candidates=" << candidates.size()
                  << " samples_per_component=" << samples
                  << " current=" << candidate_string(current_mask)
                  << " build_seconds=" << build_seconds << '\n';

        std::vector<GroupProfile> profiles(selected.size());
        const double profile_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long ordinal = 0; ordinal < (long long)selected.size(); ordinal++) {
            const SelectedGroup& item = selected[size_t(ordinal)];
            profiles[size_t(ordinal)] = profile_group(
                groups[item.index], item.sampling_weight, samples,
                selector_samples, candidates, prefix_index, distributions);
        }
        const double profile_seconds = seconds_now() - profile_start;

        std::vector<size_t> train, test, all(profiles.size());
        std::iota(all.begin(), all.end(), 0);
        for (size_t index = 0; index < profiles.size(); index++) {
            bool training = paths.size() > 1
                ? profiles[index].file + 1 < paths.size()
                : (mix64(profiles[index].right) & 1) == 0;
            (training ? train : test).push_back(index);
        }
        if (train.empty() || test.empty())
            throw std::runtime_error("train/test split is empty");
        for (size_t size = 1; size <= maximum_portfolio; size *= 2) {
            std::vector<size_t> portfolio =
                greedy_portfolio(profiles, train, size);
            std::cout << "PORTFOLIO_CONFIG size=" << size;
            for (size_t candidate : portfolio)
                std::cout << " config=" << candidate_string(candidates[candidate]);
            std::cout << '\n';
            report("train", profiles, train, portfolio, current_candidate);
            report("test", profiles, test, portfolio, current_candidate);
            report("all", profiles, all, portfolio, current_candidate);
            report_lookup_selector(profiles, train, test, portfolio,
                                   current_candidate);
            report_right_selector(profiles, test, portfolio,
                                  current_candidate);
            report_sample_selector(profiles, test, portfolio,
                                   current_candidate);
        }
        std::cout << "ORACLE_DONE files=" << paths.size()
                  << " selected_groups=" << selected.size()
                  << " train=" << train.size() << " test=" << test.size()
                  << " prefixes=" << prefixes.size()
                  << " build_seconds=" << build_seconds
                  << " profile_seconds=" << profile_seconds
                  << " total_seconds=" << seconds_now() - start << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
