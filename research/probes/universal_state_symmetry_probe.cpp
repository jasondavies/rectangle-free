#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using U128 = unsigned __int128;

struct Options {
    unsigned rows = 0;
    unsigned columns = 0;
    uint64_t max_states = 5'000'000;
    uint64_t max_cache = 10'000'000;
    uint64_t max_transitions = 5'000'000'000ULL;
    uint64_t alignment_samples = 0;
    bool stabilizer_census = false;
    bool direct_cube = false;
    bool self_test = false;
};

struct ResourceLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double seconds_since(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

long peak_rss_kib() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

std::string decimal(U128 value) {
    if (!value) return "0";
    std::string result;
    while (value) {
        result.push_back(char('0' + value % 10));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

uint64_t parse_u64(const char *value, const char *name) {
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!value[0] || !end || *end) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

Options parse_options(int argc, char **argv) {
    Options options;
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        options.self_test = true;
        return options;
    }
    if (argc < 3) {
        throw std::runtime_error(
            "usage: universal_state_symmetry_probe ROWS COLUMNS "
            "[--max-states N] [--max-cache N] [--max-transitions N] "
            "[--stabilizer-census] [--alignment-samples N] [--direct-cube]");
    }
    options.rows = parse_u64(argv[1], "row count");
    options.columns = parse_u64(argv[2], "column count");
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--stabilizer-census") {
            options.stabilizer_census = true;
            continue;
        }
        if (argument == "--direct-cube") {
            options.direct_cube = true;
            continue;
        }
        if (index + 1 == argc) {
            throw std::runtime_error("missing value after " + argument);
        }
        const uint64_t value = parse_u64(argv[++index], argument.c_str());
        if (argument == "--max-states") options.max_states = value;
        else if (argument == "--max-cache") options.max_cache = value;
        else if (argument == "--max-transitions") {
            options.max_transitions = value;
        } else if (argument == "--alignment-samples") {
            options.alignment_samples = value;
        } else {
            throw std::runtime_error("unknown option " + argument);
        }
    }
    if (options.rows < 2 || options.rows > 6) {
        throw std::runtime_error("rows must be in 2..6");
    }
    if (options.columns > 9) {
        throw std::runtime_error("columns must be at most 9");
    }
    if (!options.max_states || !options.max_transitions) {
        throw std::runtime_error("state and transition caps must be positive");
    }
    return options;
}

std::array<uint16_t, 4> unpack(const uint64_t state) {
    return {
        uint16_t(state), uint16_t(state >> 16),
        uint16_t(state >> 32), uint16_t(state >> 48),
    };
}

uint64_t pack(std::array<uint16_t, 4> planes) {
    const auto compare_swap = [&planes](const unsigned first,
                                         const unsigned second) {
        if (planes[second] < planes[first]) {
            std::swap(planes[first], planes[second]);
        }
    };
    compare_swap(0, 1);
    compare_swap(2, 3);
    compare_swap(0, 2);
    compare_swap(1, 3);
    compare_swap(1, 2);
    return uint64_t(planes[0])
        | uint64_t(planes[1]) << 16
        | uint64_t(planes[2]) << 32
        | uint64_t(planes[3]) << 48;
}

class Canonicalizer {
public:
    Canonicalizer(const unsigned rows, const uint64_t cache_limit)
        : rows_(rows), pair_count_(rows * (rows - 1) / 2),
          domain_(uint32_t(1) << pair_count_), cache_limit_(cache_limit) {
        std::array<std::array<int, 6>, 6> pair_index{};
        for (auto &row : pair_index) row.fill(-1);
        std::vector<std::pair<unsigned, unsigned>> pairs;
        for (unsigned first = 0; first < rows_; ++first) {
            for (unsigned second = first + 1; second < rows_; ++second) {
                pair_index[first][second] = pair_index[second][first]
                    = int(pairs.size());
                pairs.emplace_back(first, second);
            }
        }

        std::vector<unsigned> permutation(rows_);
        std::iota(permutation.begin(), permutation.end(), 0);
        do {
            std::array<uint16_t, 15> bit_images{};
            for (unsigned index = 0; index < pairs.size(); ++index) {
                unsigned first = permutation[pairs[index].first];
                unsigned second = permutation[pairs[index].second];
                bit_images[index] = uint16_t(
                    1U << pair_index[first][second]);
            }
            const size_t base = images_.size();
            images_.resize(base + domain_);
            for (uint32_t mask = 1; mask < domain_; ++mask) {
                const uint32_t bit = mask & -mask;
                images_[base + mask] = uint16_t(
                    images_[base + mask - bit]
                    | bit_images[__builtin_ctz(bit)]);
            }
            ++permutation_count_;
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        if (cache_limit_) {
            cache_.reserve(size_t(std::min<uint64_t>(cache_limit_, 1'000'000)));
        }
    }

    uint64_t operator()(const uint64_t raw_state) {
        const auto found = cache_.find(raw_state);
        if (found != cache_.end()) {
            ++cache_hits_;
            return found->second;
        }
        ++cache_misses_;
        const auto raw = unpack(raw_state);
        uint64_t best = std::numeric_limits<uint64_t>::max();
        for (size_t permutation = 0; permutation < permutation_count_;
             ++permutation) {
            const size_t base = permutation * domain_;
            std::array<uint16_t, 4> transformed{
                images_[base + raw[0]], images_[base + raw[1]],
                images_[base + raw[2]], images_[base + raw[3]],
            };
            const uint64_t candidate = pack(transformed);
            if (candidate < best) best = candidate;
        }
        if (cache_.size() < cache_limit_) cache_.emplace(raw_state, best);
        return best;
    }

    size_t cache_size() const { return cache_.size(); }
    uint64_t cache_hits() const { return cache_hits_; }
    uint64_t cache_misses() const { return cache_misses_; }
    size_t permutation_count() const { return permutation_count_; }
    size_t image_bytes() const { return images_.size() * sizeof(uint16_t); }

    uint64_t stabilizer_size(const uint64_t canonical_state) const {
        const auto original = unpack(canonical_state);
        uint64_t colour_multiplicity = 1;
        for (unsigned begin = 0; begin < 4;) {
            unsigned end = begin + 1;
            while (end < 4 && original[end] == original[begin]) ++end;
            for (unsigned factor = 2; factor <= end - begin; ++factor) {
                colour_multiplicity *= factor;
            }
            begin = end;
        }
        uint64_t preserving_rows = 0;
        for (size_t permutation = 0; permutation < permutation_count_;
             ++permutation) {
            const size_t base = permutation * domain_;
            std::array<uint16_t, 4> transformed{
                images_[base + original[0]], images_[base + original[1]],
                images_[base + original[2]], images_[base + original[3]],
            };
            std::sort(transformed.begin(), transformed.end());
            if (transformed == original) ++preserving_rows;
        }
        return preserving_rows * colour_multiplicity;
    }

    uint64_t transform(const uint64_t state, const size_t row_permutation,
                       const std::array<unsigned, 4> &colour_permutation) const {
        const auto source = unpack(state);
        const size_t base = row_permutation * domain_;
        std::array<uint16_t, 4> row_transformed{
            images_[base + source[0]], images_[base + source[1]],
            images_[base + source[2]], images_[base + source[3]],
        };
        std::array<uint16_t, 4> result{};
        for (unsigned colour = 0; colour < 4; ++colour) {
            result[colour_permutation[colour]] = row_transformed[colour];
        }
        return uint64_t(result[0])
            | uint64_t(result[1]) << 16
            | uint64_t(result[2]) << 32
            | uint64_t(result[3]) << 48;
    }

private:
    unsigned rows_;
    unsigned pair_count_;
    uint32_t domain_;
    uint64_t cache_limit_;
    size_t permutation_count_ = 0;
    std::vector<uint16_t> images_;
    std::unordered_map<uint64_t, uint64_t> cache_;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
};

std::vector<std::pair<uint64_t, uint32_t>> column_supports(
    const unsigned rows) {
    std::array<std::array<int, 6>, 6> pair_index{};
    for (auto &row : pair_index) row.fill(-1);
    unsigned pair_count = 0;
    for (unsigned first = 0; first < rows; ++first) {
        for (unsigned second = first + 1; second < rows; ++second) {
            pair_index[first][second] = pair_index[second][first]
                = pair_count++;
        }
    }
    uint64_t colouring_count = 1;
    for (unsigned row = 0; row < rows; ++row) colouring_count *= 4;
    std::unordered_map<uint64_t, uint32_t> counts;
    counts.reserve(size_t(colouring_count));
    for (uint64_t encoded = 0; encoded < colouring_count; ++encoded) {
        uint64_t remaining = encoded;
        std::array<unsigned, 6> colours{};
        for (unsigned row = 0; row < rows; ++row) {
            colours[row] = remaining & 3;
            remaining >>= 2;
        }
        std::array<uint16_t, 4> planes{};
        for (unsigned first = 0; first < rows; ++first) {
            for (unsigned second = first + 1; second < rows; ++second) {
                if (colours[first] == colours[second]) {
                    planes[colours[first]] |= uint16_t(
                        1U << pair_index[first][second]);
                }
            }
        }
        const uint64_t raw = uint64_t(planes[0])
            | uint64_t(planes[1]) << 16
            | uint64_t(planes[2]) << 32
            | uint64_t(planes[3]) << 48;
        ++counts[raw];
    }
    return {counts.begin(), counts.end()};
}

bool disjoint(const uint64_t left, const uint64_t right) {
    return !(left & right);
}

uint64_t unite(const uint64_t left, const uint64_t right) {
    return left | right;
}

struct TransferRecord {
    unsigned columns;
    uint64_t states;
    U128 coefficient_sum;
};

uint64_t factorial(const unsigned value) {
    uint64_t result = 1;
    for (unsigned factor = 2; factor <= value; ++factor) result *= factor;
    return result;
}

void print_stabilizer_census(
    const Options &options,
    const Canonicalizer &canonicalize,
    const std::unordered_map<uint64_t, U128> &states) {
    std::unordered_map<uint64_t, uint64_t> histogram;
    uint64_t labelled_supports = 0;
    const uint64_t group_order = factorial(options.rows) * 24;
    for (const auto &[state, mass] : states) {
        (void)mass;
        const uint64_t stabilizer = canonicalize.stabilizer_size(state);
        if (!stabilizer || group_order % stabilizer) {
            throw std::runtime_error("invalid stabilizer size");
        }
        ++histogram[stabilizer];
        labelled_supports += group_order / stabilizer;
    }

    U128 double_coset_lower_bound = 0;
    for (const auto &[left_stabilizer, left_count] : histogram) {
        for (const auto &[right_stabilizer, right_count] : histogram) {
            const uint64_t product = left_stabilizer * right_stabilizer;
            const uint64_t lower = (group_order + product - 1) / product;
            double_coset_lower_bound += U128(left_count) * right_count * lower;
        }
    }

    std::cout << "{\"kind\":\"symmetry_stabilizer_census\",\"rows\":"
              << options.rows << ",\"columns\":" << options.columns
              << ",\"orbit_states\":" << states.size()
              << ",\"group_order\":" << group_order
              << ",\"labelled_supports\":" << labelled_supports
              << ",\"trivial_stabilizer_states\":" << histogram[1]
              << ",\"double_coset_task_lower_bound\":\""
              << decimal(double_coset_lower_bound)
              << "\",\"stabilizer_histogram\":{\"";
    std::vector<std::pair<uint64_t, uint64_t>> sorted(
        histogram.begin(), histogram.end());
    std::sort(sorted.begin(), sorted.end());
    for (size_t index = 0; index < sorted.size(); ++index) {
        if (index) std::cout << ",\"";
        std::cout << sorted[index].first << "\":" << sorted[index].second;
    }
    std::cout << "}}\n";

    if (!options.alignment_samples || states.empty()) return;
    std::vector<uint64_t> representatives;
    representatives.reserve(states.size());
    for (const auto &[state, mass] : states) {
        (void)mass;
        representatives.push_back(state);
    }
    std::mt19937_64 generator(0x9368E53C2F6AF274ULL);
    std::uniform_int_distribution<size_t> state_distribution(
        0, representatives.size() - 1);
    std::uniform_int_distribution<size_t> row_distribution(
        0, canonicalize.permutation_count() - 1);
    uint64_t compatible = 0;
    uint64_t total_union_tokens = 0;
    for (uint64_t sample = 0; sample < options.alignment_samples; ++sample) {
        const uint64_t left = representatives[state_distribution(generator)];
        const uint64_t right = representatives[state_distribution(generator)];
        std::array<unsigned, 4> colours{0, 1, 2, 3};
        std::shuffle(colours.begin(), colours.end(), generator);
        const uint64_t aligned = canonicalize.transform(
            right, row_distribution(generator), colours);
        if (disjoint(left, aligned)) {
            ++compatible;
            total_union_tokens += __builtin_popcountll(left | aligned);
        }
    }
    std::cout << "{\"kind\":\"symmetry_alignment_sample\",\"rows\":"
              << options.rows << ",\"columns\":" << options.columns
              << ",\"samples\":" << options.alignment_samples
              << ",\"compatible\":" << compatible
              << ",\"compatible_fraction\":"
              << double(compatible) / options.alignment_samples
              << ",\"mean_compatible_union_tokens\":"
              << (compatible ? double(total_union_tokens) / compatible : 0.0)
              << "}\n";
}

uint32_t compact_state(const uint64_t packed_state, const unsigned pair_count) {
    const auto planes = unpack(packed_state);
    uint32_t result = 0;
    for (unsigned colour = 0; colour < 4; ++colour) {
        result |= uint32_t(planes[colour]) << (colour * pair_count);
    }
    return result;
}

void direct_cube_contraction(
    const Options &options,
    const Canonicalizer &canonicalize,
    const std::unordered_map<uint64_t, U128> &states) {
    const unsigned pair_count = options.rows * (options.rows - 1) / 2;
    const unsigned token_count = 4 * pair_count;
    if (token_count > 24) {
        throw std::runtime_error(
            "direct dense subset oracle is restricted to at most 24 tokens");
    }
    const auto started = std::chrono::steady_clock::now();
    const uint64_t group_order = factorial(options.rows) * 24;
    std::vector<std::array<unsigned, 4>> colour_permutations;
    std::array<unsigned, 4> colours{0, 1, 2, 3};
    do {
        colour_permutations.push_back(colours);
    } while (std::next_permutation(colours.begin(), colours.end()));

    std::unordered_map<uint32_t, uint64_t> labelled;
    uint64_t expanded_states = 0;
    for (const auto &[representative, orbit_mass] : states) {
        const uint64_t stabilizer = canonicalize.stabilizer_size(representative);
        const uint64_t orbit_size = group_order / stabilizer;
        if (orbit_mass % orbit_size) {
            throw std::runtime_error("orbit mass is not divisible by orbit size");
        }
        const U128 wide_weight = orbit_mass / orbit_size;
        if (wide_weight > std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("per-state weight exceeds uint64_t");
        }
        const uint64_t weight = uint64_t(wide_weight);
        std::unordered_set<uint32_t> orbit;
        orbit.reserve(orbit_size);
        for (size_t row_permutation = 0;
             row_permutation < canonicalize.permutation_count();
             ++row_permutation) {
            for (const auto &colour_permutation : colour_permutations) {
                orbit.insert(compact_state(canonicalize.transform(
                    representative, row_permutation, colour_permutation),
                    pair_count));
            }
        }
        if (orbit.size() != orbit_size) {
            throw std::runtime_error("expanded orbit has incorrect size");
        }
        expanded_states += orbit.size();
        for (const uint32_t state : orbit) {
            const auto [position, inserted] = labelled.emplace(state, weight);
            if (!inserted && position->second != weight) {
                throw std::runtime_error("inconsistent expanded state weight");
            }
        }
    }
    if (labelled.size() != expanded_states) {
        throw std::runtime_error("distinct canonical orbits overlap");
    }

    const uint32_t domain = uint32_t(1) << token_count;
    std::vector<uint64_t> subset_sum(domain);
    for (const auto &[state, weight] : labelled) subset_sum[state] = weight;
    for (unsigned bit = 0; bit < token_count; ++bit) {
        const uint32_t flag = uint32_t(1) << bit;
        for (uint32_t mask = 0; mask < domain; ++mask) {
            if (mask & flag) subset_sum[mask] += subset_sum[mask ^ flag];
        }
    }

    const uint32_t complete = domain - 1;
    U128 answer = 0;
    uint64_t pair_tests = 0;
    uint64_t compatible_pairs = 0;
    for (const auto &[representative, orbit_mass] : states) {
        const uint64_t stabilizer = canonicalize.stabilizer_size(representative);
        const uint64_t orbit_size = group_order / stabilizer;
        const uint64_t left_weight = uint64_t(orbit_mass / orbit_size);
        const uint32_t left = compact_state(representative, pair_count);
        for (const auto &[right, right_weight] : labelled) {
            ++pair_tests;
            if (left & right) continue;
            ++compatible_pairs;
            const uint32_t allowed = complete ^ (left | right);
            answer += U128(orbit_size) * left_weight * right_weight
                * subset_sum[allowed];
        }
    }
    std::cout << "{\"kind\":\"symmetry_direct_cube\",\"rows\":"
              << options.rows << ",\"block_columns\":" << options.columns
              << ",\"orbit_states\":" << states.size()
              << ",\"labelled_states\":" << labelled.size()
              << ",\"pair_tests\":" << pair_tests
              << ",\"compatible_pairs\":" << compatible_pairs
              << ",\"subset_domain\":" << domain
              << ",\"answer\":\"" << decimal(answer)
              << "\",\"seconds\":" << seconds_since(started)
              << ",\"peak_rss_kib\":" << peak_rss_kib() << "}\n";
    if (options.rows == 3 && options.columns == 3
            && decimal(answer) != "4287132405909504") {
        throw std::runtime_error("3x9 direct-cube fixture mismatch");
    }
    if (options.rows == 4 && options.columns == 3
            && decimal(answer) != "257910839431786879488") {
        throw std::runtime_error("4x9 direct-cube fixture mismatch");
    }
}

std::vector<TransferRecord> run_transfer(const Options &options) {
    const auto overall_start = std::chrono::steady_clock::now();
    Canonicalizer canonicalize(options.rows, options.max_cache);
    const auto supports = column_supports(options.rows);
    std::unordered_map<uint64_t, U128> states;
    states.emplace(0, 1);
    std::vector<TransferRecord> records{{0, 1, 1}};
    uint64_t total_transitions = 0;

    std::cout << "{\"kind\":\"symmetry_compiled_header\",\"rows\":"
              << options.rows << ",\"target_columns\":" << options.columns
              << ",\"column_supports\":" << supports.size()
              << ",\"row_permutations\":" << canonicalize.permutation_count()
              << ",\"row_image_bytes\":" << canonicalize.image_bytes()
              << "}\n" << std::flush;

    for (unsigned column_count = 1; column_count <= options.columns;
         ++column_count) {
        const auto started = std::chrono::steady_clock::now();
        std::unordered_map<uint64_t, U128> following;
        following.reserve(size_t(std::min<uint64_t>(
            options.max_states, std::max<uint64_t>(1024, states.size() * 16))));
        uint64_t processed_states = 0;
        uint64_t step_transitions = 0;
        const uint64_t progress_step = std::max<uint64_t>(
            1000, states.size() / 20);
        for (const auto &[state, orbit_mass] : states) {
            std::unordered_map<uint64_t, uint32_t> target_counts;
            target_counts.reserve(supports.size());
            for (const auto &[support, multiplicity] : supports) {
                ++step_transitions;
                if (total_transitions + step_transitions
                        > options.max_transitions) {
                    throw ResourceLimit("transition cap exceeded while building P^"
                                        + std::to_string(column_count));
                }
                if (disjoint(state, support)) {
                    const uint64_t target = canonicalize(unite(state, support));
                    target_counts[target] += multiplicity;
                }
            }
            for (const auto &[target, count] : target_counts) {
                const auto [position, inserted] = following.try_emplace(target, 0);
                position->second += orbit_mass * count;
                if (inserted && following.size() > options.max_states) {
                    throw ResourceLimit("state cap exceeded while building P^"
                                        + std::to_string(column_count));
                }
            }
            ++processed_states;
            if (!(processed_states % progress_step)) {
                std::cerr << "P^" << column_count << ": " << processed_states
                          << '/' << states.size() << " sources, "
                          << following.size() << " target states, "
                          << seconds_since(started) << "s\n";
            }
        }
        total_transitions += step_transitions;
        states = std::move(following);
        U128 coefficient_sum = 0;
        for (const auto &[state, mass] : states) {
            (void)state;
            coefficient_sum += mass;
        }
        records.push_back({column_count, states.size(), coefficient_sum});
        std::cout << "{\"kind\":\"symmetry_compiled_step\",\"rows\":"
                  << options.rows << ",\"columns\":" << column_count
                  << ",\"orbit_states\":" << states.size()
                  << ",\"coefficient_sum\":\"" << decimal(coefficient_sum)
                  << "\",\"step_transitions\":" << step_transitions
                  << ",\"total_transitions\":" << total_transitions
                  << ",\"canonical_cache_entries\":"
                  << canonicalize.cache_size()
                  << ",\"canonical_cache_hits\":"
                  << canonicalize.cache_hits()
                  << ",\"canonical_cache_misses\":"
                  << canonicalize.cache_misses()
                  << ",\"seconds\":" << seconds_since(started)
                  << ",\"total_seconds\":" << seconds_since(overall_start)
                  << ",\"peak_rss_kib\":" << peak_rss_kib()
                  << "}\n" << std::flush;
    }
    if (options.stabilizer_census) {
        print_stabilizer_census(options, canonicalize, states);
    }
    if (options.direct_cube) {
        direct_cube_contraction(options, canonicalize, states);
    }
    return records;
}

void self_test() {
    Options three;
    three.rows = 3;
    three.columns = 3;
    three.direct_cube = true;
    const auto first = run_transfer(three);
    if (first.back().states != 18
            || decimal(first.back().coefficient_sum) != "228984") {
        throw std::runtime_error("3x3 quotient self-test failed");
    }
    Options four;
    four.rows = 4;
    four.columns = 4;
    const auto second = run_transfer(four);
    if (second.back().states != 1182
            || decimal(second.back().coefficient_sum) != "2545607472") {
        throw std::runtime_error("4x4 quotient self-test failed");
    }
    std::cout << "{\"kind\":\"symmetry_compiled_self_test\","
                 "\"status\":\"ok\"}\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.self_test) self_test();
        else run_transfer(options);
        return 0;
    } catch (const ResourceLimit &error) {
        std::cout << "{\"kind\":\"symmetry_compiled_resource_limit\","
                     "\"reason\":\"" << error.what()
                  << "\",\"peak_rss_kib\":" << peak_rss_kib() << "}\n";
        return 3;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
