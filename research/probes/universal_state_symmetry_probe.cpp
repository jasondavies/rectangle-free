#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
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
            "[--max-states N] [--max-cache N] [--max-transitions N]");
    }
    options.rows = parse_u64(argv[1], "row count");
    options.columns = parse_u64(argv[2], "column count");
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 == argc) {
            throw std::runtime_error("missing value after " + argument);
        }
        const uint64_t value = parse_u64(argv[++index], argument.c_str());
        if (argument == "--max-states") options.max_states = value;
        else if (argument == "--max-cache") options.max_cache = value;
        else if (argument == "--max-transitions") {
            options.max_transitions = value;
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
    return records;
}

void self_test() {
    Options three;
    three.rows = 3;
    three.columns = 3;
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
