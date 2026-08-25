// Exact packed defect-orbit and residual-hafnian workload gate for
// T_4(6,30-slack), currently covering 6x29 through 6x27.

#include <nauty/nauty.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
#endif

#include "../../src/hafnian/six_by_twenty_nine_catalog.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using U128 = unsigned __int128;
using six_by_twenty_nine::COLOURS;
using six_by_twenty_nine::Geometry;
using six_by_twenty_nine::PAIRS;
using six_by_twenty_nine::TOKENS;
using six_by_twenty_nine::WeightedSupport;

struct Record {
    uint64_t key = 0;
    uint64_t coefficient = 0;
};

struct Sector {
    uint64_t configurations = 0;
    uint64_t raw_unions = 0;
    uint64_t symmetry_orbits = 0;
    uint64_t graph_orbits = 0;
    U128 coefficient = 0;
    U128 symmetry_sign_terms = 0;
};

struct Options {
    unsigned slack = 2;
    unsigned threads = 1;
    uint64_t maximum_configurations = 0;
    bool graph_isomorphism = false;
    bool raw_census = false;
    uint64_t matching_max_states = 0;
    uint64_t matching_max_roots = 0;
    double matching_max_seconds = 0;
};

uint64_t parse_u64(const char* text) {
    char* end = nullptr;
    uint64_t value = std::strtoull(text, &end, 10);
    if (!end || *end) throw std::runtime_error("invalid integer");
    return value;
}

double parse_double(const char* text) {
    char* end = nullptr;
    double value = std::strtod(text, &end);
    if (!end || *end) throw std::runtime_error("invalid real number");
    return value;
}

std::string decimal(U128 value) {
    if (!value) return "0";
    std::string result;
    while (value) {
        result.push_back(char('0' + unsigned(value % 10)));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

uint64_t packed_key(uint64_t occupied, unsigned count) {
    if (occupied >> 60 || count > 15)
        throw std::runtime_error("defect key exceeds packed representation");
    return occupied | (uint64_t(count) << 60);
}

uint64_t occupied_from_key(uint64_t key) {
    return key & ((UINT64_C(1) << 60) - 1);
}

unsigned count_from_key(uint64_t key) {
    return unsigned(key >> 60);
}

void enumerate_extensions(const std::vector<WeightedSupport>& defects,
                          size_t begin, uint64_t occupied, unsigned count,
                          unsigned excess, unsigned budget,
                          uint64_t coefficient, std::vector<Record>& output,
                          uint64_t maximum_configurations) {
    if (maximum_configurations && output.size() >= maximum_configurations)
        throw std::runtime_error("per-thread configuration cap exceeded");
    output.push_back(Record{packed_key(occupied, count), coefficient});
    for (size_t index = begin; index < defects.size(); index++) {
        const WeightedSupport& support = defects[index];
        if (excess + support.excess > budget ||
            (occupied & support.mask)) continue;
        enumerate_extensions(defects, index + 1,
                             occupied | support.mask, count + 1,
                             excess + support.excess, budget,
                             coefficient * support.weight, output,
                             maximum_configurations);
    }
}

void exact_sort(std::vector<Record>& records, unsigned threads) {
    auto less = [](const Record& a, const Record& b) {
        return a.key < b.key;
    };
#ifdef _OPENMP
    if (threads > 1 && records.size() >= 1000000) {
        omp_set_num_threads(int(threads));
        __gnu_parallel::sort(records.begin(), records.end(), less);
        return;
    }
#else
    (void)threads;
#endif
    std::sort(records.begin(), records.end(), less);
}

void reduce_records(std::vector<Record>& records) {
    size_t write = 0;
    for (size_t begin = 0; begin < records.size();) {
        size_t end = begin + 1;
        U128 coefficient = records[begin].coefficient;
        while (end < records.size() && records[end].key == records[begin].key) {
            coefficient += records[end].coefficient;
            end++;
        }
        if (coefficient > std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("defect coefficient exceeds uint64_t");
        records[write++] = Record{records[begin].key, uint64_t(coefficient)};
        begin = end;
    }
    records.resize(write);
    records.shrink_to_fit();
}

std::string canonical_graph_key(const Geometry& geometry, uint64_t occupied,
                                unsigned unmatched) {
    std::vector<unsigned> originals;
    for (unsigned token = 0; token < TOKENS; token++)
        if (!(occupied & (UINT64_C(1) << token))) originals.push_back(token);
    const unsigned original_count = unsigned(originals.size());
    const unsigned vertices = original_count + unmatched;
    if (vertices > 64 || vertices % 2)
        throw std::runtime_error("invalid augmented residual graph order");
    const int words = SETWORDSNEEDED(int(vertices));
    std::vector<graph> input(size_t(words) * vertices);
    std::vector<graph> output(size_t(words) * vertices);
    auto token_adjacent = [&](unsigned left, unsigned right) {
        unsigned left_colour = left / PAIRS;
        unsigned right_colour = right / PAIRS;
        auto [a, b] = geometry.pairs[left % PAIRS];
        auto [c, d] = geometry.pairs[right % PAIRS];
        return left_colour != right_colour && a != c && a != d &&
               b != c && b != d;
    };
    for (unsigned first = 0; first < original_count; first++) {
        for (unsigned second = first + 1; second < original_count; second++) {
            if (!token_adjacent(originals[first], originals[second])) continue;
            ADDELEMENT(GRAPHROW(input.data(), first, words), second);
            ADDELEMENT(GRAPHROW(input.data(), second, words), first);
        }
        for (unsigned dummy = original_count; dummy < vertices; dummy++) {
            ADDELEMENT(GRAPHROW(input.data(), first, words), dummy);
            ADDELEMENT(GRAPHROW(input.data(), dummy, words), first);
        }
    }
    std::vector<int> labels(vertices), partition(vertices), orbits(vertices);
    std::iota(labels.begin(), labels.end(), 0);
    std::fill(partition.begin(), partition.end(), 1);
    if (vertices) partition.back() = 0;
    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    statsblk stats{};
    densenauty(input.data(), labels.data(), partition.data(), orbits.data(),
               &options, &stats, words, int(vertices), output.data());
    if (stats.errstatus) throw std::runtime_error("nauty canonicalization failed");
    std::string result;
    result.resize(sizeof(uint16_t) + output.size() * sizeof(graph));
    uint16_t encoded_vertices = uint16_t(vertices);
    std::memcpy(result.data(), &encoded_vertices, sizeof(encoded_vertices));
    std::memcpy(result.data() + sizeof(encoded_vertices), output.data(),
                output.size() * sizeof(graph));
    return result;
}

std::pair<unsigned, unsigned> sector_of(uint64_t key) {
    unsigned count = count_from_key(key);
    unsigned occupied = unsigned(__builtin_popcountll(occupied_from_key(key)));
    return {occupied - 2 * count, count};
}

uint64_t factorial(unsigned value) {
    uint64_t result = 1;
    for (unsigned factor = 2; factor <= value; factor++) result *= factor;
    return result;
}

struct BigUInt {
    std::array<uint64_t, 6> limbs{};
    explicit BigUInt(uint64_t value = 0) { limbs[0] = value; }
    void multiply(uint64_t factor) {
        U128 carry = 0;
        for (uint64_t& limb : limbs) {
            U128 value = U128(limb) * factor + carry;
            limb = uint64_t(value);
            carry = value >> 64;
        }
        if (carry) throw std::overflow_error("BigUInt multiplication overflow");
    }
    void add(const BigUInt& other) {
        U128 carry = 0;
        for (size_t index = 0; index < limbs.size(); index++) {
            U128 value = U128(limbs[index]) + other.limbs[index] + carry;
            limbs[index] = uint64_t(value);
            carry = value >> 64;
        }
        if (carry) throw std::overflow_error("BigUInt addition overflow");
    }
    void shift_left(unsigned bits) {
        if (!bits) return;
        const unsigned words = bits / 64;
        const unsigned remainder = bits % 64;
        if (words >= limbs.size())
            throw std::overflow_error("BigUInt shift overflow");
        std::array<uint64_t, 6> shifted{};
        for (size_t source = 0; source < limbs.size(); source++) {
            const size_t destination = source + words;
            if (destination >= limbs.size()) {
                if (limbs[source])
                    throw std::overflow_error("BigUInt shift overflow");
                continue;
            }
            shifted[destination] |= limbs[source] << remainder;
            if (remainder && destination + 1 < limbs.size())
                shifted[destination + 1] |= limbs[source] >> (64 - remainder);
            else if (remainder && (limbs[source] >> (64 - remainder)))
                throw std::overflow_error("BigUInt shift overflow");
        }
        limbs = shifted;
    }
    unsigned bit_length() const {
        for (size_t index = limbs.size(); index-- > 0;)
            if (limbs[index])
                return unsigned(64 * index + 64 - __builtin_clzll(limbs[index]));
        return 0;
    }
    std::string decimal() const {
        BigUInt copy = *this;
        std::string reversed;
        while (copy.bit_length()) {
            U128 remainder = 0;
            for (size_t index = copy.limbs.size(); index-- > 0;) {
                U128 value = (remainder << 64) | copy.limbs[index];
                copy.limbs[index] = uint64_t(value / 10);
                remainder = value % 10;
            }
            reversed.push_back(char('0' + unsigned(remainder)));
        }
        if (reversed.empty()) return "0";
        return std::string(reversed.rbegin(), reversed.rend());
    }
};

uint64_t binomial(unsigned n, unsigned k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    uint64_t result = 1;
    for (unsigned i = 1; i <= k; i++) result = result * (n - k + i) / i;
    return result;
}

unsigned ceil_log2_factorial(unsigned degree) {
    uint64_t value = factorial(degree);
    if (value <= 1) return 0;
    const unsigned floor = 63U - unsigned(__builtin_clzll(value));
    return (value & (value - 1)) ? floor + 1 : floor;
}

struct FriedlandSummary {
    BigUInt colouring_bound;
    std::array<uint64_t, 5> queries_by_prime_count{};
    std::array<U128, 5> sign_terms_by_prime_count{};
    std::array<uint64_t, 67> queries_by_order{};
    std::array<U128, 67> adaptive_sign_terms_by_order{};
};

unsigned required_matching_primes(unsigned bound_power) {
    constexpr std::array<uint32_t, 4> primes{
        2147483647U, 2147483629U, 2147483587U, 2147483579U};
    if (bound_power >= 128)
        throw std::overflow_error("matching bound exceeds four-prime CRT");
    U128 modulus = 1;
    const U128 bound = U128(1) << bound_power;
    for (unsigned count = 1; count <= primes.size(); count++) {
        modulus *= primes[count - 1];
        if (modulus > bound) return count;
    }
    throw std::overflow_error("matching bound exceeds four-prime CRT");
}

FriedlandSummary friedland_summary(const Geometry& geometry,
                                   const std::vector<Record>& orbit,
                                   unsigned slack) {
    // lcm(2,4,...,36).  This makes the sum of the exponents
    // ceil(log2(d!))/(2d) exact as a rational number.
    constexpr uint64_t EXPONENT_DENOMINATOR = 24504480;
    std::array<uint64_t, TOKENS> neighbours{};
    for (unsigned left = 0; left < TOKENS; left++) {
        const unsigned left_colour = left / PAIRS;
        auto [a, b] = geometry.pairs[left % PAIRS];
        for (unsigned right = 0; right < TOKENS; right++) {
            const unsigned right_colour = right / PAIRS;
            auto [c, d] = geometry.pairs[right % PAIRS];
            if (left_colour != right_colour && a != c && a != d &&
                b != c && b != d)
                neighbours[left] |= UINT64_C(1) << right;
        }
    }

    const unsigned width = 30 - slack;
    FriedlandSummary summary;
    const uint64_t full = (UINT64_C(1) << TOKENS) - 1;
    for (const Record& record : orbit) {
        auto [excess, count] = sector_of(record.key);
        const unsigned unmatched = 2 * slack - excess;
        const unsigned matching_edges = 30 - slack - count;
        const uint64_t remaining = full & ~occupied_from_key(record.key);
        const unsigned vertices = unsigned(__builtin_popcountll(remaining));
        uint64_t exponent_numerator = 0;
        uint64_t scan = remaining;
        while (scan) {
            const unsigned vertex = unsigned(__builtin_ctzll(scan));
            scan &= scan - 1;
            const unsigned degree = unsigned(__builtin_popcountll(
                neighbours[vertex] & remaining));
            if (degree)
                exponent_numerator +=
                    uint64_t(ceil_log2_factorial(degree)) *
                    (EXPONENT_DENOMINATOR / (2 * degree));
        }
        const unsigned exponent = unsigned(
            (exponent_numerator + EXPONENT_DENOMINATOR - 1) /
            EXPONENT_DENOMINATOR);

        const uint64_t unmatched_choices = binomial(vertices, unmatched);
        const unsigned bound_power = exponent +
            (unmatched_choices <= 1 ? 0 :
             64U - unsigned(__builtin_clzll(unmatched_choices - 1)));
        const unsigned prime_count = required_matching_primes(bound_power);
        const unsigned augmented_vertices = vertices + unmatched;
        if (!augmented_vertices || (augmented_vertices & 1))
            throw std::runtime_error("invalid augmented matching order");
        const U128 sign_terms = U128(1) << (augmented_vertices / 2 - 1);
        summary.queries_by_prime_count[prime_count]++;
        summary.sign_terms_by_prime_count[prime_count] += sign_terms;
        summary.queries_by_order[augmented_vertices]++;
        summary.adaptive_sign_terms_by_order[augmented_vertices] +=
            U128(prime_count) * sign_terms;

        BigUInt query(record.coefficient);
        query.multiply(unmatched_choices);
        for (unsigned factor = 2; factor <= width; factor++)
            query.multiply(factor);
        query.shift_left(matching_edges + exponent);
        summary.colouring_bound.add(query);
    }
    return summary;
}

struct MatchingLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class SharedMatchingCounter {
  public:
    SharedMatchingCounter(const Geometry& geometry, const Options& options)
        : geometry_(geometry), options_(options), started_(Clock::now()) {
        for (unsigned left = 0; left < TOKENS; left++) {
            unsigned left_colour = left / PAIRS;
            auto [a, b] = geometry_.pairs[left % PAIRS];
            for (unsigned right = 0; right < TOKENS; right++) {
                unsigned right_colour = right / PAIRS;
                auto [c, d] = geometry_.pairs[right % PAIRS];
                if (left_colour != right_colour && a != c && a != d &&
                    b != c && b != d)
                    neighbours_[left] |= UINT64_C(1) << right;
            }
        }
        canonical_.reserve(size_t(std::min<uint64_t>(
            options_.matching_max_states ? 2 * options_.matching_max_states
                                         : UINT64_C(20000000),
            UINT64_C(40000000))));
        memo_.reserve(size_t(std::min<uint64_t>(
            options_.matching_max_states ? options_.matching_max_states
                                         : UINT64_C(10000000),
            UINT64_C(20000000))));
    }

    void run(std::vector<Record> roots, unsigned slack) {
        std::sort(roots.begin(), roots.end(), [](const Record& a,
                                                 const Record& b) {
            unsigned ap = unsigned(__builtin_popcountll(
                occupied_from_key(a.key)));
            unsigned bp = unsigned(__builtin_popcountll(
                occupied_from_key(b.key)));
            return ap != bp ? ap > bp : a.key < b.key;
        });
        uint64_t wanted = options_.matching_max_roots
            ? std::min<uint64_t>(options_.matching_max_roots, roots.size())
            : roots.size();
        uint32_t aggregate = 0;
        try {
            while (completed_roots_ < wanted) {
                const Record& root = roots[completed_roots_];
                auto [excess, count] = sector_of(root.key);
                unsigned unmatched = 2 * slack - excess;
                uint64_t remaining = ((UINT64_C(1) << TOKENS) - 1) &
                                     ~occupied_from_key(root.key);
                uint32_t value = solve(remaining, unmatched);
                aggregate = uint32_t((aggregate +
                    (U128(root.coefficient % PRIME) * value) % PRIME) % PRIME);
                completed_roots_++;
                if (completed_roots_ % 100 == 0 || completed_roots_ == wanted)
                    report("PROGRESS", aggregate);
            }
            report("COMPLETE", aggregate);
        } catch (const MatchingLimit& limit) {
            report("LIMIT", aggregate);
            std::printf("DEFECT28_MATCHING_LIMIT reason=%s\n", limit.what());
        }
    }

  private:
    static constexpr uint32_t PRIME = 2147483647U;
    const Geometry& geometry_;
    const Options& options_;
    Clock::time_point started_;
    std::array<uint64_t, TOKENS> neighbours_{};
    std::unordered_map<uint64_t, uint64_t> canonical_;
    std::unordered_map<uint64_t, uint32_t> memo_;
    uint64_t states_ = 0, branches_ = 0, memo_hits_ = 0;
    uint64_t canonical_hits_ = 0, completed_roots_ = 0;

    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - started_).count();
    }

    uint64_t canonical(uint64_t state) {
        auto inserted = canonical_.emplace(state, 0);
        if (inserted.second)
            inserted.first->second =
                six_by_twenty_nine::canonicalize(geometry_, state);
        else
            canonical_hits_++;
        return inserted.first->second;
    }

    uint32_t add(uint32_t a, uint32_t b) const {
        uint32_t result = a + b;
        if (result >= PRIME || result < a) result -= PRIME;
        return result;
    }

    uint32_t solve(uint64_t raw_state, unsigned unmatched) {
        unsigned vertices = unsigned(__builtin_popcountll(raw_state));
        if (vertices < unmatched || ((vertices - unmatched) & 1)) return 0;
        if (vertices == unmatched) return 1;
        uint64_t state = canonical(raw_state);
        uint64_t key = state | (uint64_t(unmatched) << 60);
        auto known = memo_.find(key);
        if (known != memo_.end()) {
            memo_hits_++;
            return known->second;
        }
        states_++;
        if (options_.matching_max_states &&
            states_ > options_.matching_max_states)
            throw MatchingLimit("state cap exceeded");
        if ((states_ & 4095) == 0 && options_.matching_max_seconds &&
            elapsed() > options_.matching_max_seconds)
            throw MatchingLimit("time cap exceeded");

        unsigned pivot = 0, best_degree = TOKENS + 1;
        uint64_t scan = state;
        while (scan) {
            unsigned vertex = unsigned(__builtin_ctzll(scan));
            scan &= scan - 1;
            unsigned degree = unsigned(__builtin_popcountll(
                neighbours_[vertex] & state));
            if (degree < best_degree) {
                pivot = vertex;
                best_degree = degree;
                if (!degree) break;
            }
        }
        uint64_t without_pivot = state & ~(UINT64_C(1) << pivot);
        uint32_t result = unmatched ? solve(without_pivot, unmatched - 1) : 0;
        uint64_t candidates = neighbours_[pivot] & without_pivot;
        while (candidates) {
            unsigned mate = unsigned(__builtin_ctzll(candidates));
            candidates &= candidates - 1;
            result = add(result, solve(
                without_pivot & ~(UINT64_C(1) << mate), unmatched));
            branches_++;
        }
        memo_.emplace(key, result);
        return result;
    }

    void report(const char* status, uint32_t aggregate) const {
        std::printf(
            "DEFECT28_MATCHING status=%s roots=%" PRIu64
            " states=%" PRIu64 " branches=%" PRIu64
            " memo=%zu memo_hits=%" PRIu64 " canonical=%zu"
            " canonical_hits=%" PRIu64 " aggregate_mod=%u elapsed=%.6f\n",
            status, completed_roots_, states_, branches_, memo_.size(),
            memo_hits_, canonical_.size(), canonical_hits_, aggregate,
            elapsed());
        std::fflush(stdout);
    }
};

int run_orbit_census(const Geometry& geometry,
                     const std::vector<WeightedSupport>& defects,
                     const Options& options, const Clock::time_point& started) {
    const unsigned budget = 2 * options.slack;
    using Coefficients = std::unordered_map<uint64_t, U128>;
    std::vector<std::vector<Coefficients>> layers(
        budget + 1, std::vector<Coefficients>(budget + 1));
    layers[0][0][0] = 1;
    uint64_t transitions = 0;

    auto elapsed = [&] {
        return std::chrono::duration<double>(Clock::now() - started).count();
    };
    for (unsigned count = 0; count < budget; count++) {
        std::vector<std::vector<Coefficients>> local(
            options.threads, std::vector<Coefficients>(budget + 1));
        std::vector<std::unordered_map<uint64_t, uint64_t>> canonical_cache(
            options.threads);
        for (unsigned excess = 0; excess < budget; excess++) {
            std::vector<std::pair<uint64_t, U128>> parents(
                layers[count][excess].begin(), layers[count][excess].end());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(+:transitions)
#endif
            for (long long index = 0; index < (long long)parents.size(); index++) {
                unsigned worker = 0;
#ifdef _OPENMP
                worker = unsigned(omp_get_thread_num());
#endif
                uint64_t occupied = parents[size_t(index)].first;
                U128 coefficient = parents[size_t(index)].second;
                for (const WeightedSupport& support : defects) {
                    unsigned child_excess = excess + support.excess;
                    if (child_excess > budget || (occupied & support.mask))
                        continue;
                    uint64_t raw_child = occupied | support.mask;
                    auto inserted = canonical_cache[worker].emplace(raw_child, 0);
                    if (inserted.second)
                        inserted.first->second =
                            six_by_twenty_nine::canonicalize(geometry, raw_child);
                    local[worker][child_excess][inserted.first->second] +=
                        coefficient * support.weight;
                    transitions++;
                }
            }
        }
        for (unsigned worker = 0; worker < options.threads; worker++) {
            for (unsigned excess = count + 1; excess <= budget; excess++) {
                Coefficients& destination = layers[count + 1][excess];
                destination.reserve(destination.size() +
                                    local[worker][excess].size());
                for (const auto& [occupied, coefficient] :
                     local[worker][excess])
                    destination[occupied] += coefficient;
            }
        }
        uint64_t states = 0;
        for (unsigned excess = 0; excess <= budget; excess++)
            states += layers[count + 1][excess].size();
        std::printf("DEFECT28_ORBIT_LEVEL defects=%u states=%" PRIu64
                    " transitions=%" PRIu64 " elapsed=%.6f\n",
                    count + 1, states, transitions, elapsed());
        std::fflush(stdout);
    }

    std::vector<Record> orbit;
    std::map<std::pair<unsigned, unsigned>, Sector> sectors;
    for (unsigned count = 0; count <= budget; count++) {
        uint64_t divisor = factorial(count);
        for (unsigned excess = 0; excess <= budget; excess++) {
            for (const auto& [occupied, ordered_coefficient] :
                 layers[count][excess]) {
                if (ordered_coefficient % divisor)
                    throw std::runtime_error(
                        "ordered defect coefficient is not divisible by d!");
                U128 coefficient = ordered_coefficient / divisor;
                if (coefficient > std::numeric_limits<uint64_t>::max())
                    throw std::overflow_error(
                        "defect orbit coefficient exceeds uint64_t");
                orbit.push_back(Record{packed_key(occupied, count),
                                       uint64_t(coefficient)});
                Sector& sector = sectors[{excess, count}];
                sector.symmetry_orbits++;
                sector.coefficient += coefficient;
                const unsigned glynn_exponent =
                    29 + options.slack - count - excess;
                sector.symmetry_sign_terms += U128(1) << glynn_exponent;
            }
        }
    }

    if (options.graph_isomorphism) {
        std::map<std::pair<unsigned, unsigned>,
                 std::unordered_map<std::string, U128>> graph_orbits;
        for (const Record& record : orbit) {
            auto sector_key = sector_of(record.key);
            unsigned unmatched = 2 * options.slack - sector_key.first;
            std::string key = canonical_graph_key(
                geometry, occupied_from_key(record.key), unmatched);
            graph_orbits[sector_key][std::move(key)] += record.coefficient;
        }
        for (auto& [key, values] : graph_orbits)
            sectors[key].graph_orbits = values.size();
    }

    uint64_t total_orbits = 0, total_graph_orbits = 0;
    U128 total_symmetry_terms = 0, total_graph_terms = 0;
    BigUInt colouring_bound;
    for (auto& [key, sector] : sectors) {
        unsigned excess = key.first, count = key.second;
        unsigned unmatched = 2 * options.slack - excess;
        unsigned vertices = 60 - (2 * count + excess) + unmatched;
        if (!sector.graph_orbits) sector.graph_orbits = sector.symmetry_orbits;
        const unsigned glynn_exponent =
            29 + options.slack - count - excess;
        U128 graph_terms =
            U128(sector.graph_orbits) << glynn_exponent;
        total_orbits += sector.symmetry_orbits;
        total_graph_orbits += sector.graph_orbits;
        total_symmetry_terms += sector.symmetry_sign_terms;
        total_graph_terms += graph_terms;
        unsigned original_vertices = 60 - 2 * count - excess;
        const unsigned width = 30 - options.slack;
        unsigned matching_edges = width - count;
        if (sector.coefficient > std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("sector coefficient exceeds uint64_t");
        BigUInt sector_bound(uint64_t(sector.coefficient));
        for (unsigned factor = matching_edges + 1; factor <= width; factor++)
            sector_bound.multiply(factor);
        for (unsigned factor = unmatched + 1; factor <= original_vertices;
             factor++)
            sector_bound.multiply(factor);
        colouring_bound.add(sector_bound);
        std::printf(
            "DEFECT28_SECTOR excess=%u defects=%u unmatched=%u vertices=%u "
            "symmetry_orbits=%" PRIu64 " graph_orbits=%" PRIu64
            " coefficient=%s symmetry_sign_terms=%s graph_sign_terms=%s\n",
            excess, count, unmatched, vertices, sector.symmetry_orbits,
            sector.graph_orbits, decimal(sector.coefficient).c_str(),
            decimal(sector.symmetry_sign_terms).c_str(),
            decimal(graph_terms).c_str());
    }
    unsigned bound_bits = colouring_bound.bit_length();
    std::cout << "DEFECT28_BOUND value=" << colouring_bound.decimal()
              << " bits=" << bound_bits
              << " required_31bit_primes=" << (bound_bits + 30) / 31
              << " exact=OK\n";
    const FriedlandSummary friedland =
        friedland_summary(geometry, orbit, options.slack);
    U128 adaptive_sign_terms = 0;
    for (unsigned primes = 1; primes < friedland.queries_by_prime_count.size();
         primes++) {
        adaptive_sign_terms +=
            U128(primes) * friedland.sign_terms_by_prime_count[primes];
        std::printf(
            "DEFECT28_ADAPTIVE primes=%u queries=%" PRIu64
            " sign_terms_per_prime=%s weighted_sign_terms=%s\n",
            primes, friedland.queries_by_prime_count[primes],
            decimal(friedland.sign_terms_by_prime_count[primes]).c_str(),
            decimal(U128(primes) *
                    friedland.sign_terms_by_prime_count[primes]).c_str());
    }
    std::printf("DEFECT28_ADAPTIVE_TOTAL sign_terms=%s exact=OK\n",
                decimal(adaptive_sign_terms).c_str());
    for (unsigned order = 0; order < friedland.queries_by_order.size(); order++) {
        if (!friedland.queries_by_order[order]) continue;
        std::printf(
            "DEFECT28_ADAPTIVE_ORDER vertices=%u queries=%" PRIu64
            " weighted_sign_terms=%s\n",
            order, friedland.queries_by_order[order],
            decimal(friedland.adaptive_sign_terms_by_order[order]).c_str());
    }
    const BigUInt& friedland_bound = friedland.colouring_bound;
    const unsigned friedland_bits = friedland_bound.bit_length();
    std::cout << "DEFECT28_FRIEDLAND_BOUND value="
              << friedland_bound.decimal() << " bits=" << friedland_bits
              << " required_31bit_primes=" << (friedland_bits + 30) / 31
              << " exact=OK\n";
    std::printf("DEFECT28_DONE mode=orbit slack=%u width=%u"
                " symmetry_orbits=%" PRIu64 " graph_orbits=%" PRIu64
                " symmetry_sign_terms_per_prime=%s"
                " graph_sign_terms_per_prime=%s transitions=%" PRIu64
                " elapsed=%.6f exact=OK\n",
                options.slack, 30 - options.slack, total_orbits,
                total_graph_orbits, decimal(total_symmetry_terms).c_str(),
                decimal(total_graph_terms).c_str(), transitions, elapsed());

    if (options.slack == 1 && total_orbits != 29)
        throw std::runtime_error("6x29 orbit-DP census mismatch");
    if (options.slack == 2 && total_orbits != 36398)
        throw std::runtime_error("6x28 orbit-DP census mismatch");
    if (options.slack == 3 && total_orbits != 45007139)
        throw std::runtime_error("6x27 orbit-DP census mismatch");
    if (options.slack == 2 &&
        adaptive_sign_terms != U128(UINT64_C(1063130234880)))
        throw std::runtime_error("6x28 adaptive workload mismatch");
    if (options.slack == 3 &&
        adaptive_sign_terms != U128(UINT64_C(134616715362304)))
        throw std::runtime_error("6x27 adaptive workload mismatch");
    if (options.matching_max_states || options.matching_max_seconds ||
        options.matching_max_roots)
        SharedMatchingCounter(geometry, options).run(orbit, options.slack);
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    Options options;
    for (int argument = 1; argument < argc; argument++) {
        std::string value = argv[argument];
        if (value == "--slack" && argument + 1 < argc)
            options.slack = unsigned(parse_u64(argv[++argument]));
        else if (value == "--threads" && argument + 1 < argc)
            options.threads = unsigned(parse_u64(argv[++argument]));
        else if (value == "--max-configurations" && argument + 1 < argc)
            options.maximum_configurations = parse_u64(argv[++argument]);
        else if (value == "--graph-isomorphism")
            options.graph_isomorphism = true;
        else if (value == "--raw")
            options.raw_census = true;
        else if (value == "--matching-max-states" && argument + 1 < argc)
            options.matching_max_states = parse_u64(argv[++argument]);
        else if (value == "--matching-max-roots" && argument + 1 < argc)
            options.matching_max_roots = parse_u64(argv[++argument]);
        else if (value == "--matching-max-seconds" && argument + 1 < argc)
            options.matching_max_seconds = parse_double(argv[++argument]);
        else
            throw std::runtime_error(
                "usage: six_by_twenty_eight_defect_census "
                "[--slack 1|2|3] [--threads N] [--max-configurations N] "
                "[--graph-isomorphism] [--raw] "
                "[--matching-max-states N] [--matching-max-roots N] "
                "[--matching-max-seconds S]");
    }
    if (!options.threads || options.slack < 1 || options.slack > 3)
        throw std::runtime_error(
            "slack must be one, two, or three and threads positive");
    if (options.raw_census && options.slack != 1)
        throw std::runtime_error(
            "raw census is retained only for the bounded 6x29 regression");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(int(options.threads));
#else
    if (options.threads != 1)
        throw std::runtime_error("binary lacks OpenMP support");
#endif

    const auto started = Clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(Clock::now() - started).count();
    };
    Geometry geometry;
    std::vector<WeightedSupport> supports =
        six_by_twenty_nine::weighted_supports(geometry);
    const unsigned budget = 2 * options.slack;
    std::vector<WeightedSupport> defects;
    for (const WeightedSupport& support : supports)
        if (support.excess && support.excess <= budget)
            defects.push_back(support);

    if (!options.raw_census)
        return run_orbit_census(geometry, defects, options, started);

    std::vector<std::vector<Record>> local(options.threads);
    for (auto& records : local) records.reserve(1 << 20);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (long long first = 0; first < (long long)defects.size(); first++) {
        unsigned worker = 0;
#ifdef _OPENMP
        worker = unsigned(omp_get_thread_num());
#endif
        const WeightedSupport& support = defects[size_t(first)];
        enumerate_extensions(defects, size_t(first) + 1, support.mask, 1,
                             support.excess, budget, support.weight,
                             local[worker], options.maximum_configurations);
    }
    uint64_t configurations = 1;
    for (const auto& records : local) configurations += records.size();
    std::vector<Record> raw;
    raw.resize(size_t(configurations));
    raw[0] = Record{packed_key(0, 0), 1};
    size_t offset = 1;
    for (auto& records : local) {
        std::memcpy(raw.data() + offset, records.data(),
                    records.size() * sizeof(Record));
        offset += records.size();
        records.clear();
        records.shrink_to_fit();
    }
    std::printf("DEFECT28_CONFIGURATIONS slack=%u budget=%u candidates=%zu "
                "configurations=%" PRIu64 " bytes=%zu elapsed=%.6f\n",
                options.slack, budget, defects.size(), configurations,
                raw.size() * sizeof(Record), elapsed());
    std::fflush(stdout);

    std::map<std::pair<unsigned, unsigned>, Sector> sectors;
    for (const Record& record : raw)
        sectors[sector_of(record.key)].configurations++;
    exact_sort(raw, options.threads);
    reduce_records(raw);
    const size_t raw_union_count = raw.size();
    for (const Record& record : raw)
        sectors[sector_of(record.key)].raw_unions++;
    std::printf("DEFECT28_RAW unions=%zu bytes=%zu elapsed=%.6f\n",
                raw.size(), raw.size() * sizeof(Record), elapsed());
    std::fflush(stdout);

    std::vector<Record> orbit(raw.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (long long index = 0; index < (long long)raw.size(); index++) {
        uint64_t occupied = occupied_from_key(raw[size_t(index)].key);
        unsigned count = count_from_key(raw[size_t(index)].key);
        orbit[size_t(index)] = Record{
            packed_key(six_by_twenty_nine::canonicalize(geometry, occupied),
                       count),
            raw[size_t(index)].coefficient};
    }
    raw.clear();
    raw.shrink_to_fit();
    exact_sort(orbit, options.threads);
    reduce_records(orbit);
    for (const Record& record : orbit) {
        auto sector_key = sector_of(record.key);
        Sector& sector = sectors[sector_key];
        sector.symmetry_orbits++;
        sector.coefficient += record.coefficient;
        unsigned excess = sector_key.first;
        unsigned count = sector_key.second;
        unsigned exponent = 29 + options.slack - count - excess;
        sector.symmetry_sign_terms += U128(1) << exponent;
    }
    std::printf("DEFECT28_SYMMETRY orbits=%zu bytes=%zu elapsed=%.6f\n",
                orbit.size(), orbit.size() * sizeof(Record), elapsed());
    std::fflush(stdout);

    if (options.graph_isomorphism) {
        std::map<std::pair<unsigned, unsigned>,
                 std::unordered_map<std::string, U128>> graph_orbits;
        for (const Record& record : orbit) {
            auto sector_key = sector_of(record.key);
            unsigned unmatched = 2 * options.slack - sector_key.first;
            std::string key = canonical_graph_key(
                geometry, occupied_from_key(record.key), unmatched);
            graph_orbits[sector_key][std::move(key)] += record.coefficient;
        }
        for (auto& [key, values] : graph_orbits)
            sectors[key].graph_orbits = values.size();
        std::printf("DEFECT28_GRAPH_ISOMORPHISM elapsed=%.6f exact=OK\n",
                    elapsed());
    }

    uint64_t total_orbits = 0, total_graph_orbits = 0;
    U128 total_symmetry_terms = 0, total_graph_terms = 0;
    for (auto& [key, sector] : sectors) {
        unsigned excess = key.first, count = key.second;
        unsigned unmatched = 2 * options.slack - excess;
        unsigned vertices = 60 - (2 * count + excess) + unmatched;
        if (!sector.graph_orbits) sector.graph_orbits = sector.symmetry_orbits;
        total_orbits += sector.symmetry_orbits;
        total_graph_orbits += sector.graph_orbits;
        const unsigned glynn_exponent =
            29 + options.slack - count - excess;
        U128 graph_terms =
            U128(sector.graph_orbits) << glynn_exponent;
        total_symmetry_terms += sector.symmetry_sign_terms;
        total_graph_terms += graph_terms;
        std::printf(
            "DEFECT28_SECTOR excess=%u defects=%u unmatched=%u vertices=%u "
            "configurations=%" PRIu64 " raw_unions=%" PRIu64
            " symmetry_orbits=%" PRIu64 " graph_orbits=%" PRIu64
            " coefficient=%s symmetry_sign_terms=%s graph_sign_terms=%s\n",
            excess, count, unmatched, vertices, sector.configurations,
            sector.raw_unions, sector.symmetry_orbits, sector.graph_orbits,
            decimal(sector.coefficient).c_str(),
            decimal(sector.symmetry_sign_terms).c_str(),
            decimal(graph_terms).c_str());
    }
    std::printf("DEFECT28_DONE slack=%u width=%u configurations=%" PRIu64
                " raw_unions=%zu symmetry_orbits=%" PRIu64
                " graph_orbits=%" PRIu64
                " symmetry_sign_terms_per_prime=%s"
                " graph_sign_terms_per_prime=%s "
                "elapsed=%.6f exact=OK\n",
                options.slack, 30 - options.slack, configurations,
                raw_union_count, total_orbits, total_graph_orbits,
                decimal(total_symmetry_terms).c_str(),
                decimal(total_graph_terms).c_str(), elapsed());

    if (options.slack == 1 &&
        (raw_union_count != 83071 || orbit.size() != 29))
        throw std::runtime_error("6x29 regression census mismatch");
    if (options.slack == 2 && total_orbits != 36398)
        throw std::runtime_error("6x28 canonical census mismatch");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 2;
}
