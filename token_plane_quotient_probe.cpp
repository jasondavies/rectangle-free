#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using U128 = unsigned __int128;

namespace {

struct Increment {
    uint64_t mask;
    uint32_t weight;
};

struct Entry {
    uint64_t mask;
    uint64_t weight;
};

struct OrbitEntry {
    uint64_t representative;
    uint64_t weight;
    uint8_t orbit_size;
};

struct GeometryResult {
    uint64_t entries = 0;
    uint64_t fixed_entries = 0;
    uint64_t missing_partners = 0;
    uint64_t weight_mismatches = 0;
    uint64_t prefix_mismatches = 0;
    uint64_t join_checks = 0;
    size_t maximum_support = 0;
    size_t maximum_weight_classes = 0;
    size_t maximum_orbit_classes = 0;
};

static std::string u128_string(U128 value) {
    char digits[40];
    size_t length = 0;
    do {
        digits[length++] = char('0' + value % 10);
        value /= 10;
    } while (value);
    std::string result;
    while (length) result.push_back(digits[--length]);
    return result;
}

static uint64_t low_bits(unsigned bits) {
    return bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
}

static uint64_t swap_planes(uint64_t mask, unsigned pairs) {
    uint64_t plane = low_bits(pairs);
    return ((mask & plane) << pairs) | ((mask >> pairs) & plane);
}

static uint64_t swap_split_planes(uint64_t value, unsigned plane_bits) {
    uint64_t plane = low_bits(plane_bits);
    return ((value & plane) << plane_bits) |
           ((value >> plane_bits) & plane);
}

class DistributionBuilder {
  public:
    DistributionBuilder(unsigned rows, unsigned columns)
        : rows_(rows), columns_(columns), pairs_(rows * (rows - 1) / 2),
          increments_(size_t(1) << rows) {
        if (rows < 2 || rows > 8 || columns < 1 || rows * columns > 63 ||
            2 * pairs_ > 63)
            throw std::invalid_argument("unsupported probe geometry");
        unsigned pair = 0;
        for (unsigned first = 0; first < rows_; first++)
            for (unsigned second = first + 1; second < rows_; second++)
                pair_index_[first][second] = pair++;
        initialise_increments();
    }

    unsigned pairs() const { return pairs_; }

    std::vector<Entry> build(uint64_t key) const {
        std::array<uint8_t, 8> patterns{};
        uint64_t pattern_mask = low_bits(columns_);
        for (int row = int(rows_) - 1; row >= 0; row--) {
            patterns[size_t(row)] = uint8_t(key & pattern_mask);
            key >>= columns_;
        }

        std::unordered_map<uint64_t, uint64_t> current, next;
        current.emplace(0, 1);
        for (unsigned column = 0; column < columns_; column++) {
            unsigned active = 0;
            for (unsigned row = 0; row < rows_; row++)
                if (patterns[row] & (1U << column)) active |= 1U << row;
            next.clear();
            next.reserve(std::max<size_t>(16, current.size() * 4));
            for (const auto& [mask, weight] : current) {
                for (const Increment& increment : increments_[active]) {
                    if (mask & increment.mask) continue;
                    uint64_t product = weight * increment.weight;
                    if (increment.weight && product / increment.weight != weight)
                        throw std::overflow_error("distribution weight overflow");
                    next[mask | increment.mask] += product;
                }
            }
            current.swap(next);
        }

        std::vector<Entry> result;
        result.reserve(current.size());
        for (const auto& [mask, weight] : current)
            result.push_back(Entry{mask, weight});
        std::sort(result.begin(), result.end(),
                  [](const Entry& lhs, const Entry& rhs) {
                      return lhs.mask < rhs.mask;
                  });
        return result;
    }

  private:
    void initialise_increments() {
        for (unsigned active = 0; active < (1U << rows_); active++) {
            std::vector<uint64_t> masks;
            for (unsigned assignment = active;;
                 assignment = (assignment - 1U) & active) {
                uint64_t mask = 0;
                for (unsigned first = 0; first < rows_; first++) {
                    for (unsigned second = first + 1; second < rows_; second++) {
                        if (!(active & (1U << first)) ||
                            !(active & (1U << second)))
                            continue;
                        unsigned first_colour = (assignment >> first) & 1U;
                        unsigned second_colour = (assignment >> second) & 1U;
                        if (first_colour == second_colour)
                            mask |= UINT64_C(1)
                                    << (first_colour * pairs_ +
                                        pair_index_[first][second]);
                    }
                }
                masks.push_back(mask);
                if (!assignment) break;
            }
            std::sort(masks.begin(), masks.end());
            for (size_t begin = 0; begin < masks.size();) {
                size_t end = begin + 1;
                while (end < masks.size() && masks[end] == masks[begin]) end++;
                increments_[active].push_back(
                    Increment{masks[begin], uint32_t(end - begin)});
                begin = end;
            }
        }
    }

    unsigned rows_, columns_, pairs_;
    unsigned pair_index_[8][8]{};
    std::vector<std::vector<Increment>> increments_;
};

static std::vector<OrbitEntry> quotient_distribution(
    const std::vector<Entry>& entries, unsigned pairs,
    GeometryResult& result) {
    std::unordered_map<uint64_t, uint64_t> weights;
    weights.reserve(entries.size() * 2);
    for (const Entry& entry : entries) weights.emplace(entry.mask, entry.weight);

    std::vector<OrbitEntry> quotient;
    quotient.reserve((entries.size() + 1) / 2);
    for (const Entry& entry : entries) {
        uint64_t swapped = swap_planes(entry.mask, pairs);
        auto partner = weights.find(swapped);
        if (partner == weights.end()) {
            result.missing_partners++;
            continue;
        }
        if (partner->second != entry.weight) result.weight_mismatches++;
        if (swapped < entry.mask) continue;
        uint8_t orbit_size = swapped == entry.mask ? 1 : 2;
        result.fixed_entries += orbit_size == 1;
        quotient.push_back(OrbitEntry{entry.mask, entry.weight, orbit_size});
    }
    return quotient;
}

static U128 direct_join(const std::vector<Entry>& lhs,
                        const std::vector<Entry>& rhs) {
    U128 sum = 0;
    for (const Entry& left : lhs)
        for (const Entry& right : rhs)
            if (!(left.mask & right.mask))
                sum += U128(left.weight) * right.weight;
    return sum;
}

static U128 quotient_join(const std::vector<OrbitEntry>& lhs,
                          const std::vector<OrbitEntry>& rhs,
                          unsigned pairs) {
    U128 sum = 0;
    for (const OrbitEntry& left : lhs) {
        for (const OrbitEntry& right : rhs) {
            U128 product = U128(left.orbit_size) * left.weight * right.weight;
            if (!(left.representative & right.representative)) sum += product;
            if (right.orbit_size == 2 &&
                !(left.representative &
                  swap_planes(right.representative, pairs)))
                sum += product;
        }
    }
    return sum;
}

static void reorient_quotient(std::vector<OrbitEntry>& entries,
                              unsigned pairs, uint64_t salt) {
    for (OrbitEntry& entry : entries) {
        if (entry.orbit_size == 2 &&
            ((entry.representative * UINT64_C(0x9e3779b97f4a7c15) ^ salt)
             >> 63))
            entry.representative =
                swap_planes(entry.representative, pairs);
    }
}

static void verify_prefix_equivariance(const std::vector<Entry>& entries,
                                       unsigned pairs,
                                       unsigned prefix_pairs,
                                       GeometryResult& result) {
    uint64_t pair_prefix = low_bits(prefix_pairs);
    for (const Entry& entry : entries) {
        uint64_t prefix = (entry.mask & pair_prefix) |
                          (((entry.mask >> pairs) & pair_prefix)
                           << prefix_pairs);
        uint64_t suffix = 0;
        unsigned suffix_bit = 0;
        for (unsigned colour = 0; colour < 2; colour++)
            for (unsigned pair = prefix_pairs; pair < pairs; pair++)
                suffix |= ((entry.mask >> (colour * pairs + pair)) & 1U)
                          << suffix_bit++;

        uint64_t swapped = swap_planes(entry.mask, pairs);
        uint64_t swapped_prefix = (swapped & pair_prefix) |
                                  (((swapped >> pairs) & pair_prefix)
                                   << prefix_pairs);
        uint64_t swapped_suffix = 0;
        suffix_bit = 0;
        for (unsigned colour = 0; colour < 2; colour++)
            for (unsigned pair = prefix_pairs; pair < pairs; pair++)
                swapped_suffix |=
                    ((swapped >> (colour * pairs + pair)) & 1U)
                    << suffix_bit++;
        if (swapped_prefix != swap_split_planes(prefix, prefix_pairs) ||
            swapped_suffix != swap_split_planes(
                                  suffix, pairs - prefix_pairs))
            result.prefix_mismatches++;
    }
}

static size_t weight_class_count(const std::vector<Entry>& entries) {
    std::unordered_set<uint64_t> classes;
    for (const Entry& entry : entries) classes.insert(entry.weight);
    return classes.size();
}

static size_t orbit_class_count(const std::vector<OrbitEntry>& entries) {
    std::unordered_set<uint64_t> classes;
    for (const OrbitEntry& entry : entries)
        classes.insert((entry.weight << 1) | (entry.orbit_size == 2));
    return classes.size();
}

static GeometryResult run_geometry(unsigned rows, unsigned columns,
                                   unsigned prefix_pairs, unsigned samples,
                                   unsigned join_samples, uint64_t seed) {
    DistributionBuilder builder(rows, columns);
    GeometryResult result;
    std::mt19937_64 random(seed);
    uint64_t key_mask = low_bits(rows * columns);
    std::vector<std::pair<std::vector<Entry>, std::vector<OrbitEntry>>>
        join_candidates;

    for (unsigned sample = 0; sample < samples; sample++) {
        uint64_t key = random() & key_mask;
        std::vector<Entry> distribution = builder.build(key);
        result.entries += distribution.size();
        result.maximum_support =
            std::max(result.maximum_support, distribution.size());
        verify_prefix_equivariance(distribution, builder.pairs(),
                                   prefix_pairs, result);
        std::vector<OrbitEntry> quotient = quotient_distribution(
            distribution, builder.pairs(), result);
        result.maximum_weight_classes = std::max(
            result.maximum_weight_classes, weight_class_count(distribution));
        result.maximum_orbit_classes = std::max(
            result.maximum_orbit_classes, orbit_class_count(quotient));
        if (distribution.size() <= 4096 &&
            join_candidates.size() < join_samples) {
            reorient_quotient(quotient, builder.pairs(),
                              seed + join_candidates.size());
            join_candidates.push_back(
                {std::move(distribution), std::move(quotient)});
        }
    }

    // Sparse genuine half-masks provide inexpensive target-geometry joins
    // when all random dense supports exceed the direct-oracle limit.
    for (unsigned attempt = 0;
         join_candidates.size() < join_samples && attempt < 10000; attempt++) {
        uint64_t key = random() & random() & key_mask;
        std::vector<Entry> distribution = builder.build(key);
        if (distribution.size() > 4096) continue;
        GeometryResult ignored;
        std::vector<OrbitEntry> quotient = quotient_distribution(
            distribution, builder.pairs(), ignored);
        if (ignored.missing_partners || ignored.weight_mismatches)
            throw std::runtime_error("sparse quotient invariant failed");
        reorient_quotient(quotient, builder.pairs(),
                          seed + join_candidates.size());
        join_candidates.push_back(
            {std::move(distribution), std::move(quotient)});
    }
    if (join_candidates.size() < 2)
        throw std::runtime_error("insufficient exact join candidates");

    for (size_t index = 0; index < join_candidates.size(); index++) {
        size_t other = (index * 7 + 3) % join_candidates.size();
        U128 direct = direct_join(join_candidates[index].first,
                                  join_candidates[other].first);
        U128 orbit = quotient_join(join_candidates[index].second,
                                   join_candidates[other].second,
                                   builder.pairs());
        if (direct != orbit)
            throw std::runtime_error(
                "quotient join mismatch: direct=" + u128_string(direct) +
                " orbit=" + u128_string(orbit));
        result.join_checks++;
    }
    return result;
}

static void print_result(unsigned rows, unsigned columns, unsigned samples,
                         const GeometryResult& result) {
    long double quotient_entries =
        (static_cast<long double>(result.entries) + result.fixed_entries) / 2;
    long double reduction = quotient_entries
        ? static_cast<long double>(result.entries) / quotient_entries : 0;
    std::cout << "TOKEN_PLANE rows=" << rows
              << " columns=" << columns
              << " samples=" << samples
              << " entries=" << result.entries
              << " fixed=" << result.fixed_entries
              << " fixed_fraction="
              << static_cast<double>(result.fixed_entries) /
                     static_cast<double>(result.entries)
              << " quotient_entries=" << uint64_t(quotient_entries)
              << " reduction=" << static_cast<double>(reduction)
              << " max_support=" << result.maximum_support
              << " max_weight_classes=" << result.maximum_weight_classes
              << " max_orbit_classes=" << result.maximum_orbit_classes
              << " join_checks=" << result.join_checks
              << " missing=" << result.missing_partners
              << " weight_mismatches=" << result.weight_mismatches
              << " prefix_mismatches=" << result.prefix_mismatches
              << " exact=OK\n";
}

}  // namespace

int main(int argc, char** argv) {
    unsigned samples = argc > 1 ? unsigned(std::strtoul(argv[1], nullptr, 10))
                                : 256;
    unsigned joins = argc > 2 ? unsigned(std::strtoul(argv[2], nullptr, 10))
                              : 32;
    if (!samples || !joins) return 2;
    try {
        GeometryResult eight = run_geometry(
            8, 4, 7, samples, joins, UINT64_C(0x8a4f1177));
        GeometryResult seven = run_geometry(
            7, 5, 5, samples, joins, UINT64_C(0x7a5f1177));
        print_result(8, 4, samples, eight);
        print_result(7, 5, samples, seven);
        if (eight.missing_partners || eight.weight_mismatches ||
            eight.prefix_mismatches || seven.missing_partners ||
            seven.weight_mismatches || seven.prefix_mismatches)
            return 1;
    } catch (const std::exception& error) {
        std::cerr << "token-plane quotient probe: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
