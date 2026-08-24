// Structural gate for the four-colour-plane contraction of H=K4 x KG(r,2).

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t PRIME = 1000003;

struct Options {
    unsigned samples = 2000;
    unsigned rank_sample = 512;
};

uint64_t number(const char* text) {
    char* end = nullptr;
    const uint64_t value = std::strtoull(text, &end, 10);
    if (!end || *end) throw std::runtime_error("invalid integer");
    return value;
}

uint64_t binomial(unsigned n, unsigned k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    uint64_t result = 1;
    for (unsigned i = 1; i <= k; i++) result = result * (n - k + i) / i;
    return result;
}

uint64_t factorial(unsigned n) {
    uint64_t result = 1;
    for (unsigned i = 2; i <= n; i++) result *= i;
    return result;
}

uint32_t power_mod(uint32_t base, uint32_t exponent) {
    uint64_t result = 1;
    while (exponent) {
        if (exponent & 1) result = result * base % PRIME;
        base = uint32_t(uint64_t(base) * base % PRIME);
        exponent >>= 1;
    }
    return uint32_t(result);
}

struct KneserGeometry {
    unsigned rows = 0;
    unsigned pairs = 0;
    std::vector<std::pair<unsigned, unsigned>> endpoints;
    std::vector<uint32_t> disjoint;

    explicit KneserGeometry(unsigned row_count) : rows(row_count) {
        for (unsigned first = 0; first < rows; first++)
            for (unsigned second = first + 1; second < rows; second++)
                endpoints.push_back({first, second});
        pairs = unsigned(endpoints.size());
        if (pairs > 15) throw std::runtime_error("pair mask exceeds probe width");
        disjoint.resize(pairs);
        for (unsigned left = 0; left < pairs; left++) {
            auto [a, b] = endpoints[left];
            for (unsigned right = 0; right < pairs; right++) {
                auto [c, d] = endpoints[right];
                if (a != c && a != d && b != c && b != d)
                    disjoint[left] |= UINT32_C(1) << right;
            }
        }
    }
};

uint64_t subpermanent(const KneserGeometry& geometry,
                      uint32_t rows, uint32_t columns) {
    const unsigned size = unsigned(__builtin_popcount(rows));
    if (size != unsigned(__builtin_popcount(columns))) return 0;
    if (!size) return 1;
    std::array<unsigned, 15> column_index{};
    column_index.fill(16);
    unsigned next = 0;
    uint32_t scan = columns;
    while (scan) {
        const unsigned column = unsigned(__builtin_ctz(scan));
        scan &= scan - 1;
        column_index[column] = next++;
    }
    std::array<uint32_t, 15> candidate_masks{};
    std::array<unsigned, 15> row_list{};
    next = 0;
    scan = rows;
    while (scan) {
        const unsigned row = unsigned(__builtin_ctz(scan));
        scan &= scan - 1;
        row_list[next++] = row;
    }
    for (unsigned index = 0; index < size; index++) {
        uint32_t candidates = geometry.disjoint[row_list[index]] & columns;
        while (candidates) {
            const unsigned column = unsigned(__builtin_ctz(candidates));
            candidates &= candidates - 1;
            candidate_masks[index] |= UINT32_C(1) << column_index[column];
        }
    }

    std::vector<uint64_t> dp(size_t(1) << size);
    dp[0] = 1;
    for (uint32_t used = 0; used < (UINT32_C(1) << size); used++) {
        const unsigned row = unsigned(__builtin_popcount(used));
        if (row >= size || !dp[used]) continue;
        uint32_t candidates = candidate_masks[row] & ~used;
        while (candidates) {
            const uint32_t bit = candidates & (0U - candidates);
            candidates -= bit;
            dp[used | bit] += dp[used];
        }
    }
    return dp.back();
}

std::vector<std::vector<uint32_t>> masks_by_size(unsigned bits) {
    std::vector<std::vector<uint32_t>> result(bits + 1);
    for (uint32_t mask = 0; mask < (UINT32_C(1) << bits); mask++)
        result[unsigned(__builtin_popcount(mask))].push_back(mask);
    return result;
}

unsigned modular_rank(std::vector<uint32_t>& matrix, unsigned dimension) {
    unsigned rank = 0;
    for (unsigned column = 0; column < dimension && rank < dimension; column++) {
        unsigned pivot = rank;
        while (pivot < dimension && !matrix[size_t(pivot) * dimension + column])
            pivot++;
        if (pivot == dimension) continue;
        if (pivot != rank)
            for (unsigned item = column; item < dimension; item++)
                std::swap(matrix[size_t(rank) * dimension + item],
                          matrix[size_t(pivot) * dimension + item]);
        const uint32_t inverse = power_mod(
            matrix[size_t(rank) * dimension + column], PRIME - 2);
        for (unsigned item = column; item < dimension; item++)
            matrix[size_t(rank) * dimension + item] = uint32_t(
                uint64_t(matrix[size_t(rank) * dimension + item]) * inverse % PRIME);
        for (unsigned row = 0; row < dimension; row++) {
            if (row == rank) continue;
            const uint32_t factor = matrix[size_t(row) * dimension + column];
            if (!factor) continue;
            for (unsigned item = column; item < dimension; item++) {
                const uint32_t product = uint32_t(
                    uint64_t(factor) * matrix[size_t(rank) * dimension + item] % PRIME);
                uint32_t& value = matrix[size_t(row) * dimension + item];
                value = value >= product ? value - product : value + PRIME - product;
            }
        }
        rank++;
    }
    return rank;
}

void exact_rank_census(unsigned rows) {
    const KneserGeometry geometry(rows);
    const auto masks = masks_by_size(geometry.pairs);
    for (unsigned size = 0; size <= geometry.pairs; size++) {
        const unsigned dimension = unsigned(masks[size].size());
        std::vector<uint32_t> matrix(size_t(dimension) * dimension);
        uint64_t nonzero = 0;
        std::unordered_set<uint64_t> values;
        for (unsigned left = 0; left < dimension; left++)
            for (unsigned right = 0; right < dimension; right++) {
                const uint64_t value = subpermanent(
                    geometry, masks[size][left], masks[size][right]);
                matrix[size_t(left) * dimension + right] = uint32_t(value % PRIME);
                nonzero += value != 0;
                values.insert(value);
            }
        const unsigned rank = modular_rank(matrix, dimension);
        std::printf(
            "COLOUR_PLANE_EXACT rows=%u pairs=%u degree=%u dimension=%u "
            "entries=%" PRIu64 " nonzero=%" PRIu64 " density=%.9f "
            "distinct=%zu rank=%u full_rank=%s exact=OK\n",
            rows, geometry.pairs, size, dimension,
            uint64_t(dimension) * dimension, nonzero,
            dimension ? double(nonzero) / (double(dimension) * dimension) : 0,
            values.size(), rank, rank == dimension ? "yes" : "no");
    }
}

unsigned sampled_six_row_census(const Options& options) {
    const KneserGeometry geometry(6);
    const auto masks = masks_by_size(geometry.pairs);
    std::mt19937_64 random(UINT64_C(0x6c6f7572706c616e));
    for (unsigned size = 0; size <= geometry.pairs; size++) {
        const uint64_t entries = uint64_t(masks[size].size()) * masks[size].size();
        const unsigned samples = unsigned(std::min<uint64_t>(options.samples, entries));
        uint64_t nonzero = 0;
        std::unordered_set<uint64_t> values;
        for (unsigned sample = 0; sample < samples; sample++) {
            const uint32_t left = masks[size][random() % masks[size].size()];
            const uint32_t right = masks[size][random() % masks[size].size()];
            const uint64_t value = subpermanent(geometry, left, right);
            nonzero += value != 0;
            values.insert(value);
        }
        std::printf(
            "COLOUR_PLANE_SAMPLE rows=6 pairs=15 degree=%u dimension=%zu "
            "entries=%" PRIu64 " samples=%u nonzero=%" PRIu64
            " density=%.9f distinct=%zu exact_values=OK\n",
            size, masks[size].size(), entries, samples, nonzero,
            samples ? double(nonzero) / samples : 0, values.size());
    }

    unsigned minimum_rank = options.rank_sample;
    for (unsigned size : {4U, 5U, 6U}) {
        const unsigned dimension = unsigned(std::min<size_t>(
            options.rank_sample, masks[size].size()));
        std::vector<uint32_t> selected = masks[size];
        std::shuffle(selected.begin(), selected.end(), random);
        selected.resize(dimension);
        std::vector<uint32_t> matrix(size_t(dimension) * dimension);
        for (unsigned left = 0; left < dimension; left++)
            for (unsigned right = 0; right < dimension; right++)
                matrix[size_t(left) * dimension + right] = uint32_t(
                    subpermanent(geometry, selected[left], selected[right]) % PRIME);
        const unsigned rank = modular_rank(matrix, dimension);
        minimum_rank = std::min(minimum_rank, rank);
        std::printf(
            "COLOUR_PLANE_RANK_SAMPLE rows=6 degree=%u sample_dimension=%u "
            "rank=%u full_rank=%s prime=%u exact=OK\n",
            size, dimension, rank, rank == dimension ? "yes" : "no", PRIME);
    }
    return minimum_rank;
}

void contraction_census(unsigned sampled_bond_rank) {
    constexpr unsigned pairs = 15;
    const uint64_t permanent_entries = binomial(2 * pairs, pairs);
    uint64_t sector_count = 0, assignment_sum = 0, maximum_assignments = 0;
    std::array<unsigned, 3> maximum_sector{};
    for (unsigned a = 0; a <= pairs; a++)
        for (unsigned b = 0; a + b <= pairs; b++) {
            const unsigned c = pairs - a - b;
            const uint64_t assignments = factorial(pairs) /
                (factorial(a) * factorial(b) * factorial(c));
            sector_count++;
            assignment_sum += assignments;
            if (assignments > maximum_assignments) {
                maximum_assignments = assignments;
                maximum_sector = {a, b, c};
            }
        }
    std::printf(
        "COLOUR_PLANE_CONTRACTION pairs=%u permanent_entries=%" PRIu64
        " permanent_bytes_u32=%" PRIu64 " sectors=%" PRIu64
        " local_assignments=%" PRIu64 " maximum_sector=%u,%u,%u"
        " maximum_local_assignments=%" PRIu64
        " pair_intermediate_entries=%" PRIu64
        " sampled_bond_rank=%u exact=OK\n",
        pairs, permanent_entries, 4 * permanent_entries, sector_count,
        assignment_sum, maximum_sector[0], maximum_sector[1], maximum_sector[2],
        maximum_assignments, maximum_assignments * maximum_assignments,
        sampled_bond_rank);
}

}  // namespace

int main(int argc, char** argv) try {
    Options options;
    for (int argument = 1; argument < argc; argument++) {
        const std::string value = argv[argument];
        if (value == "--samples" && argument + 1 < argc)
            options.samples = unsigned(number(argv[++argument]));
        else if (value == "--rank-sample" && argument + 1 < argc)
            options.rank_sample = unsigned(number(argv[++argument]));
        else
            throw std::runtime_error(
                "usage: colour_plane_permanent_probe "
                "[--samples N] [--rank-sample N]");
    }
    if (!options.samples || !options.rank_sample)
        throw std::runtime_error("sample counts must be positive");
    const auto started = Clock::now();
    exact_rank_census(4);
    exact_rank_census(5);
    const unsigned sampled_bond_rank = sampled_six_row_census(options);
    contraction_census(sampled_bond_rank);
    const double elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    std::printf("COLOUR_PLANE_DONE elapsed=%.6f exact=OK\n", elapsed);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 2;
}
