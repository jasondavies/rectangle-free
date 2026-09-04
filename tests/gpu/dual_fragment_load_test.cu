// Host execution of the actual fragment loaders; no CUDA device is needed.
#define GRID_COLUMNS 8
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 4
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "RTEST01"
#include "twocolour_prefix_algebra.cuh"
#if GRID_ROWS == 7
#include "twocolour_prefix_core.cuh"
#endif
#include "twocolour_weight_class_join.cuh"

#include <array>
#include <stdexcept>

static void check(bool condition) {
    if (!condition) throw std::runtime_error("dual fragment mismatch");
}

constexpr unsigned plane_bits = PAIRS - PREFIX_PAIR_COUNT;
constexpr uint64_t suffix_mask = (UINT64_C(1) << (2 * plane_bits)) - 1;

// Deliberately independent of the production swap and packing helpers.
static uint64_t reference_swap(uint64_t mask) {
    uint64_t result = 0;
    for (unsigned bit = 0; bit < 2 * plane_bits; ++bit)
        if ((mask >> bit) & 1)
            result |= UINT64_C(1) << ((bit + plane_bits) % (2 * plane_bits));
    return result;
}

static uint32_t reference_word(uint64_t mask, unsigned word) {
    return word < 2 ? uint32_t(mask >> (32 * word)) : 0;
}

static uint32_t reference_fp4(uint64_t mask, unsigned byte) {
    uint32_t result = 0;
    for (unsigned bit = 0; bit < 8; ++bit)
        if ((mask >> (8 * byte + bit)) & 1)
            result |= 2U << (4 * bit);
    return result;
}

int main() {
    std::array<PrefixSuffix, 64> suffixes{};
    uint64_t random = UINT64_C(0x284918471839);
    for (unsigned sample = 0; sample < 128; ++sample) {
        for (unsigned i = 0; i < suffixes.size(); ++i) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            uint64_t mask = random & suffix_mask;
            if (sample == 0) mask = 0;
            if (sample == 1) mask = suffix_mask;
            if (sample == 2) {
                mask &= (UINT64_C(1) << plane_bits) - 1;
                mask |= mask << plane_bits; // Fixed plane-exchange orbit.
            }
            if (sample >= 3 && sample < 3 + 2 * plane_bits)
                mask = UINT64_C(1) << (sample - 3);
            suffixes[i] = PrefixSuffix(mask);
        }
        for (unsigned count = 0; count <= 16; ++count) {
            for (unsigned lane = 0; lane < 32; ++lane) {
                for (bool swapped : {false, true}) {
                    constexpr unsigned offset = 3, base = 5;
                    const unsigned group = lane >> 2, word = lane & 3;
                    const unsigned bcount = std::min(count, 8U);
                    PtxFragmentA pa_other;
                    PtxFragmentB pb_other;
                    WeightClassFp4A fa_other;
                    WeightClassFp4B fb_other;
                    auto pa = load_weight_class_ptx_a(suffixes.data(), offset,
                        count, base, lane, swapped, &pa_other);
                    auto pb = load_weight_class_ptx_b(suffixes.data(), offset,
                        bcount, base, lane, swapped, &pb_other);
                    auto fa = load_weight_class_fp4_a(suffixes.data(), offset,
                        count, base, lane, swapped, &fa_other);
                    auto fb = load_weight_class_fp4_b(suffixes.data(), offset,
                        bcount, base, lane, swapped, &fb_other);
                    for (unsigned opposite = 0; opposite < 2; ++opposite) {
                        auto a = opposite ? pa_other : pa;
                        auto b = opposite ? pb_other : pb;
                        auto f = opposite ? fa_other : fa;
                        auto g = opposite ? fb_other : fb;
                        const bool exchange = swapped != bool(opposite);
                        auto mask = [&](unsigned row, unsigned size) {
                            uint64_t value = row < size
                                ? uint64_t(suffixes[offset + base + row]) : 0;
                            return exchange ? reference_swap(value) : value;
                        };
                        uint64_t a0 = mask(group, count);
                        uint64_t a1 = mask(group + 8, count);
                        uint64_t b0 = mask(group, bcount);
                        check(a.valid0 == (group < count));
                        check(a.valid1 == (group + 8 < count));
                        check(b.valid0 == (2 * word < bcount));
                        check(b.valid1 == (2 * word + 1 < bcount));
                        check(f.valid0 == a.valid0 && f.valid1 == a.valid1);
                        check(g.valid0 == b.valid0 && g.valid1 == b.valid1);
                        check(a.bits0 == reference_word(a0, word));
                        check(a.bits1 == reference_word(a1, word));
                        check(b.bits == reference_word(b0, word));
                        check(f.bits0 == reference_fp4(a0, word));
                        check(f.bits1 == reference_fp4(a1, word));
                        check(f.bits2 == reference_fp4(a0, word + 4));
                        check(f.bits3 == reference_fp4(a1, word + 4));
                        check(g.bits0 == reference_fp4(b0, word));
                        check(g.bits1 == reference_fp4(b0, word + 4));
                    }
                }
            }
        }
    }
    std::printf("dual fragment host checks passed: rows=%d\n", ROWS);
}
