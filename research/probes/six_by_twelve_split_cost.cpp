#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1201"
#define TWCOLOUR_WIDE_ORBIT_RECORD 1

#include "../../src/gpu/twocolour_gpu_common.cuh"

#include <atomic>

namespace {

struct SplitTotals {
    U128 selected_product = 0;
    U128 complement_product = 0;
    uint64_t left_selected = 0;
    uint64_t left_complement = 0;
    uint64_t right_selected = 0;
    uint64_t right_complement = 0;
};

static PrefixKey extract_columns(U128 key, unsigned begin, unsigned count) {
    PrefixKey result = 0;
    const U128 row_mask = (U128(1) << COLUMNS) - 1U;
    const U128 column_mask = (U128(1) << count) - 1U;
    for (int row = 0; row < ROWS; ++row) {
        const unsigned shift = COLUMNS * (ROWS - 1U - row);
        const U128 pattern = (key >> shift) & row_mask;
        result = (result << count) |
                 PrefixKey((pattern >> begin) & column_mask);
    }
    return result;
}

static uint64_t support_size(PrefixKey key, unsigned width, bool complement) {
    return quotient_token_planes(
               build_distribution(key, int(width), complement)).entries.size();
}

static SplitTotals measure(U128 key, unsigned left_width) {
    const unsigned right_width = COLUMNS - left_width;
    const PrefixKey left = extract_columns(key, 0, left_width);
    const PrefixKey right = extract_columns(key, left_width, right_width);
    SplitTotals result;
    result.left_selected = support_size(left, left_width, false);
    result.left_complement = support_size(left, left_width, true);
    result.right_selected = support_size(right, right_width, false);
    result.right_complement = support_size(right, right_width, true);
    result.selected_product = U128(result.left_selected) * result.right_selected;
    result.complement_product =
        U128(result.left_complement) * result.right_complement;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "Usage: %s SAMPLE_6X12.orbits [SAMPLES=1024]\n",
                     argv[0]);
        return 2;
    }
    const uint64_t wanted = argc == 3 ? std::strtoull(argv[2], nullptr, 10)
                                      : 1024;
    std::ifstream input(argv[1], std::ios::binary);
    char magic[8];
    uint32_t width = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || width != COLUMNS ||
        !wanted || wanted > count)
        throw std::runtime_error("invalid 6x12 sample");
    std::vector<OrbitRecord> records(size_t(wanted), OrbitRecord{});
    for (uint64_t sample = 0; sample < wanted; ++sample) {
        const uint64_t index =
            (U128(sample * 2 + 1) * count) / (U128(2) * wanted);
        input.seekg(std::streamoff(20 + index * sizeof(OrbitRecord)));
        input.read(reinterpret_cast<char*>(&records[size_t(sample)]),
                   sizeof(OrbitRecord));
    }
    if (!input) throw std::runtime_error("truncated 6x12 sample");
    initialise_tables();
    U128 cost66 = 0;
    U128 cost57 = 0;
    uint64_t entries66 = 0;
    uint64_t entries57_left = 0;
    uint64_t entries57_right = 0;
    std::atomic<uint64_t> completed{0};
    double start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1) reduction(+:cost66,cost57,entries66,entries57_left,entries57_right)
    for (long long index = 0; index < (long long)records.size(); ++index) {
        const OrbitRecord& record = records[size_t(index)];
        const U128 key = (U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) |
                         record.low;
        SplitTotals six = measure(key, 6);
        SplitTotals five = measure(key, 5);
        cost66 += six.selected_product + six.complement_product;
        cost57 += five.selected_product + five.complement_product;
        entries66 += six.left_selected + six.left_complement +
                     six.right_selected + six.right_complement;
        entries57_left += five.left_selected + five.left_complement;
        entries57_right += five.right_selected + five.right_complement;
        completed.fetch_add(1, std::memory_order_relaxed);
    }
    std::printf(
        "SIX_BY_TWELVE_SPLIT_COST samples=%zu cost_6_6=%s cost_5_7=%s "
        "cost_ratio_5_7=%.12f entries_6_6=%llu entries_5=%llu "
        "entries_7=%llu mean_7=%.6f seconds=%.6f OK\n",
        records.size(), u128_string(cost66).c_str(),
        u128_string(cost57).c_str(), double(cost57) / double(cost66),
        (unsigned long long)entries66,
        (unsigned long long)entries57_left,
        (unsigned long long)entries57_right,
        double(entries57_right) / (2.0 * records.size()),
        seconds_now() - start);
    return 0;
}
