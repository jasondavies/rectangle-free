// Compile-only fixture: no device is required to exercise the contracts.
#define GRID_COLUMNS 8
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 4
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "RTEST01"
#include "twocolour_prefix_algebra.cuh"

__global__ void check_split(uint64_t mask, uint64_t* output) {
    uint16_t prefix;
    PrefixSuffix suffix;
    split_pair_mask(mask, prefix, suffix);
    *output = join_pair_mask(prefix, suffix);
}
