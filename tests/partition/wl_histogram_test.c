#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../../research/probes/partition_wl_histogram.h"

int main(void) {
    uint32_t seed = 471;
    unsigned checks = 0;
    for (int n = 1; n <= 32; n++) for (int colours = 1; colours <= 12 && colours <= n; colours++) {
        uint32_t mask = n == 32 ? UINT32_MAX : (UINT32_C(1) << n) - 1;
        for (int sample = 0; sample < 256; sample++) {
            uint64_t units[32], all_units = 0;
            uint32_t class_masks[12] = {0};
            for (int v = 0; v < n; v++) {
                seed = seed * 1664525U + 1013904223U;
                int colour = v < colours ? v : (int)((seed >> 16) % colours);
                units[v] = UINT64_C(1) << (5 * colour);
                all_units += units[v];
                class_masks[colour] |= UINT32_C(1) << v;
            }
            seed = seed * 1664525U + 1013904223U;
            uint32_t row = seed & mask;
            if (sample == 0) row = 0;
            if (sample == 1) row = mask;
            // No self-loop: counts cannot exceed 31 even for a 32-vertex class.
            row &= ~(UINT32_C(1) << (sample % n));
            uint64_t expected = 0;
            for (int c = 0; c < colours; c++)
                expected |= (uint64_t)__builtin_popcount(row & class_masks[c]) << (5 * c);
            assert(wl_neighbor_histogram32(row, mask, units, all_units, 0) == expected);
            assert(wl_neighbor_histogram32(row, mask, units, all_units, 1) == expected);
            checks += 2;
        }
    }
    printf("WL histogram exact checks: %u\n", checks);
}
