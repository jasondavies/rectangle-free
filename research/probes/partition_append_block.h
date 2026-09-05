#ifndef RECT_PARTITION_APPEND_BLOCK_H
#define RECT_PARTITION_APPEND_BLOCK_H
#include <stdint.h>

/* Four row-major nibbles, transposed without another overlap-table access. */
static inline uint32_t partition_transpose4(uint32_t x) {
    uint32_t t = (x ^ (x >> 3)) & UINT32_C(0x0a0a);
    x ^= t ^ (t << 3);
    t = (x ^ (x >> 6)) & UINT32_C(0x00cc);
    return x ^ t ^ (t << 6);
}

#ifdef RECT_APPEND_BLOCK_AB
/* Requires the partition solver types; excluded from the standalone bit test.
 * Earlier columns own disjoint vertex ranges, so each old row is saved once. */
static inline void partition_append_overlap_blocks(PartialGraphState* st, int depth,
        int pid, const int* stack, int base_new, int num_complex,
        PartialGraphAppendFrame* frame) {
    for (int prev = 0; prev < depth; prev++) {
        int prev_base = st->base[prev];
        const ComplexMask* overlap = overlap_mask_row(pid, stack[prev]);
        uint32_t block = 0, touched = 0;
        for (int i = 0; i < num_complex; i++) {
            uint32_t row = overlap[i];
            st->g.adj[base_new + i] |= (uint64_t)row << prev_base;
            block |= row << (4 * i);
            touched |= row;
        }
        uint32_t reverse = partition_transpose4(block);
        while (touched) {
            int i = __builtin_ctz(touched);
            int v = prev_base + i;
            unsigned slot = frame->touched_prev_count++;
            frame->touched_prev_idx[slot] = (uint8_t)v;
            frame->touched_prev_old_adj[slot] = st->g.adj[v];
            st->g.adj[v] |= (uint64_t)((reverse >> (4 * i)) & 15u) << base_new;
            touched &= touched - 1;
        }
    }
}
#endif
#endif
