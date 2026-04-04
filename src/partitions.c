#include "partition_poly.h"

// --- INITIALISATION ---

int num_partitions = 0;
int perm_count = 0;
int max_partition_capacity = 0;
int max_complex_per_partition = 0;
Partition* partitions = NULL;
int (*perms)[MAX_ROWS] = NULL;
uint16_t* perm_table = NULL;
uint16_t* perm_order_by_value = NULL;
uint16_t* perm_value_prefix_end = NULL;
uint16_t* partition_id_lookup = NULL;
uint32_t partition_id_lookup_size = 0;
uint64_t factorial[20];
ComplexMask* overlap_mask = NULL;
ComplexMask* intra_mask = NULL;
Poly* partition_weight_poly = NULL;
uint8_t* partition_weight4 = NULL;
#if RECT_COUNT_K4_FEASIBILITY
uint32_t* pair_shadow_mask = NULL;
uint8_t* pair_shadow_pairs = NULL;
uint8_t* suffix_min_pairs = NULL;
int pair_index[MAX_ROWS][MAX_ROWS];
int num_row_pairs = 0;
int min_partition_pairs = 0;
#endif

static int bell_number_upper_bound(int rows) {
    static const int bell_numbers[] = {0, 1, 2, 5, 15, 52, 203, 877};
    if (rows < 0 || rows >= (int)(sizeof(bell_numbers) / sizeof(bell_numbers[0]))) {
        fprintf(stderr, "Unsupported row count for Bell number lookup: %d\n", rows);
        exit(1);
    }
    return bell_numbers[rows];
}

void init_row_dependent_tables(void) {
    max_partition_capacity = bell_number_upper_bound(g_rows);
    perm_count = (int)factorial[g_rows];
    max_complex_per_partition = g_rows / 2;

    partitions = checked_calloc((size_t)max_partition_capacity, sizeof(*partitions), "partitions");
    perms = checked_calloc((size_t)perm_count, sizeof(*perms), "perms");
}

void init_partition_lookup_tables(void) {
    size_t partition_count = (size_t)num_partitions;
    partition_id_lookup_size = 1u << (3 * g_rows);

    perm_table = checked_calloc(partition_count * (size_t)perm_count, sizeof(*perm_table), "perm_table");
    perm_order_by_value =
        checked_calloc(partition_count * (size_t)perm_count, sizeof(*perm_order_by_value),
                       "perm_order_by_value");
    perm_value_prefix_end =
        checked_calloc(partition_count * partition_count, sizeof(*perm_value_prefix_end),
                       "perm_value_prefix_end");
    partition_id_lookup =
        checked_aligned_alloc(64, (size_t)partition_id_lookup_size * sizeof(*partition_id_lookup),
                              "partition_id_lookup");
    memset(partition_id_lookup, 0xff, (size_t)partition_id_lookup_size * sizeof(*partition_id_lookup));
    overlap_mask = checked_calloc(partition_count * partition_count * (size_t)max_complex_per_partition,
                                  sizeof(*overlap_mask), "overlap_mask");
    intra_mask = checked_calloc(partition_count * (size_t)max_complex_per_partition,
                                sizeof(*intra_mask), "intra_mask");
    partition_weight_poly =
        checked_calloc(partition_count, sizeof(*partition_weight_poly), "partition_weight_poly");
    partition_weight4 = checked_calloc(partition_count, sizeof(*partition_weight4), "partition_weight4");
#if RECT_COUNT_K4_FEASIBILITY
    pair_shadow_mask = checked_calloc(partition_count, sizeof(*pair_shadow_mask), "pair_shadow_mask");
    pair_shadow_pairs = checked_calloc(partition_count, sizeof(*pair_shadow_pairs), "pair_shadow_pairs");
    suffix_min_pairs = checked_calloc(partition_count, sizeof(*suffix_min_pairs), "suffix_min_pairs");
#endif
}

void free_row_dependent_tables(void) {
    free(partitions);
    free(perms);
    free(perm_table);
    free(perm_order_by_value);
    free(perm_value_prefix_end);
    free(partition_id_lookup);
    free(overlap_mask);
    free(intra_mask);
    free(partition_weight_poly);
    free(partition_weight4);
#if RECT_COUNT_K4_FEASIBILITY
    free(pair_shadow_mask);
    free(pair_shadow_pairs);
    free(suffix_min_pairs);
#endif

    partitions = NULL;
    perms = NULL;
    perm_table = NULL;
    perm_order_by_value = NULL;
    perm_value_prefix_end = NULL;
    partition_id_lookup = NULL;
    partition_id_lookup_size = 0;
    overlap_mask = NULL;
    intra_mask = NULL;
    partition_weight_poly = NULL;
    partition_weight4 = NULL;
#if RECT_COUNT_K4_FEASIBILITY
    pair_shadow_mask = NULL;
    pair_shadow_pairs = NULL;
    suffix_min_pairs = NULL;
    num_row_pairs = 0;
    min_partition_pairs = 0;
#endif
    num_partitions = 0;
    perm_count = 0;
    max_partition_capacity = 0;
    max_complex_per_partition = 0;
}

void generate_permutations(void) {
    int p[MAX_ROWS];
    for (int i = 0; i < g_rows; i++) p[i] = i;
    int count = 0;
    while (1) {
        memcpy(perms[count], p, sizeof(int) * g_rows);
        count++;
        int i = g_rows - 2;
        while (i >= 0 && p[i] >= p[i + 1]) i--;
        if (i < 0) break;
        int j = g_rows - 1;
        while (p[j] <= p[i]) j--;
        int temp = p[i]; p[i] = p[j]; p[j] = temp;
        int l = i + 1, r = g_rows - 1;
        while (l < r) { temp = p[l]; p[l] = p[r]; p[r] = temp; l++; r--; }
    }
    perm_count = count;
}

static void normalize_partition(uint8_t* p) {
    uint8_t map[MAX_ROWS];
    memset(map, 255, sizeof(map));
    uint8_t next = 0;
    for (int i = 0; i < g_rows; i++) {
        if (map[p[i]] == 255) map[p[i]] = next++;
        p[i] = map[p[i]];
    }
}

static int partition_pair_shadow_size(const Partition* part) {
    int pairs = 0;
    for (int ci = 0; ci < part->num_complex; ci++) {
        int block = part->complex_blocks[ci];
        int sz = __builtin_popcount(part->block_masks[block]);
        pairs += (sz * (sz - 1)) / 2;
    }
    return pairs;
}

static int partition_largest_complex_block(const Partition* part) {
    int best = 0;
    for (int ci = 0; ci < part->num_complex; ci++) {
        int block = part->complex_blocks[ci];
        int sz = __builtin_popcount(part->block_masks[block]);
        if (sz > best) best = sz;
    }
    return best;
}

static int compare_partition_hardness(const void* lhs, const void* rhs) {
    const Partition* a = (const Partition*)lhs;
    const Partition* b = (const Partition*)rhs;

    if (a->num_complex != b->num_complex) return b->num_complex - a->num_complex;

    int ap = partition_pair_shadow_size(a);
    int bp = partition_pair_shadow_size(b);
    if (ap != bp) return bp - ap;

    int al = partition_largest_complex_block(a);
    int bl = partition_largest_complex_block(b);
    if (al != bl) return bl - al;

    if (a->num_blocks != b->num_blocks) return b->num_blocks - a->num_blocks;

    return memcmp(a->mapping, b->mapping, (size_t)g_rows);
}

void reorder_partitions_by_hardness(void) {
    qsort(partitions, (size_t)num_partitions, sizeof(partitions[0]), compare_partition_hardness);
}

void generate_partitions_recursive(int idx, uint8_t* current, int max_val) {
    if (idx == g_rows) {
        Partition part;
        memset(&part, 0, sizeof(part));
        memcpy(part.mapping, current, g_rows);
        part.num_blocks = max_val + 1;
#if RECT_COUNT_K4
        if (part.num_blocks > 4) return;
#endif
        
        int counts[MAX_ROWS];
        for(int k=0; k<g_rows; k++) counts[k]=0;
        
        for (int r = 0; r < g_rows; r++) {
            part.block_masks[current[r]] |= (1 << r);
            counts[current[r]]++;
        }
        for (int b = 0; b < part.num_blocks; b++) {
            part.is_complex[b] = (counts[b] >= 2);
            if (part.is_complex[b]) part.num_complex++;
            else part.num_singletons++;
        }
        int ci = 0;
        for (int b = 0; b < part.num_blocks; b++) {
            if (part.is_complex[b]) part.complex_blocks[ci++] = (uint8_t)b;
        }
        if (num_partitions >= max_partition_capacity) {
            fprintf(stderr, "Partition table capacity exceeded for %d rows\n", g_rows);
            exit(1);
        }
        partitions[num_partitions++] = part;
        return;
    }
    for (int i = 0; i <= max_val; i++) {
        current[idx] = i;
        generate_partitions_recursive(idx + 1, current, max_val);
    }
    if (max_val < g_rows - 1
#if RECT_COUNT_K4
        && max_val + 2 <= 4
#endif
    ) {
        current[idx] = max_val + 1;
        generate_partitions_recursive(idx + 1, current, max_val + 1);
    }
}

static int get_partition_id(uint8_t* map) {
    uint32_t key = 0;
    for (int i = 0; i < g_rows; i++) {
        key |= (uint32_t)map[i] << (3 * i);
    }
    if (key >= partition_id_lookup_size) return -1;
    uint16_t val = partition_id_lookup[key];
    return (val == UINT16_MAX) ? -1 : (int)val;
}

void build_partition_id_lookup(void) {
    for (int id = 0; id < num_partitions; id++) {
        uint32_t key = 0;
        for (int i = 0; i < g_rows; i++) {
            key |= (uint32_t)partitions[id].mapping[i] << (3 * i);
        }
        partition_id_lookup[key] = (uint16_t)id;
    }
}

void build_perm_table(void) {
    uint8_t temp[MAX_ROWS];
    
    for (int id = 0; id < num_partitions; id++) {
        for (int pi = 0; pi < perm_count; pi++) {
            for (int r = 0; r < g_rows; r++) {
                temp[r] = partitions[id].mapping[perms[pi][r]];
            }
            normalize_partition(temp);
            int pid = get_partition_id(temp);
            if (pid < 0 || pid > UINT16_MAX) {
                fprintf(stderr, "partition id out of range in build_perm_table: %d\n", pid);
                exit(1);
            }
            perm_table[(size_t)id * (size_t)perm_count + (size_t)pi] = (uint16_t)pid;
        }
    }
}

void build_terminal_perm_order_tables(void) {
    uint16_t* counts =
        checked_calloc((size_t)num_partitions, sizeof(*counts), "terminal_perm_order_counts");
    uint16_t* offsets =
        checked_calloc((size_t)num_partitions, sizeof(*offsets), "terminal_perm_order_offsets");

    for (int id = 0; id < num_partitions; id++) {
        uint16_t* sorted_row = perm_order_by_value + (size_t)id * (size_t)perm_count;
        uint16_t* prefix_row = perm_value_prefix_end + (size_t)id * (size_t)num_partitions;
        const uint16_t* perm_row = perm_table + (size_t)id * (size_t)perm_count;

        memset(counts, 0, (size_t)num_partitions * sizeof(*counts));
        for (int p = 0; p < perm_count; p++) {
            counts[perm_row[p]]++;
        }

        uint16_t next = 0;
        for (int value = 0; value < num_partitions; value++) {
            offsets[value] = next;
            next = (uint16_t)(next + counts[value]);
            prefix_row[value] = next;
        }

        for (int p = 0; p < perm_count; p++) {
            uint16_t value = perm_row[p];
            sorted_row[offsets[value]++] = (uint16_t)p;
        }
    }

    free(counts);
    free(offsets);
}

void build_overlap_table(void) {
    memset(overlap_mask, 0,
           (size_t)num_partitions * (size_t)num_partitions * (size_t)max_complex_per_partition *
               sizeof(*overlap_mask));
    memset(intra_mask, 0,
           (size_t)num_partitions * (size_t)max_complex_per_partition * sizeof(*intra_mask));
    for (int pid1 = 0; pid1 < num_partitions; pid1++) {
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            ComplexMask mask = 0;
            for (int i2 = 0; i2 < partitions[pid1].num_complex; i2++) {
                if (i1 != i2) mask |= (ComplexMask)(1u << i2);
            }
            intra_mask_row(pid1)[i1] = mask;
        }
        for (int i1 = 0; i1 < partitions[pid1].num_complex; i1++) {
            int b1 = partitions[pid1].complex_blocks[i1];
            uint32_t m1 = partitions[pid1].block_masks[b1];
            for (int pid2 = 0; pid2 < num_partitions; pid2++) {
                ComplexMask mask = 0;
                for (int i2 = 0; i2 < partitions[pid2].num_complex; i2++) {
                    int b2 = partitions[pid2].complex_blocks[i2];
                    uint32_t m2 = partitions[pid2].block_masks[b2];
                    if (__builtin_popcount(m1 & m2) >= 2) {
                        mask |= (ComplexMask)(1u << i2);
                    }
                }
                overlap_mask_row(pid1, pid2)[i1] = mask;
            }
        }
    }
}

#if RECT_COUNT_K4_FEASIBILITY
void init_pair_index(void) {
    memset(pair_index, -1, sizeof(pair_index));
    num_row_pairs = 0;
    for (int i = 0; i < g_rows; i++) {
        for (int j = i + 1; j < g_rows; j++) {
            pair_index[i][j] = num_row_pairs;
            pair_index[j][i] = num_row_pairs;
            num_row_pairs++;
        }
    }
}

void build_partition_shadow_table(void) {
    min_partition_pairs = MAX_ROW_PAIRS;
    memset(pair_shadow_mask, 0, (size_t)num_partitions * sizeof(*pair_shadow_mask));
    memset(pair_shadow_pairs, 0, (size_t)num_partitions * sizeof(*pair_shadow_pairs));
    memset(suffix_min_pairs, 0, (size_t)num_partitions * sizeof(*suffix_min_pairs));

    for (int pid = 0; pid < num_partitions; pid++) {
        uint32_t shadow = 0;
        const Partition* part = &partitions[pid];
        for (int ci = 0; ci < part->num_complex; ci++) {
            int block = part->complex_blocks[ci];
            uint32_t mask = part->block_masks[block];
            for (int i = 0; i < g_rows; i++) {
                if (((mask >> i) & 1U) == 0) continue;
                for (int j = i + 1; j < g_rows; j++) {
                    if ((mask >> j) & 1U) {
                        shadow |= (uint32_t)(1U << pair_index[i][j]);
                    }
                }
            }
        }
        pair_shadow_mask[pid] = shadow;
        pair_shadow_pairs[pid] = (uint8_t)__builtin_popcount(shadow);
        if (pair_shadow_pairs[pid] < min_partition_pairs) {
            min_partition_pairs = pair_shadow_pairs[pid];
        }
    }

    if (num_partitions == 0) min_partition_pairs = 0;
    for (int pid = num_partitions - 1; pid >= 0; pid--) {
        if (pid == num_partitions - 1) suffix_min_pairs[pid] = pair_shadow_pairs[pid];
        else suffix_min_pairs[pid] = pair_shadow_pairs[pid] < suffix_min_pairs[pid + 1]
                                       ? pair_shadow_pairs[pid]
                                       : suffix_min_pairs[pid + 1];
    }
}
#endif

void build_partition_weight_table(void) {
    for (int pid = 0; pid < num_partitions; pid++) {
        Poly weight;
        poly_one_ref(&weight);
        int c = partitions[pid].num_complex;
        int s = partitions[pid].num_singletons;
        if (s > 0) {
            poly_mul_falling_ref(&weight, c, s, &weight);
        }
        partition_weight_poly[pid] = weight;
    }
}

#if RECT_COUNT_K4
static inline uint8_t falling4_weight(int c, int s) {
    if (c + s > 4) return 0;
    uint8_t w = 1;
    for (int i = 0; i < s; i++) w = (uint8_t)(w * (uint8_t)(4 - c - i));
    return w;
}

void build_partition_weight4_table(void) {
    for (int pid = 0; pid < num_partitions; pid++) {
        partition_weight4[pid] =
            falling4_weight(partitions[pid].num_complex, partitions[pid].num_singletons);
    }
}
#endif

#if RECT_COUNT_K4
void weight_accum_one(WeightAccum* out) {
    *out = 1;
}

void weight_accum_from_partition(int pid, WeightAccum* out) {
    *out = (WeightAccum)partition_weight4[pid];
}

void weight_accum_mul_partition(const WeightAccum* src, int pid, WeightAccum* out) {
    *out = (*src) * (WeightAccum)partition_weight4[pid];
}

void weight_accum_scale_to_poly(const WeightAccum* weight_prod, long long mult_coeff,
                                long long row_orbit, uint64_t graph_count4, Poly* out) {
    WeightAccum total = *weight_prod;
    total *= (WeightAccum)mult_coeff;
    total *= (WeightAccum)row_orbit;
    total *= (WeightAccum)graph_count4;
    poly_zero(out);
    out->coeffs[0] = (PolyCoeff)total;
}
#else
void weight_accum_one(WeightAccum* out) {
    poly_one_ref(out);
}

void weight_accum_from_partition(int pid, WeightAccum* out) {
    *out = partition_weight_poly[pid];
}

void weight_accum_mul_partition(const WeightAccum* src, int pid, WeightAccum* out) {
    poly_mul_ref(src, &partition_weight_poly[pid], out);
}
#endif
