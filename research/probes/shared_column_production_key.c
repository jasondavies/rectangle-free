/* Research bridge to the actual 8x8 corpus representative rule. No generator
 * is executed and no production source is modified. Keep this a C TU. */
#define ORBIT_ROWS 8
#define ORBIT_MAX_COLUMNS 8
#define ORBIT_ROW_BITS 8
#define ORBIT_MAGIC "R8ORB01"
#define main shared_column_unused_generator_main
#include "../../tools/corpus/binary_orbit_augment.c"
#undef main

uint64_t shared_column_production_key(uint64_t key) {
    if (__builtin_popcountll(key) > 32) key = ~key;
    RowPattern rows[8];
    unpack_rows(key, rows);
    key = solve_representative(canonical_key(rows, 8));
    uint64_t partner = solve_transpose_partner(key, NULL);
    return key < partner ? key : partner;
}
