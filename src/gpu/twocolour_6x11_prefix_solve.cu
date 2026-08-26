#define GRID_ROWS 6
#define GRID_COLUMNS 11
#define LEFT_COLUMNS 5
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 11
#define ORBIT_MAGIC "R6W1101"
#define TWCOLOUR_GEOMETRY "6x11"
#define TWCOLOUR_RESULT_MAGIC "RECT6X11_PREFIX_RESULT"
#define TWCOLOUR_TRANSPOSE_QUOTIENT 0
#define TWCOLOUR_PRESERVE_JOIN_ORDER 1
#define TWCOLOUR_WIDE_ORBIT_RECORD 1
#ifndef TWCOLOUR_PREFIX_PAIR_COUNT
#define TWCOLOUR_PREFIX_PAIR_COUNT 4
#endif
#ifndef TWCOLOUR_THREADS
#define TWCOLOUR_THREADS 128
#endif

// The asymmetric six-row solver retains independent canonical 6x5 and 6x6
// token-plane-quotient caches. The 66-bit outer key is compact only on disk;
// both labelled half keys remain native uint64_t values in the hot path.
#include "twocolour_8x8_prefix_solve.cu"
