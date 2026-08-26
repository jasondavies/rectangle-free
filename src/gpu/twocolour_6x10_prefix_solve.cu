#define GRID_ROWS 6
#define GRID_COLUMNS 10
#define LEFT_COLUMNS 5
#define RIGHT_COLUMNS 5
#define ORBIT_ROW_BITS 10
#define ORBIT_MAGIC "R6ORB01"
#define TWCOLOUR_GEOMETRY "6x10"
#define TWCOLOUR_RESULT_MAGIC "RECT6X10_PREFIX_RESULT"
#define TWCOLOUR_TRANSPOSE_QUOTIENT 0
#define TWCOLOUR_PRESERVE_JOIN_ORDER 1
#ifndef TWCOLOUR_PREFIX_PAIR_COUNT
#define TWCOLOUR_PREFIX_PAIR_COUNT 4
#endif
#ifndef TWCOLOUR_THREADS
#define TWCOLOUR_THREADS 128
#endif

// The equal-width six-row solver shares one canonical 6x5 cache between both
// sides. Token-plane quotienting and the architecture-native join are
// intrinsic properties of the maintained production core.
#include "twocolour_8x8_prefix_solve.cu"
