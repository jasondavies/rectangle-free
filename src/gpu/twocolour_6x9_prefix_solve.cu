#define GRID_ROWS 6
#define GRID_COLUMNS 9
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 5
#define ORBIT_ROW_BITS 10
#define ORBIT_MAGIC "R6ORB01"
#define TWCOLOUR_GEOMETRY "6x9"
#define TWCOLOUR_RESULT_MAGIC "RECT6X9_PREFIX_RESULT"
#define TWCOLOUR_TRANSPOSE_QUOTIENT 0
#define TWCOLOUR_PRESERVE_JOIN_ORDER 1
#ifndef TWCOLOUR_PREFIX_PAIR_COUNT
#define TWCOLOUR_PREFIX_PAIR_COUNT 3
#endif
#ifndef TWCOLOUR_THREADS
#define TWCOLOUR_THREADS 128
#endif

// The maintained six-row solver shares the production grouped-layout and
// architecture-native join implementation with the 8x8 solver.  Keeping this
// entry point declarative makes token-plane quotienting and subsequent kernel
// fixes intrinsic rather than optional historical tuning modes.
#include "twocolour_8x8_prefix_solve.cu"
