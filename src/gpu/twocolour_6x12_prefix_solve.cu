#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1202"
#define TWCOLOUR_GEOMETRY "6x12"
#define TWCOLOUR_RESULT_MAGIC "RECT6X12_PREFIX_RESULT"
#define TWCOLOUR_TRANSPOSE_QUOTIENT 0
#define TWCOLOUR_PRESERVE_JOIN_ORDER 1
#define TWCOLOUR_WIDE_ORBIT_RECORD 1
#define TWCOLOUR_RIGHT_MAJOR_ORBIT_RECORD 1
#define TWCOLOUR_RETAINED_ORBIT_CORPUS 1
#define TWCOLOUR_SIX_BY_SIX_CACHE_ARTIFACT 1
#ifndef TWCOLOUR_PREFIX_PAIR_COUNT
#define TWCOLOUR_PREFIX_PAIR_COUNT 5
#endif
#if TWCOLOUR_PREFIX_PAIR_COUNT == 5 && !defined(TWCOLOUR_PREFIX_PAIR_MASK)
// Exhaustive A/B testing of all 15 unlabeled five-edge graphs on six rows
// selected K4 minus one edge.  The generic ranking happens to choose the same
// topology, but spelling it out binds result provenance to the measured shape.
#define TWCOLOUR_PREFIX_PAIR_MASK 0x67
#endif
#ifndef TWCOLOUR_THREADS
#define TWCOLOUR_THREADS 128
#endif

// The symmetric six-row solver shares one canonical 6x6 cache between both
// sides.  Token-plane quotienting and the architecture-native exact join are
// mandatory properties of the maintained production representation.
#include "twocolour_8x8_prefix_solve.cu"
