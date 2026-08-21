#define GRID_ROWS 7
#define GRID_COLUMNS 9
#define LEFT_COLUMNS 4
#define RIGHT_COLUMNS 5
#define ORBIT_ROW_BITS 9
#define ORBIT_MAGIC "R7ORB09"

#include "twocolour_prefix_core.cuh"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s CANONICAL_7X5.orbits PACKED_7X5.cache\n",
                     argv[0]);
        return 2;
    }
    initialise_tables();
    validate_mask_split();
    PackedUniversalCache cache =
        build_packed_universal_cache_from_orbits(argv[1], false);
    write_packed_universal_cache(cache, argv[2]);
    return 0;
}
