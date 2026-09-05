// Exact one-word WL neighbour-colour histogram, shared with its unit test.
// A five-bit count cannot carry into the next field: a simple <=32-vertex
// graph has at most 31 neighbours of any colour. Subtraction also recovers
// the correct packed count when the one colour class itself has size 32.
static inline uint64_t wl_neighbor_histogram32(uint32_t row, uint32_t mask,
                                              const uint64_t* units,
                                              uint64_t all_units, int complement) {
    uint32_t bits = complement ? (mask ^ row) : row;
    uint64_t sum = 0;
    while (bits) {
        unsigned v = (unsigned)__builtin_ctz(bits);
        sum += units[v];
        bits &= bits - 1;
    }
    return complement ? all_units - sum : sum;
}
