/* Exact CPU probe for a reusable canonical disjointness-query circuit. */
#define main twocolour_4x4_probe_historical_main
#include "twocolour_4x4_probe.c"
#undef main

#include <inttypes.h>

enum { QUERY_BITS = 2 * PAIRS };

typedef struct {
    uint64_t value;
    uint32_t low, high;
    uint8_t level;
} QueryCircuitNode;

typedef struct {
    uint64_t key;
    uint32_t value;
} QueryApplyEntry;

typedef struct {
    QueryCircuitNode* nodes;
    uint32_t* unique_slots;
    uint32_t* terminal_slots;
    QueryApplyEntry* apply;
    uint32_t* transformed;
    size_t count, node_cap, slot_capacity;
    size_t terminal_count, apply_capacity, apply_count;
    int aborted;
} QueryCircuit;

typedef struct {
    int prefix_index, complement;
    size_t support;
} CircuitCandidate;

typedef struct {
    uint8_t original_bit[QUERY_BITS];
    const char* name;
} VariableOrder;

static size_t query_power_of_two(size_t value) {
    size_t result = 16;
    while (result < value) {
        if (result > SIZE_MAX / 2) exit(1);
        result <<= 1;
    }
    return result;
}

static void circuit_init(QueryCircuit* circuit, size_t cap,
                         size_t zdd_nodes) {
    memset(circuit, 0, sizeof(*circuit));
    circuit->node_cap = cap;
    circuit->slot_capacity = query_power_of_two(cap * 2U + 16U);
    /* ADD Apply can visit more operand pairs than it ultimately retains
     * unique result nodes.  Give construction a wider disposable table so
     * NODE_CAP remains principally a persistent-circuit gate. */
    circuit->apply_capacity = query_power_of_two(cap * 4U + 16U);
    circuit->nodes = xcalloc(cap, sizeof(circuit->nodes[0]));
    circuit->unique_slots =
        xcalloc(circuit->slot_capacity, sizeof(circuit->unique_slots[0]));
    circuit->terminal_slots =
        xcalloc(circuit->slot_capacity, sizeof(circuit->terminal_slots[0]));
    circuit->apply =
        xcalloc(circuit->apply_capacity, sizeof(circuit->apply[0]));
    circuit->transformed = xcalloc(zdd_nodes, sizeof(circuit->transformed[0]));
    circuit->count = 1; /* Exact zero terminal. */
    circuit->nodes[0].level = QUERY_BITS;
}

static void circuit_free(QueryCircuit* circuit) {
    free(circuit->transformed);
    free(circuit->apply);
    free(circuit->terminal_slots);
    free(circuit->unique_slots);
    free(circuit->nodes);
    memset(circuit, 0, sizeof(*circuit));
}

static uint32_t circuit_append(QueryCircuit* circuit, QueryCircuitNode node) {
    if (circuit->aborted) return 0;
    if (circuit->count >= circuit->node_cap || circuit->count > UINT32_MAX) {
        circuit->aborted = 1;
        return 0;
    }
    uint32_t id = (uint32_t)circuit->count++;
    circuit->nodes[id] = node;
    return id;
}

static uint32_t circuit_terminal(QueryCircuit* circuit, uint64_t value) {
    if (!value) return 0;
    size_t slot = (size_t)mix64(value) & (circuit->slot_capacity - 1U);
    while (circuit->terminal_slots[slot]) {
        uint32_t id = circuit->terminal_slots[slot];
        if (circuit->nodes[id].value == value) return id;
        slot = (slot + 1U) & (circuit->slot_capacity - 1U);
    }
    uint32_t id = circuit_append(
        circuit, (QueryCircuitNode){.value = value, .level = QUERY_BITS});
    if (!circuit->aborted) {
        circuit->terminal_slots[slot] = id;
        circuit->terminal_count++;
    }
    return id;
}

static uint64_t circuit_node_hash(uint8_t level, uint32_t low,
                                  uint32_t high) {
    return mix64((uint64_t)level << 56 ^ (uint64_t)low << 28 ^ high);
}

static uint32_t circuit_node(QueryCircuit* circuit, uint8_t level,
                             uint32_t low, uint32_t high) {
    if (circuit->aborted) return 0;
    if (low == high) return low;
    size_t slot = (size_t)circuit_node_hash(level, low, high) &
                  (circuit->slot_capacity - 1U);
    while (circuit->unique_slots[slot]) {
        uint32_t id = circuit->unique_slots[slot];
        QueryCircuitNode* node = &circuit->nodes[id];
        if (node->level == level && node->low == low && node->high == high)
            return id;
        slot = (slot + 1U) & (circuit->slot_capacity - 1U);
    }
    uint32_t id = circuit_append(
        circuit, (QueryCircuitNode){.low = low, .high = high, .level = level});
    if (!circuit->aborted) circuit->unique_slots[slot] = id;
    return id;
}

static uint32_t circuit_add(QueryCircuit* circuit, uint32_t lhs,
                            uint32_t rhs) {
    if (circuit->aborted) return 0;
    if (!lhs) return rhs;
    if (!rhs) return lhs;
    if (lhs > rhs) {
        uint32_t temporary = lhs;
        lhs = rhs;
        rhs = temporary;
    }
    QueryCircuitNode left = circuit->nodes[lhs];
    QueryCircuitNode right = circuit->nodes[rhs];
    if (left.level == QUERY_BITS && right.level == QUERY_BITS) {
        if (UINT64_MAX - left.value < right.value) {
            fprintf(stderr, "query-circuit terminal overflow\n");
            exit(1);
        }
        return circuit_terminal(circuit, left.value + right.value);
    }

    uint64_t key = (uint64_t)lhs << 32 | rhs;
    size_t slot = (size_t)mix64(key) & (circuit->apply_capacity - 1U);
    while (circuit->apply[slot].key) {
        if (circuit->apply[slot].key == key)
            return circuit->apply[slot].value;
        slot = (slot + 1U) & (circuit->apply_capacity - 1U);
    }
    if (circuit->apply_count * 10U >= circuit->apply_capacity * 7U) {
        circuit->aborted = 1;
        return 0;
    }

    uint8_t level = left.level < right.level ? left.level : right.level;
    uint32_t left_low = left.level == level ? left.low : lhs;
    uint32_t left_high = left.level == level ? left.high : lhs;
    uint32_t right_low = right.level == level ? right.low : rhs;
    uint32_t right_high = right.level == level ? right.high : rhs;
    uint32_t low = circuit_add(circuit, left_low, right_low);
    uint32_t high = circuit_add(circuit, left_high, right_high);
    uint32_t result = circuit_node(circuit, level, low, high);
    if (!circuit->aborted) {
        circuit->apply[slot] = (QueryApplyEntry){.key = key, .value = result};
        circuit->apply_count++;
    }
    return result;
}

/* Transform support ZDD P=P0+xP1 into the ADD for Q(U).  For U_x=1,
 * Q=Q0; for U_x=0, Q=Q0+Q1. */
static uint32_t circuit_transform(QueryCircuit* circuit, const Oracle* zdd,
                                  uint32_t id) {
    if (!id || circuit->aborted) return 0;
    if (circuit->transformed[id]) return circuit->transformed[id];
    const OracleNode* source = &zdd->nodes[id];
    uint32_t result;
    if (!source->bit) {
        result = circuit_terminal(circuit, source->sum);
    } else {
        uint32_t without = circuit_transform(circuit, zdd, source->low);
        uint32_t with = circuit_transform(circuit, zdd, source->high);
        uint32_t allowed = circuit_add(circuit, without, with);
        unsigned bit = (unsigned)__builtin_ctzll(source->bit);
        uint8_t level = (uint8_t)(QUERY_BITS - 1U - bit);
        result = circuit_node(circuit, level, allowed, without);
    }
    if (!circuit->aborted) circuit->transformed[id] = result;
    return result;
}

static uint64_t circuit_query(const QueryCircuit* circuit, uint32_t root,
                              uint64_t forbidden, uint64_t* visits) {
    uint32_t id = root;
    while (circuit->nodes[id].level != QUERY_BITS) {
        const QueryCircuitNode* node = &circuit->nodes[id];
        uint64_t bit = UINT64_C(1) << (QUERY_BITS - 1U - node->level);
        id = forbidden & bit ? node->high : node->low;
        (*visits)++;
    }
    return circuit->nodes[id].value;
}

static uint64_t remap_mask(uint64_t mask, const VariableOrder* order) {
    uint64_t result = 0;
    for (unsigned position = 0; position < QUERY_BITS; position++)
        if (mask & (UINT64_C(1) << order->original_bit[position]))
            result |= UINT64_C(1) << position;
    return result;
}

static Distribution remap_distribution(const Distribution* source,
                                       const VariableOrder* order) {
    Distribution result = {.entries = xcalloc(source->count,
                                               sizeof(result.entries[0])),
                           .count = source->count};
    for (size_t i = 0; i < source->count; i++) {
        result.entries[i] = source->entries[i];
        result.entries[i].mask = remap_mask(source->entries[i].mask, order);
    }
    return result;
}

static uint64_t direct_query(const Distribution* indexed, uint64_t forbidden) {
    uint64_t result = 0;
    for (size_t i = 0; i < indexed->count; i++)
        if (!(indexed->entries[i].mask & forbidden))
            result += indexed->entries[i].weight;
    return result;
}

static int candidate_compare(const void* lhs_pointer,
                             const void* rhs_pointer) {
    const CircuitCandidate* lhs = lhs_pointer;
    const CircuitCandidate* rhs = rhs_pointer;
    if (lhs->support < rhs->support) return -1;
    if (lhs->support > rhs->support) return 1;
    if (lhs->prefix_index != rhs->prefix_index)
        return lhs->prefix_index - rhs->prefix_index;
    return lhs->complement - rhs->complement;
}

typedef struct {
    uint8_t bit;
    uint64_t frequency;
} BitFrequency;

static int frequency_descending(const void* lhs_pointer,
                                const void* rhs_pointer) {
    const BitFrequency* lhs = lhs_pointer;
    const BitFrequency* rhs = rhs_pointer;
    if (lhs->frequency > rhs->frequency) return -1;
    if (lhs->frequency < rhs->frequency) return 1;
    return (int)lhs->bit - (int)rhs->bit;
}

static int frequency_ascending(const void* lhs_pointer,
                               const void* rhs_pointer) {
    return -frequency_descending(lhs_pointer, rhs_pointer);
}

static void initialise_orders(const Distribution* distribution,
                              VariableOrder orders[4]) {
    static const uint8_t production_pairs[PAIRS] = {
        0, 1, 7, 2, 8, 13, 27, 3, 14, 18, 9, 24, 26, 25,
        23, 22, 4, 10, 15, 19, 5, 11, 16, 20, 6, 12, 17, 21};
    orders[0].name = "pair-major";
    orders[1].name = "production-pair";
    orders[2].name = "frequent-first";
    orders[3].name = "rare-first";
    for (unsigned pair = 0; pair < PAIRS; pair++) {
        orders[0].original_bit[2 * pair] = (uint8_t)pair;
        orders[0].original_bit[2 * pair + 1] = (uint8_t)(PAIRS + pair);
        orders[1].original_bit[2 * pair] = production_pairs[pair];
        orders[1].original_bit[2 * pair + 1] =
            (uint8_t)(PAIRS + production_pairs[pair]);
    }
    BitFrequency frequencies[QUERY_BITS];
    for (unsigned bit = 0; bit < QUERY_BITS; bit++)
        frequencies[bit] = (BitFrequency){.bit = (uint8_t)bit};
    for (size_t entry = 0; entry < distribution->count; entry++) {
        uint64_t mask = distribution->entries[entry].mask;
        while (mask) {
            unsigned bit = (unsigned)__builtin_ctzll(mask);
            frequencies[bit].frequency += distribution->entries[entry].weight;
            mask &= mask - 1U;
        }
    }
    qsort(frequencies, QUERY_BITS, sizeof(frequencies[0]),
          frequency_descending);
    for (unsigned position = 0; position < QUERY_BITS; position++)
        orders[2].original_bit[position] = frequencies[position].bit;
    qsort(frequencies, QUERY_BITS, sizeof(frequencies[0]), frequency_ascending);
    for (unsigned position = 0; position < QUERY_BITS; position++)
        orders[3].original_bit[position] = frequencies[position].bit;
}

static void benchmark_circuit(const CircuitCandidate* candidate,
                              const char* quantile, size_t node_cap) {
    uint8_t rows[ROWS];
    histogram_rows(g_prefixes[candidate->prefix_index].histogram, rows);
    Distribution indexed = build_distribution(rows, candidate->complement);
    Distribution queries = build_distribution(rows, !candidate->complement);
    VariableOrder orders[4];
    initialise_orders(&indexed, orders);
    printf("CIRCUIT_SOURCE quantile=%s prefix=%d component=%s support=%zu "
           "queries=%zu cap=%zu\n", quantile, candidate->prefix_index,
           candidate->complement ? "complement" : "selected", indexed.count,
           queries.count, node_cap);

    for (size_t order_index = 0; order_index < 4; order_index++) {
        const VariableOrder* order = &orders[order_index];
        Distribution ordered_indexed = remap_distribution(&indexed, order);
        Distribution ordered_queries = remap_distribution(&queries, order);
        uint32_t zdd_root = 0;
        double zdd_start = seconds_now();
        Oracle zdd = oracle_build(&ordered_indexed, 0, &zdd_root);
        double zdd_seconds = seconds_now() - zdd_start;
        QueryCircuit circuit;
        circuit_init(&circuit, node_cap, zdd.count);
        double transform_start = seconds_now();
        uint32_t root = circuit_transform(&circuit, &zdd, zdd_root);
        double transform_seconds = seconds_now() - transform_start;
        if (circuit.aborted) {
            printf("CIRCUIT_RESULT quantile=%s order=%s status=CAP support=%zu "
                   "zdd_nodes=%zu add_nodes=%zu terminals=%zu "
                   "apply_states=%zu zdd_seconds=%.6f transform_seconds=%.6f\n",
                   quantile, order->name, indexed.count, zdd.count,
                   circuit.count, circuit.terminal_count, circuit.apply_count,
                   zdd_seconds, transform_seconds);
            circuit_free(&circuit);
            oracle_free(&zdd);
            free_distribution(&ordered_queries);
            free_distribution(&ordered_indexed);
            continue;
        }

        uint64_t sampled_visits = 0;
        size_t direct_checks = queries.count < 64 ? queries.count : 64;
        for (size_t check = 0; check < direct_checks; check++) {
            size_t query_index =
                (size_t)mix64(UINT64_C(0x9e3779b97f4a7c15) * (check + 1U)) %
                queries.count;
            uint64_t forbidden = queries.entries[query_index].mask;
            uint64_t ordered =
                oracle_order_mask(remap_mask(forbidden, order), 0);
            uint64_t actual = circuit_query(&circuit, root, ordered,
                                            &sampled_visits);
            uint64_t expected = direct_query(&indexed, forbidden);
            if (actual != expected) {
                fprintf(stderr,
                        "query-circuit mismatch prefix=%d order=%s "
                        "actual=%" PRIu64 " expected=%" PRIu64 "\n",
                        candidate->prefix_index, order->name, actual, expected);
                exit(1);
            }
        }

        U128 circuit_join = 0;
        uint64_t query_visits = 0;
        double query_start = seconds_now();
        for (size_t query = 0; query < queries.count; query++) {
            uint64_t ordered = oracle_order_mask(
                remap_mask(queries.entries[query].mask, order), 0);
            uint64_t value = circuit_query(&circuit, root, ordered,
                                           &query_visits);
            circuit_join += (U128)queries.entries[query].weight * value;
        }
        double query_seconds = seconds_now() - query_start;
        if (circuit_join > UINT64_MAX) {
            fprintf(stderr, "query-circuit join overflow\n");
            exit(1);
        }
        zdd.visits = 0;
        double oracle_start = seconds_now();
        uint64_t oracle_join =
            oracle_query_built(&ordered_queries, &zdd, zdd_root);
        double oracle_seconds = seconds_now() - oracle_start;
        if ((uint64_t)circuit_join != oracle_join) {
            fprintf(stderr, "query-circuit aggregate mismatch\n");
            exit(1);
        }

        printf("CIRCUIT_RESULT quantile=%s order=%s status=OK support=%zu "
               "queries=%zu zdd_nodes=%zu add_nodes=%zu terminals=%zu "
               "apply_states=%zu persistent_node_bytes=%zu "
               "zdd_seconds=%.6f transform_seconds=%.6f query_seconds=%.6f "
               "oracle_seconds=%.6f avg_depth=%.3f oracle_visits=%" PRIu64
               " direct_checks=%zu value=%" PRIu64 "\n",
               quantile, order->name, indexed.count, queries.count, zdd.count,
               circuit.count, circuit.terminal_count, circuit.apply_count,
               circuit.count * sizeof(circuit.nodes[0]), zdd_seconds,
               transform_seconds, query_seconds, oracle_seconds,
               queries.count ? (double)query_visits / queries.count : 0,
               zdd.visits, direct_checks, oracle_join);
        circuit_free(&circuit);
        oracle_free(&zdd);
        free_distribution(&ordered_queries);
        free_distribution(&ordered_indexed);
    }
    free_distribution(&queries);
    free_distribution(&indexed);
}

int main(int argc, char** argv) {
    size_t sample_prefixes = argc > 1 ? strtoull(argv[1], NULL, 10) : 256;
    size_t node_cap = argc > 2 ? strtoull(argv[2], NULL, 10) : 1000000;
    if (!sample_prefixes || node_cap < 1024) {
        fprintf(stderr, "Usage: %s [PREFIX_SAMPLES] [ADD_NODE_CAP]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    g_factorial[0] = 1;
    for (int i = 1; i <= ROWS; i++)
        g_factorial[i] = g_factorial[i - 1] * (uint64_t)i;
    int pair_index = 0;
    for (int first = 0; first < ROWS; first++)
        for (int second = first + 1; second < ROWS; second++)
            g_pair_index[first][second] = pair_index++;
    generate_permutations(0, 0);
    enumerate_histograms(0, ROWS);
    if (g_prefix_count != PREFIX_ORBITS) {
        fprintf(stderr, "prefix orbit count failed: %d\n", g_prefix_count);
        return 1;
    }
    if (sample_prefixes > (size_t)g_prefix_count)
        sample_prefixes = (size_t)g_prefix_count;

    CircuitCandidate* candidates =
        xcalloc(2 * sample_prefixes, sizeof(candidates[0]));
    uint64_t state = UINT64_C(0x243f6a8885a308d3);
    double census_start = seconds_now();
    for (size_t sample = 0; sample < sample_prefixes; sample++) {
        state = mix64(state);
        int prefix = (int)(state % (uint64_t)g_prefix_count);
        uint8_t rows[ROWS];
        histogram_rows(g_prefixes[prefix].histogram, rows);
        Distribution selected = build_distribution(rows, 0);
        Distribution complement = build_distribution(rows, 1);
        candidates[2 * sample] = (CircuitCandidate){prefix, 0, selected.count};
        candidates[2 * sample + 1] =
            (CircuitCandidate){prefix, 1, complement.count};
        free_distribution(&selected);
        free_distribution(&complement);
        if ((sample + 1U) % 32U == 0 || sample + 1U == sample_prefixes)
            printf("CIRCUIT_CENSUS progress=%zu/%zu seconds=%.3f\n", sample + 1U,
                   sample_prefixes, seconds_now() - census_start);
    }
    qsort(candidates, 2 * sample_prefixes, sizeof(candidates[0]),
          candidate_compare);
    size_t total = 2 * sample_prefixes;
    size_t positions[5] = {total / 10U, total / 2U, total * 9U / 10U,
                           total * 99U / 100U, total - 1U};
    const char* labels[5] = {"p10", "median", "p90", "p99", "max"};
    printf("CIRCUIT_CENSUS sampled_prefixes=%zu components=%zu min=%zu "
           "median=%zu p90=%zu p99=%zu max=%zu seconds=%.3f\n",
           sample_prefixes, total, candidates[0].support,
           candidates[positions[1]].support, candidates[positions[2]].support,
           candidates[positions[3]].support, candidates[positions[4]].support,
           seconds_now() - census_start);
    for (size_t quantile = 0; quantile < 5; quantile++)
        benchmark_circuit(&candidates[positions[quantile]], labels[quantile],
                          node_cap);
    free(candidates);
    return 0;
}
