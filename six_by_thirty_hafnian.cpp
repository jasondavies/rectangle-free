// Exact finite-field hafnian solver for T_4(6,30).
//
// The 60 vertices are (colour,row-pair) tokens.  Two tokens are adjacent when
// their colours differ and their row pairs are disjoint, so the graph is
// K_4 x KG(6,2).  Its perfect matchings are the extremal 6x30 colourings up to
// column order and the two singleton-colour assignments in every column:
//
//   T_4(6,30) = 30! * 2^30 * haf(A).
//
// This implementation evaluates the exact Glynn power-trace formula modulo an
// odd prime.  One sign is fixed, leaving 2^29 independent terms.  Work ranges
// are self-contained and may be distributed arbitrarily; the companion Python
// reducer verifies range coverage and reconstructs the integer with CRT.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "sha256.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr unsigned TARGET_ROWS = 6;
constexpr unsigned TARGET_COLOURS = 4;
constexpr unsigned TARGET_PAIRS = 15;
constexpr unsigned TARGET_VERTICES = 60;
constexpr unsigned TARGET_HALF = 30;
constexpr uint64_t TARGET_TERMS = UINT64_C(1) << (TARGET_HALF - 1);
constexpr const char* ALGORITHM = "glynn-trace-hessenberg-v1";

struct Mod {
    uint32_t p;
    uint64_t reciprocal;

    explicit Mod(uint32_t prime)
        : p(prime), reciprocal(uint64_t((static_cast<unsigned __int128>(1) << 64) / prime)) {}

    uint32_t add(uint32_t a, uint32_t b) const {
        uint32_t value = a + b;
        if (value >= p || value < a) value -= p;
        return value;
    }
    uint32_t sub(uint32_t a, uint32_t b) const {
        return a >= b ? a - b : uint32_t(uint64_t(a) + p - b);
    }
    uint32_t neg(uint32_t a) const { return a ? p - a : 0; }
    uint32_t mul(uint32_t a, uint32_t b) const {
        uint64_t product = uint64_t(a) * b;
        uint64_t quotient = uint64_t(static_cast<unsigned __int128>(product) * reciprocal >> 64);
        uint64_t remainder = product - quotient * p;
        if (remainder >= p) remainder -= p;
        if (remainder >= p) remainder -= p;
        return uint32_t(remainder);
    }
    uint32_t power(uint32_t a, uint64_t exponent) const {
        uint32_t result = 1;
        while (exponent) {
            if (exponent & 1) result = mul(result, a);
            a = mul(a, a);
            exponent >>= 1;
        }
        return result;
    }
    uint32_t inverse(uint32_t a) const {
        if (!a) throw std::runtime_error("attempt to invert zero");
        return power(a, p - 2);
    }
};

uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t modulus) {
    return uint64_t((unsigned __int128)a * b % modulus);
}

uint64_t pow_mod_u64(uint64_t a, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    while (exponent) {
        if (exponent & 1) result = mul_mod_u64(result, a, modulus);
        a = mul_mod_u64(a, a, modulus);
        exponent >>= 1;
    }
    return result;
}

bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t small : {2U, 3U, 5U, 7U, 11U, 13U, 17U, 19U, 23U, 29U, 31U, 37U}) {
        if (n == small) return true;
        if (n % small == 0) return false;
    }
    uint32_t d = n - 1, s = 0;
    while (!(d & 1)) { d >>= 1; ++s; }
    // These bases are deterministic for all 32-bit integers.
    for (uint32_t base : {2U, 3U, 5U, 7U, 11U}) {
        if (base >= n) continue;
        uint64_t x = pow_mod_u64(base, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (uint32_t r = 1; r < s; ++r) {
            x = mul_mod_u64(x, x, n);
            if (x == n - 1) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}

struct Matrix {
    unsigned n = 0;
    std::array<uint32_t, TARGET_VERTICES * TARGET_VERTICES> a{};
    uint32_t& at(unsigned row, unsigned column) { return a[row * TARGET_VERTICES + column]; }
    uint32_t at(unsigned row, unsigned column) const { return a[row * TARGET_VERTICES + column]; }
};

struct Graph {
    unsigned rows = 0, colours = 0, pair_count = 0, vertices = 0, half = 0;
    std::vector<std::pair<unsigned, unsigned>> pairs;
    std::vector<uint8_t> adjacency;
    std::vector<unsigned> order;
};

Graph build_graph(unsigned rows, unsigned colours) {
    Graph graph;
    graph.rows = rows;
    graph.colours = colours;
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = i + 1; j < rows; ++j) graph.pairs.push_back({i, j});
    graph.pair_count = unsigned(graph.pairs.size());
    graph.vertices = colours * graph.pair_count;
    if (graph.vertices & 1 || graph.vertices > TARGET_VERTICES)
        throw std::runtime_error("unsupported graph order");
    graph.half = graph.vertices / 2;
    graph.adjacency.assign(size_t(graph.vertices) * graph.vertices, 0);
    auto disjoint = [&](unsigned p, unsigned q) {
        auto [a, b] = graph.pairs[p];
        auto [c, d] = graph.pairs[q];
        return a != c && a != d && b != c && b != d;
    };
    for (unsigned c = 0; c < colours; ++c)
        for (unsigned d = 0; d < colours; ++d)
            for (unsigned p = 0; p < graph.pair_count; ++p)
                for (unsigned q = 0; q < graph.pair_count; ++q)
                    graph.adjacency[(c * graph.pair_count + p) * graph.vertices +
                                    d * graph.pair_count + q] = uint8_t(c != d && disjoint(p, q));

    // Find a supported permutation between two copies of KG(rows,2).
    std::vector<int> right_match(graph.pair_count, -1);
    auto augment = [&](auto&& self, unsigned left, std::vector<uint8_t>& seen) -> bool {
        for (unsigned right = 0; right < graph.pair_count; ++right) {
            if (!disjoint(left, right) || seen[right]) continue;
            seen[right] = 1;
            if (right_match[right] < 0 || self(self, unsigned(right_match[right]), seen)) {
                right_match[right] = int(left);
                return true;
            }
        }
        return false;
    };
    for (unsigned left = 0; left < graph.pair_count; ++left) {
        std::vector<uint8_t> seen(graph.pair_count);
        if (!augment(augment, left, seen)) throw std::runtime_error("KG matching construction failed");
    }
    std::vector<unsigned> mate(graph.pair_count);
    for (unsigned right = 0; right < graph.pair_count; ++right)
        mate[unsigned(right_match[right])] = right;

    // Pair colour lanes (0,1), (2,3), ... .  Small regression geometries use
    // an even number of colours; the production target has four.
    if (colours & 1) throw std::runtime_error("test graph requires an even colour count");
    std::vector<unsigned> first, second;
    for (unsigned c = 0; c < colours; c += 2) {
        for (unsigned p = 0; p < graph.pair_count; ++p) {
            first.push_back(c * graph.pair_count + p);
            second.push_back((c + 1) * graph.pair_count + mate[p]);
        }
    }
    graph.order = first;
    graph.order.insert(graph.order.end(), second.begin(), second.end());
    if (graph.order.size() != graph.vertices) throw std::runtime_error("pairing order size mismatch");
    std::vector<unsigned> sorted = graph.order;
    std::sort(sorted.begin(), sorted.end());
    for (unsigned i = 0; i < graph.vertices; ++i)
        if (sorted[i] != i) throw std::runtime_error("pairing order is not a permutation");
    for (unsigned i = 0; i < graph.half; ++i)
        if (!graph.adjacency[graph.order[i] * graph.vertices + graph.order[i + graph.half]])
            throw std::runtime_error("reference pairing is not a graph matching");
    return graph;
}

std::string graph_sha256(const Graph& graph) {
    Sha256 hash;
    const std::string header = "token-graph-Kq-cross-KG-r-2-v1\n";
    hash.update(header);
    const uint8_t dimensions[] = {
        uint8_t(graph.rows), uint8_t(graph.colours), uint8_t(graph.pair_count),
        uint8_t(graph.vertices), uint8_t(graph.half)};
    hash.update(dimensions, sizeof(dimensions));
    hash.update(graph.adjacency.data(), graph.adjacency.size());
    for (unsigned vertex : graph.order) {
        uint8_t encoded = uint8_t(vertex);
        hash.update(&encoded, 1);
    }
    return hash.finish_hex();
}

// Reduce A by similarity to upper Hessenberg form.  This preserves its
// characteristic polynomial and works over any field.
void upper_hessenberg(Matrix& matrix, const Mod& mod) {
    const unsigned n = matrix.n;
    for (unsigned column = 0; column + 2 < n; ++column) {
        unsigned pivot = column + 1;
        while (pivot < n && matrix.at(pivot, column) == 0) ++pivot;
        if (pivot == n) continue;
        if (pivot != column + 1) {
            for (unsigned j = 0; j < n; ++j)
                std::swap(matrix.at(pivot, j), matrix.at(column + 1, j));
            for (unsigned i = 0; i < n; ++i)
                std::swap(matrix.at(i, pivot), matrix.at(i, column + 1));
        }
        const uint32_t inverse = mod.inverse(matrix.at(column + 1, column));
        for (unsigned row = column + 2; row < n; ++row) {
            uint32_t below = matrix.at(row, column);
            if (!below) continue;
            uint32_t factor = mod.mul(below, inverse);
            for (unsigned j = column; j < n; ++j)
                matrix.at(row, j) = mod.sub(matrix.at(row, j), mod.mul(factor, matrix.at(column + 1, j)));
            for (unsigned i = 0; i < n; ++i)
                matrix.at(i, column + 1) = mod.add(matrix.at(i, column + 1), mod.mul(factor, matrix.at(i, row)));
        }
    }
}

// Return traces(A^1),...,traces(A^degree) from the characteristic polynomial
// of an upper-Hessenberg similarity transform (La Budde recurrence + Newton).
std::array<uint32_t, TARGET_HALF + 1>
power_traces(Matrix matrix, unsigned degree, const Mod& mod) {
    upper_hessenberg(matrix, mod);
    const unsigned n = matrix.n;
    std::array<std::array<uint32_t, TARGET_VERTICES + 1>, TARGET_VERTICES + 1> poly{};
    poly[0][0] = 1;
    for (unsigned size = 1; size <= n; ++size) {
        const unsigned diagonal = size - 1;
        for (unsigned d = 0; d < size; ++d) {
            poly[size][d + 1] = mod.add(poly[size][d + 1], poly[size - 1][d]);
            poly[size][d] = mod.sub(poly[size][d], mod.mul(matrix.at(diagonal, diagonal), poly[size - 1][d]));
        }
        uint32_t subdiagonal_product = 1;
        for (unsigned distance = 1; distance < size; ++distance) {
            unsigned sub_row = size - distance;
            subdiagonal_product = mod.mul(subdiagonal_product, matrix.at(sub_row, sub_row - 1));
            uint32_t factor = mod.mul(subdiagonal_product, matrix.at(size - distance - 1, size - 1));
            const auto& previous = poly[size - distance - 1];
            for (unsigned d = 0; d <= size - distance - 1; ++d)
                poly[size][d] = mod.sub(poly[size][d], mod.mul(factor, previous[d]));
        }
    }
    std::array<uint32_t, TARGET_HALF + 1> traces{};
    for (unsigned k = 1; k <= degree; ++k) {
        uint32_t value = 0;
        for (unsigned j = 1; j < k; ++j) {
            uint32_t characteristic = poly[n][n - j];
            value = mod.add(value, mod.mul(characteristic, traces[k - j]));
        }
        value = mod.add(value, mod.mul(uint32_t(k), poly[n][n - k]));
        traces[k] = mod.neg(value);
    }
    return traces;
}

uint32_t trace_term(const Graph& graph, uint64_t signs, const Mod& mod) {
    const unsigned n = graph.vertices;
    const unsigned half = graph.half;
    Matrix matrix;
    matrix.n = n;
    // The first sign is fixed to +1.  A global sign flip gives the same term
    // for even half-order, so this is equivalent to the conventional -1 fix.
    auto sign = [&](unsigned edge) -> uint32_t {
        if (!edge) return 1;
        return signs & (UINT64_C(1) << (edge - 1)) ? 1 : mod.p - 1;
    };
    unsigned negatives = 0;
    for (unsigned edge = 1; edge < half; ++edge)
        negatives += !(signs & (UINT64_C(1) << (edge - 1)));

    // M=A*X*D, where X exchanges the two endpoints of every reference pair
    // and D repeats that pair's sign on both exchanged columns.
    for (unsigned row = 0; row < n; ++row) {
        unsigned original_row = graph.order[row];
        for (unsigned column = 0; column < n; ++column) {
            unsigned paired = column < half ? column + half : column - half;
            unsigned original_column = graph.order[paired];
            if (graph.adjacency[original_row * n + original_column])
                matrix.at(row, column) = sign(column % half);
        }
    }
    auto traces = power_traces(matrix, half, mod);
    std::array<uint32_t, TARGET_HALF + 1> coefficient{};
    coefficient[0] = 1;
    const uint32_t inverse_two = mod.inverse(2);
    for (unsigned degree = 1; degree <= half; ++degree) {
        uint32_t sum = 0;
        for (unsigned k = 1; k <= degree; ++k)
            sum = mod.add(sum, mod.mul(mod.mul(traces[k], inverse_two), coefficient[degree - k]));
        coefficient[degree] = mod.mul(sum, mod.inverse(degree));
    }
    return negatives & 1 ? mod.neg(coefficient[half]) : coefficient[half];
}

uint64_t brute_perfect_matchings(const Graph& graph, uint64_t vertices) {
    if (!vertices) return 1;
    unsigned first = unsigned(__builtin_ctzll(vertices));
    uint64_t rest = vertices & ~(UINT64_C(1) << first);
    uint64_t result = 0;
    uint64_t candidates = rest;
    while (candidates) {
        unsigned second = unsigned(__builtin_ctzll(candidates));
        candidates &= candidates - 1;
        if (graph.adjacency[first * graph.vertices + second])
            result += brute_perfect_matchings(graph, rest & ~(UINT64_C(1) << second));
    }
    return result;
}

uint32_t hafnian_mod_complete(const Graph& graph, uint32_t prime) {
    Mod mod(prime);
    uint64_t terms = UINT64_C(1) << (graph.half - 1);
    uint32_t sum = 0;
    for (uint64_t index = 0; index < terms; ++index) sum = mod.add(sum, trace_term(graph, index, mod));
    return mod.mul(sum, mod.power(mod.inverse(2), graph.half - 1));
}

void self_test() {
    for (uint32_t prime : {1000000007U, 1000000009U}) {
        if (!is_prime_u32(prime)) throw std::runtime_error("self-test prime failure");
        uint64_t random_state = UINT64_C(0x6305eed123456789);
        auto random = [&]() {
            random_state ^= random_state << 7;
            random_state ^= random_state >> 9;
            return random_state;
        };
        for (unsigned vertices = 2; vertices <= 12; vertices += 2) {
            for (unsigned sample = 0; sample < 6; ++sample) {
                Graph arbitrary;
                arbitrary.vertices = vertices;
                arbitrary.half = vertices / 2;
                arbitrary.order.resize(vertices);
                std::iota(arbitrary.order.begin(), arbitrary.order.end(), 0);
                arbitrary.adjacency.assign(size_t(vertices) * vertices, 0);
                for (unsigned i = 0; i < vertices; ++i)
                    for (unsigned j = i + 1; j < vertices; ++j) {
                        uint8_t edge = sample == 0 ? 1 : uint8_t(random() & 1);
                        arbitrary.adjacency[i * vertices + j] = edge;
                        arbitrary.adjacency[j * vertices + i] = edge;
                    }
                uint64_t expected = brute_perfect_matchings(
                    arbitrary, (UINT64_C(1) << vertices) - 1);
                uint32_t actual = hafnian_mod_complete(arbitrary, prime);
                if (actual != expected % prime)
                    throw std::runtime_error("random hafnian/brute-force mismatch");
            }
        }
        Graph small = build_graph(4, 2); // 12 vertices, six sign-pair terms.
        uint64_t expected = brute_perfect_matchings(small, (UINT64_C(1) << small.vertices) - 1);
        uint32_t actual = hafnian_mod_complete(small, prime);
        if (actual != expected % prime)
            throw std::runtime_error("small hafnian/brute-force mismatch");
        std::printf("HAFNIAN_SELF_TEST rows=4 colours=2 vertices=12 prime=%u expected=%" PRIu64 " actual=%u exact=OK\n",
                    prime, expected, actual);
        std::printf("HAFNIAN_RANDOM_SELF_TEST prime=%u orders=2,4,6,8,10,12 samples_per_order=6 exact=OK\n",
                    prime);
    }
    Graph target = build_graph(TARGET_ROWS, TARGET_COLOURS);
    uint64_t degree_sum = 0, edge_count = 0;
    for (unsigned i = 0; i < target.vertices; ++i) {
        unsigned degree = 0;
        for (unsigned j = 0; j < target.vertices; ++j) {
            if (target.adjacency[i * target.vertices + j] != target.adjacency[j * target.vertices + i])
                throw std::runtime_error("target adjacency is asymmetric");
            degree += target.adjacency[i * target.vertices + j];
        }
        if (degree != 18) throw std::runtime_error("target degree mismatch");
        degree_sum += degree;
    }
    edge_count = degree_sum / 2;
    if (edge_count != 540) throw std::runtime_error("target edge count mismatch");
    std::string digest = graph_sha256(target);
    std::printf("HAFNIAN_GRAPH rows=6 colours=4 vertices=60 degree=18 edges=540 reference_pairs=30 terms=%" PRIu64 " graph_sha256=%s exact=OK\n",
                TARGET_TERMS, digest.c_str());
}

struct Options {
    uint32_t prime = 1000000007U;
    uint64_t begin = 0, end = 0;
    unsigned threads = 1;
    uint64_t progress = 1000;
    std::string output;
    bool run = false, self_test_only = false;
};

uint64_t parse_u64(const char* text) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (!end || *end) throw std::runtime_error(std::string("invalid integer: ") + text);
    return uint64_t(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--self-test") options.self_test_only = true;
        else if (arg == "--run") options.run = true;
        else if (arg == "--prime" && i + 1 < argc) options.prime = uint32_t(parse_u64(argv[++i]));
        else if (arg == "--begin" && i + 1 < argc) options.begin = parse_u64(argv[++i]);
        else if (arg == "--end" && i + 1 < argc) options.end = parse_u64(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) options.threads = unsigned(parse_u64(argv[++i]));
        else if (arg == "--progress" && i + 1 < argc) options.progress = parse_u64(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) options.output = argv[++i];
        else throw std::runtime_error(
            "usage: six_by_thirty_hafnian [--self-test] [--run --prime P --begin B --end E --threads N --output FILE]");
    }
    return options;
}

void write_atomic(const std::string& path, const std::string& contents) {
    if (path.empty()) return;
    std::filesystem::path target(path);
    if (!target.parent_path().empty()) std::filesystem::create_directories(target.parent_path());
    std::filesystem::path temporary = target;
    temporary += ".tmp." + std::to_string(::getpid());
    {
        std::ofstream output(temporary);
        if (!output) throw std::runtime_error("cannot open temporary result");
        output << contents;
        output.flush();
        if (!output) throw std::runtime_error("cannot write temporary result");
    }
    std::filesystem::rename(temporary, target);
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        self_test();
        if (options.self_test_only || !options.run) return 0;
        if (!is_prime_u32(options.prime) || options.prime <= TARGET_HALF ||
            options.prime > INT32_MAX)
            throw std::runtime_error("modulus must be a prime in (30,2^31)");
        if (!options.threads) throw std::runtime_error("threads must be positive");
        if (!options.end) options.end = TARGET_TERMS;
        if (options.begin >= options.end || options.end > TARGET_TERMS)
            throw std::runtime_error("invalid sign-term range");
#ifndef _OPENMP
        if (options.threads != 1) throw std::runtime_error("binary lacks OpenMP support");
#else
        omp_set_dynamic(0);
        omp_set_num_threads(int(options.threads));
#endif
        Graph graph = build_graph(TARGET_ROWS, TARGET_COLOURS);
        const std::string graph_digest = graph_sha256(graph);
        const std::string binary_digest = sha256_file(argv[0]);
        Mod mod(options.prime);
        const auto started = Clock::now();
        std::atomic<uint64_t> completed{0};
        std::vector<uint64_t> thread_sums(options.threads);
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            unsigned thread = 0;
#ifdef _OPENMP
            thread = unsigned(omp_get_thread_num());
#pragma omp for schedule(dynamic, 1)
#endif
            for (uint64_t index = options.begin; index < options.end; ++index) {
                thread_sums[thread] += trace_term(graph, index, mod);
                if (thread_sums[thread] >= (UINT64_C(1) << 62)) thread_sums[thread] %= options.prime;
                uint64_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (options.progress && done % options.progress == 0) {
#ifdef _OPENMP
#pragma omp critical(hafnian_progress)
#endif
                    {
                        double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
                        std::printf("HAFNIAN_PROGRESS prime=%u begin=%" PRIu64 " end=%" PRIu64 " completed=%" PRIu64 " elapsed=%.3f terms_per_second=%.3f\n",
                                    options.prime, options.begin, options.end, done, elapsed, done / elapsed);
                        std::fflush(stdout);
                    }
                }
            }
        }
        uint32_t partial = 0;
        for (uint64_t thread_sum : thread_sums)
            partial = mod.add(partial, uint32_t(thread_sum % options.prime));
        double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        char buffer[4096];
        int length = std::snprintf(buffer, sizeof(buffer),
            "format six-by-thirty-hafnian-v1\n"
            "algorithm %s\n"
            "rows 6\ncolours 4\nvertices 60\nedges 540\n"
            "graph_sha256 %s\nsolver_binary_sha256 %s\n"
            "prime %u\nbegin %" PRIu64 "\nend %" PRIu64 "\ntotal_terms %" PRIu64 "\n"
            "partial_glynn_sum %u\nthreads %u\nelapsed_seconds %.9f\nstatus complete\n",
            ALGORITHM, graph_digest.c_str(), binary_digest.c_str(),
            options.prime, options.begin, options.end, TARGET_TERMS,
            partial, options.threads, elapsed);
        if (length < 0 || size_t(length) >= sizeof(buffer)) throw std::runtime_error("result formatting failed");
        std::string result(buffer, size_t(length));
        result += "result_payload_sha256 " + sha256_string(result) + "\n";
        write_atomic(options.output, result);
        std::fwrite(result.data(), 1, result.size(), stdout);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}
