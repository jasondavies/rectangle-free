// Exact deletion-contraction gate for the dense-first 9x9 decomposition.
//
// Fix one C4-free direct-colour class A.  The cells outside A are vertices,
// and every rectangle contained in the complement is a hyperedge.  A weak
// proper three-colouring of this hypergraph is exactly a completion by the
// other three direct colours.  This probe evaluates that count modulo a
// prime without enumerating the second colour class.

// This is intentionally a feasibility probe rather than a production
// solver.  It uses labelled hypergraph states, exact edge subsumption,
// connected-component factorisation, and memoised deletion-contraction.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Mask = uint64_t;
using Edges = std::vector<Mask>;

constexpr uint64_t kDefaultPrime = 2305843009213693951ULL; // 2^61 - 1

uint64_t parse_u64(std::string_view text, int base = 10) {
    uint64_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid integer: " + std::string(text));
    }
    return value;
}

std::vector<uint16_t> parse_rows(std::string_view text, unsigned columns) {
    std::vector<uint16_t> rows;
    while (!text.empty()) {
        size_t comma = text.find(',');
        std::string_view field = text.substr(0, comma);
        rows.push_back(static_cast<uint16_t>(parse_u64(field, 16)));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    const uint16_t allowed = static_cast<uint16_t>((uint32_t{1} << columns) - 1);
    for (uint16_t row : rows) {
        if (row & ~allowed) throw std::runtime_error("first row exceeds column count");
    }
    return rows;
}

uint64_t sub_mod(uint64_t left, uint64_t right, uint64_t prime) {
    return left >= right ? left - right : prime - (right - left);
}

uint64_t mul_mod(uint64_t left, uint64_t right, uint64_t prime) {
    return static_cast<uint64_t>(static_cast<__uint128_t>(left) * right % prime);
}

uint64_t pow_mod(uint64_t base, uint64_t exponent, uint64_t prime) {
    uint64_t output = 1;
    while (exponent) {
        if (exponent & 1) output = mul_mod(output, base, prime);
        base = mul_mod(base, base, prime);
        exponent >>= 1;
    }
    return output;
}

struct EdgeVectorHash {
    size_t operator()(const Edges& edges) const noexcept {
        uint64_t hash = 0x9e3779b97f4a7c15ULL ^ edges.size();
        for (uint64_t edge : edges) {
            edge ^= edge >> 30;
            edge *= 0xbf58476d1ce4e5b9ULL;
            edge ^= edge >> 27;
            edge *= 0x94d049bb133111ebULL;
            edge ^= edge >> 31;
            hash ^= edge + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        return static_cast<size_t>(hash);
    }
};

struct Normalized {
    unsigned vertices = 0;
    unsigned isolated = 0;
    bool impossible = false;
    Edges edges;
};

struct Statistics {
    uint64_t calls = 0;
    uint64_t memo_hits = 0;
    uint64_t terminals = 0;
    uint64_t graph_only_states = 0;
    uint64_t component_splits = 0;
    uint64_t articulation_splits = 0;
    uint64_t removed_supersets = 0;
    uint64_t maximum_edges = 0;
    uint64_t maximum_vertices = 0;
};

class Solver {
public:
    Solver(uint64_t prime, uint64_t state_cap, double time_cap, bool overlap_heuristic = true)
        : prime_(prime), state_cap_(state_cap), time_cap_(time_cap),
          started_(std::chrono::steady_clock::now()), overlap_heuristic_(overlap_heuristic) {
        memo_.reserve(static_cast<size_t>(std::min<uint64_t>(state_cap, 4'000'000)));
    }

    uint64_t solve(unsigned vertices, Edges edges) {
        Normalized state = normalize(vertices, std::move(edges));
        if (state.impossible) return 0;
        uint64_t factor = pow_mod(3, state.isolated, prime_);
        if (state.edges.empty()) {
            ++stats_.terminals;
            return mul_mod(factor, pow_mod(3, state.vertices, prime_), prime_);
        }
        return mul_mod(factor, solve_core(state.vertices, state.edges), prime_);
    }

    const Statistics& statistics() const { return stats_; }
    size_t memo_size() const { return memo_.size(); }
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    }

private:
    uint64_t solve_sparse(Edges edges) {
        Mask used = 0;
        for (Mask edge : edges) used |= edge;
        std::array<unsigned, 64> image{};
        unsigned next = 0;
        for (unsigned bit = 0; bit < 64; ++bit) {
            if (used & (Mask{1} << bit)) image[bit] = next++;
        }
        for (Mask& edge : edges) {
            Mask mapped = 0;
            while (edge) {
                unsigned source = static_cast<unsigned>(__builtin_ctzll(edge));
                mapped |= Mask{1} << image[source];
                edge &= edge - 1;
            }
            edge = mapped;
        }
        return solve(next, std::move(edges));
    }

    Normalized normalize(unsigned vertices, Edges edges) {
        Normalized output;
        std::sort(edges.begin(), edges.end(), [](Mask left, Mask right) {
            unsigned left_size = static_cast<unsigned>(__builtin_popcountll(left));
            unsigned right_size = static_cast<unsigned>(__builtin_popcountll(right));
            return left_size != right_size ? left_size < right_size : left < right;
        });
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        Edges minimal;
        minimal.reserve(edges.size());
        Mask used = 0;
        for (Mask edge : edges) {
            if (__builtin_popcountll(edge) <= 1) {
                output.impossible = true;
                return output;
            }
            bool redundant = false;
            for (Mask smaller : minimal) {
                if ((smaller & edge) == smaller) {
                    redundant = true;
                    ++stats_.removed_supersets;
                    break;
                }
            }
            if (!redundant) {
                minimal.push_back(edge);
                used |= edge;
            }
        }

        const unsigned used_count = static_cast<unsigned>(__builtin_popcountll(used));
        output.isolated = vertices - used_count;
        output.vertices = used_count;
        if (!used_count) return output;

        std::array<unsigned, 64> image{};
        unsigned next = 0;
        for (unsigned bit = 0; bit < vertices; ++bit) {
            if (used & (Mask{1} << bit)) image[bit] = next++;
        }
        output.edges.reserve(minimal.size());
        for (Mask edge : minimal) {
            Mask mapped = 0;
            while (edge) {
                Mask bit = edge & -edge;
                unsigned source = static_cast<unsigned>(__builtin_ctzll(edge));
                mapped |= Mask{1} << image[source];
                edge -= bit;
            }
            output.edges.push_back(mapped);
        }
        std::sort(output.edges.begin(), output.edges.end(), [](Mask left, Mask right) {
            unsigned left_size = static_cast<unsigned>(__builtin_popcountll(left));
            unsigned right_size = static_cast<unsigned>(__builtin_popcountll(right));
            return left_size != right_size ? left_size < right_size : left < right;
        });
        return output;
    }

    std::vector<std::pair<unsigned, Edges>> components(unsigned vertices, const Edges& edges) {
        std::array<unsigned, 64> parent{};
        for (unsigned i = 0; i < vertices; ++i) parent[i] = i;
        auto root = [&](unsigned value) {
            while (parent[value] != value) {
                parent[value] = parent[parent[value]];
                value = parent[value];
            }
            return value;
        };
        for (Mask edge : edges) {
            unsigned first = static_cast<unsigned>(__builtin_ctzll(edge));
            edge &= edge - 1;
            while (edge) {
                unsigned other = static_cast<unsigned>(__builtin_ctzll(edge));
                unsigned left = root(first);
                unsigned right = root(other);
                if (left != right) parent[right] = left;
                edge &= edge - 1;
            }
        }
        std::array<int, 64> component_index{};
        component_index.fill(-1);
        std::vector<std::pair<unsigned, Edges>> output;
        for (Mask edge : edges) {
            unsigned representative = root(static_cast<unsigned>(__builtin_ctzll(edge)));
            int& index = component_index[representative];
            if (index < 0) {
                index = static_cast<int>(output.size());
                output.push_back({0, {}});
            }
            output[static_cast<size_t>(index)].second.push_back(edge);
        }
        if (output.size() <= 1) return output;

        for (auto& [count, component_edges] : output) {
            Mask used = 0;
            for (Mask edge : component_edges) used |= edge;
            count = static_cast<unsigned>(__builtin_popcountll(used));
        }
        return output;
    }

    struct ArticulationPieces {
        unsigned piece_count = 0;
        std::vector<Edges> edges;
    };

    ArticulationPieces articulation_pieces(unsigned vertices, const Edges& edges) {
        ArticulationPieces best;
        for (unsigned removed = 0; removed < vertices; ++removed) {
            std::array<unsigned, 64> parent{};
            for (unsigned i = 0; i < vertices; ++i) parent[i] = i;
            auto root = [&](unsigned value) {
                while (parent[value] != value) {
                    parent[value] = parent[parent[value]];
                    value = parent[value];
                }
                return value;
            };
            for (Mask edge : edges) {
                edge &= ~(Mask{1} << removed);
                if (!edge) continue;
                unsigned first = static_cast<unsigned>(__builtin_ctzll(edge));
                edge &= edge - 1;
                while (edge) {
                    unsigned other = static_cast<unsigned>(__builtin_ctzll(edge));
                    unsigned left = root(first);
                    unsigned right = root(other);
                    if (left != right) parent[right] = left;
                    edge &= edge - 1;
                }
            }
            std::array<int, 64> indices{};
            indices.fill(-1);
            unsigned count = 0;
            for (unsigned vertex = 0; vertex < vertices; ++vertex) {
                if (vertex == removed) continue;
                unsigned representative = root(vertex);
                if (indices[representative] < 0) indices[representative] = static_cast<int>(count++);
            }
            if (count <= best.piece_count || count <= 1) continue;

            std::vector<Edges> pieces(count);
            bool valid = true;
            for (Mask edge : edges) {
                Mask remaining = edge & ~(Mask{1} << removed);
                if (!remaining) {
                    valid = false;
                    break;
                }
                unsigned representative = root(static_cast<unsigned>(__builtin_ctzll(remaining)));
                int index = indices[representative];
                // Every non-articulation vertex of an edge lies in the same
                // component by construction of the primal connectivity graph.
                pieces[static_cast<size_t>(index)].push_back(edge);
            }
            if (valid) {
                best.piece_count = count;
                best.edges = std::move(pieces);
            }
        }
        return best;
    }

    Mask choose_edge(const Edges& edges) const {
        if (!overlap_heuristic_) return edges.front();
        Mask best = edges.front();
        uint64_t best_score = 0;
        for (Mask candidate : edges) {
            uint64_t overlap_score = 0;
            for (Mask other : edges) {
                if (other == candidate) continue;
                unsigned overlap = static_cast<unsigned>(__builtin_popcountll(candidate & other));
                overlap_score += uint64_t{overlap} * overlap;
            }
            uint64_t score = overlap_score * 8 + __builtin_popcountll(candidate);
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }
        return best;
    }

    std::pair<unsigned, Edges> contract_edge(unsigned vertices, const Edges& edges, Mask edge) {
        std::array<unsigned, 64> image{};
        unsigned next = 0;
        unsigned merged_image = std::numeric_limits<unsigned>::max();
        for (unsigned vertex = 0; vertex < vertices; ++vertex) {
            if (edge & (Mask{1} << vertex)) {
                if (merged_image == std::numeric_limits<unsigned>::max()) merged_image = next++;
                image[vertex] = merged_image;
            } else {
                image[vertex] = next++;
            }
        }
        Edges output;
        output.reserve(edges.size() - 1);
        for (Mask original : edges) {
            if (original == edge) continue;
            Mask mapped = 0;
            while (original) {
                unsigned source = static_cast<unsigned>(__builtin_ctzll(original));
                mapped |= Mask{1} << image[source];
                original &= original - 1;
            }
            output.push_back(mapped);
        }
        return {next, std::move(output)};
    }

    void check_limits() {
        if (memo_.size() >= state_cap_) throw std::runtime_error("state cap reached");
        if ((stats_.calls & 0xffff) == 0 && time_cap_ > 0 && seconds() >= time_cap_) {
            throw std::runtime_error("time cap reached");
        }
    }

    uint64_t solve_core(unsigned vertices, const Edges& edges) {
        ++stats_.calls;
        stats_.maximum_edges = std::max<uint64_t>(stats_.maximum_edges, edges.size());
        stats_.maximum_vertices = std::max<uint64_t>(stats_.maximum_vertices, vertices);
        auto found = memo_.find(edges);
        if (found != memo_.end()) {
            ++stats_.memo_hits;
            return found->second;
        }
        check_limits();
        if (std::all_of(edges.begin(), edges.end(), [](Mask edge) {
                return __builtin_popcountll(edge) == 2;
            })) {
            ++stats_.graph_only_states;
        }

        auto pieces = components(vertices, edges);
        if (pieces.size() > 1) {
            ++stats_.component_splits;
            uint64_t result = 1;
            for (auto& [piece_vertices, piece_edges] : pieces) {
                (void)piece_vertices;
                result = mul_mod(result, solve_sparse(std::move(piece_edges)), prime_);
            }
            memo_.emplace(edges, result);
            return result;
        }

        ArticulationPieces articulation = articulation_pieces(vertices, edges);
        if (articulation.piece_count > 1) {
            ++stats_.articulation_splits;
            uint64_t result = 1;
            for (Edges& piece : articulation.edges) {
                result = mul_mod(result, solve_sparse(std::move(piece)), prime_);
            }
            uint64_t divisor = pow_mod(3, articulation.piece_count - 1, prime_);
            result = mul_mod(result, pow_mod(divisor, prime_ - 2, prime_), prime_);
            memo_.emplace(edges, result);
            return result;
        }

        Mask selected = choose_edge(edges);
        Edges deleted;
        deleted.reserve(edges.size() - 1);
        for (Mask edge : edges) if (edge != selected) deleted.push_back(edge);
        uint64_t without = solve(vertices, std::move(deleted));

        auto [contracted_vertices, contracted_edges] = contract_edge(vertices, edges, selected);
        uint64_t with_monochromatic = solve(contracted_vertices, std::move(contracted_edges));
        uint64_t result = sub_mod(without, with_monochromatic, prime_);
        memo_.emplace(edges, result);
        return result;
    }

    uint64_t prime_;
    uint64_t state_cap_;
    double time_cap_;
    std::chrono::steady_clock::time_point started_;
    bool overlap_heuristic_;
    Statistics stats_;
    std::unordered_map<Edges, uint64_t, EdgeVectorHash> memo_;
};

std::pair<unsigned, Edges> rectangle_hypergraph(
    unsigned rows,
    unsigned columns,
    const std::vector<uint16_t>& first_rows
) {
    std::array<std::array<int, 9>, 9> index{};
    for (auto& row : index) row.fill(-1);
    unsigned vertices = 0;
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned column = 0; column < columns; ++column) {
            if (!(first_rows[row] & (uint16_t{1} << column))) {
                index[row][column] = static_cast<int>(vertices++);
            }
        }
    }
    if (vertices > 64) throw std::runtime_error("probe supports at most 64 residual cells");
    Edges edges;
    for (unsigned first_row = 0; first_row < rows; ++first_row) {
        for (unsigned second_row = first_row + 1; second_row < rows; ++second_row) {
            for (unsigned first_column = 0; first_column < columns; ++first_column) {
                for (unsigned second_column = first_column + 1; second_column < columns; ++second_column) {
                    std::array<int, 4> corners = {
                        index[first_row][first_column], index[first_row][second_column],
                        index[second_row][first_column], index[second_row][second_column],
                    };
                    if (std::all_of(corners.begin(), corners.end(), [](int value) { return value >= 0; })) {
                        Mask edge = 0;
                        for (int corner : corners) edge |= Mask{1} << corner;
                        edges.push_back(edge);
                    }
                }
            }
        }
    }
    return {vertices, std::move(edges)};
}

uint64_t brute_force(unsigned vertices, const Edges& edges, uint64_t prime) {
    uint64_t assignments = 1;
    for (unsigned vertex = 0; vertex < vertices; ++vertex) assignments *= 3;
    uint64_t answer = 0;
    std::vector<unsigned> colours(vertices);
    for (uint64_t code = 0; code < assignments; ++code) {
        uint64_t value = code;
        for (unsigned vertex = 0; vertex < vertices; ++vertex) {
            colours[vertex] = static_cast<unsigned>(value % 3);
            value /= 3;
        }
        bool valid = true;
        for (Mask edge : edges) {
            unsigned first = static_cast<unsigned>(__builtin_ctzll(edge));
            unsigned colour = colours[first];
            edge &= edge - 1;
            bool monochromatic = true;
            while (edge) {
                unsigned vertex = static_cast<unsigned>(__builtin_ctzll(edge));
                if (colours[vertex] != colour) {
                    monochromatic = false;
                    break;
                }
                edge &= edge - 1;
            }
            if (monochromatic) {
                valid = false;
                break;
            }
        }
        answer += valid;
    }
    return answer % prime;
}

void self_test() {
    const std::vector<std::pair<unsigned, Edges>> structural_cases = {
        {4, {0b0011, 0b1100}},
        {5, {0b00011, 0b00110, 0b01100, 0b11000}},
        {5, {0b00111, 0b11100}},
        {6, {0b001111, 0b111100, 0b110011}},
    };
    for (const auto& [vertices, edges] : structural_cases) {
        Solver solver(kDefaultPrime, 1'000'000, 0);
        uint64_t actual = solver.solve(vertices, edges);
        uint64_t expected = brute_force(vertices, edges, kDefaultPrime);
        if (actual != expected) throw std::runtime_error("structural self-test mismatch");
    }
    for (unsigned columns = 2; columns <= 3; ++columns) {
        const unsigned rows = 2;
        const uint64_t masks = uint64_t{1} << (rows * columns);
        for (uint64_t first = 0; first < masks; ++first) {
            std::vector<uint16_t> first_rows(rows);
            for (unsigned row = 0; row < rows; ++row) {
                first_rows[row] = static_cast<uint16_t>(
                    (first >> (row * columns)) & ((uint64_t{1} << columns) - 1));
            }
            auto [vertices, edges] = rectangle_hypergraph(rows, columns, first_rows);
            Solver solver(kDefaultPrime, 1'000'000, 0);
            uint64_t actual = solver.solve(vertices, edges);
            uint64_t expected = brute_force(vertices, edges, kDefaultPrime);
            if (actual != expected) {
                throw std::runtime_error("self-test mismatch");
            }
        }
    }
    std::cout << "self_test=pass\n";
}

} // namespace

int main(int argc, char** argv) try {
    unsigned rows = 9;
    unsigned columns = 9;
    uint64_t prime = kDefaultPrime;
    uint64_t state_cap = 2'000'000;
    double time_cap = 60;
    bool run_self_test = false;
    bool overlap_heuristic = true;
    std::vector<uint16_t> first_rows;

    for (int index = 1; index < argc; ++index) {
        std::string_view option = argv[index];
        if (option == "--self-test") {
            run_self_test = true;
            continue;
        }
        if (++index >= argc) throw std::runtime_error("missing value for " + std::string(option));
        std::string_view value = argv[index];
        if (option == "--rows") rows = static_cast<unsigned>(parse_u64(value));
        else if (option == "--columns") columns = static_cast<unsigned>(parse_u64(value));
        else if (option == "--first") first_rows = parse_rows(value, columns);
        else if (option == "--prime") prime = parse_u64(value);
        else if (option == "--state-cap") state_cap = parse_u64(value);
        else if (option == "--time-cap") time_cap = std::stod(std::string(value));
        else if (option == "--heuristic") {
            if (value == "overlap") overlap_heuristic = true;
            else if (value == "first") overlap_heuristic = false;
            else throw std::runtime_error("--heuristic must be overlap or first");
        }
        else throw std::runtime_error("unknown option: " + std::string(option));
    }
    if (run_self_test) {
        self_test();
        return 0;
    }
    if (rows > 9 || columns > 9 || rows * columns > 81) {
        throw std::runtime_error("geometry exceeds probe bounds");
    }
    if (first_rows.size() != rows) throw std::runtime_error("--first requires one hexadecimal mask per row");
    auto [vertices, edges] = rectangle_hypergraph(rows, columns, first_rows);
    std::cout << "rows=" << rows << " columns=" << columns
              << " first_cells=" << rows * columns - vertices
              << " residual_vertices=" << vertices
              << " rectangle_edges=" << edges.size()
              << " prime=" << prime << '\n';

    Solver solver(prime, state_cap, time_cap, overlap_heuristic);
    try {
        uint64_t answer = solver.solve(vertices, std::move(edges));
        std::cout << "status=complete residue=" << answer;
    } catch (const std::runtime_error& error) {
        std::cout << "status=capped reason=\"" << error.what() << '"';
    }
    const Statistics& stats = solver.statistics();
    std::cout << " seconds=" << solver.seconds()
              << " calls=" << stats.calls
              << " unique_states=" << solver.memo_size()
              << " memo_hits=" << stats.memo_hits
              << " terminals=" << stats.terminals
              << " graph_only_states=" << stats.graph_only_states
              << " component_splits=" << stats.component_splits
              << " articulation_splits=" << stats.articulation_splits
              << " removed_supersets=" << stats.removed_supersets
              << " max_vertices=" << stats.maximum_vertices
              << " max_edges=" << stats.maximum_edges << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
