// Exact closure-lattice census for monochromatic rectangle constraints.
//
// A state is a partition of the grid cells.  Joining the four cells of a
// rectangle gives another state.  The reachable partitions form the
// intersection/closure lattice underlying rectangle inclusion-exclusion.
// This probe can enumerate labelled states, or quotient them exactly under
// row permutations, column permutations, transpose (for square grids), and
// unlabeled partition blocks using nauty.

#include <nauty/nauty.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Partition = std::string;
constexpr uint64_t kPrime = 2305843009213693951ULL;

uint64_t parse_u64(std::string_view text) {
    uint64_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid integer: " + std::string(text));
    }
    return value;
}

uint64_t mul_mod(uint64_t left, uint64_t right) {
    return static_cast<uint64_t>(static_cast<__uint128_t>(left) * right % kPrime);
}

uint64_t pow_mod(uint64_t base, uint64_t exponent) {
    uint64_t output = 1;
    while (exponent) {
        if (exponent & 1) output = mul_mod(output, base);
        base = mul_mod(base, base);
        exponent >>= 1;
    }
    return output;
}

Partition normalize(const Partition& input) {
    std::array<unsigned char, 256> image{};
    image.fill(0xff);
    unsigned char next = 0;
    Partition output(input.size(), '\0');
    for (size_t index = 0; index < input.size(); ++index) {
        unsigned char value = static_cast<unsigned char>(input[index]);
        if (image[value] == 0xff) image[value] = next++;
        output[index] = static_cast<char>(image[value]);
    }
    return output;
}

unsigned block_count(const Partition& partition) {
    unsigned maximum = 0;
    for (unsigned char value : partition) maximum = std::max(maximum, unsigned(value));
    return partition.empty() ? 0 : maximum + 1;
}

struct Rectangle {
    std::array<unsigned, 4> cells;
};

std::vector<Rectangle> rectangles(unsigned rows, unsigned columns) {
    std::vector<Rectangle> output;
    for (unsigned first_row = 0; first_row < rows; ++first_row) {
        for (unsigned second_row = first_row + 1; second_row < rows; ++second_row) {
            for (unsigned first_column = 0; first_column < columns; ++first_column) {
                for (unsigned second_column = first_column + 1;
                     second_column < columns; ++second_column) {
                    output.push_back({{
                        first_row * columns + first_column,
                        first_row * columns + second_column,
                        second_row * columns + first_column,
                        second_row * columns + second_column,
                    }});
                }
            }
        }
    }
    return output;
}

bool join_rectangle(const Partition& input, const Rectangle& rectangle, Partition& output) {
    std::array<unsigned char, 4> labels{};
    for (unsigned index = 0; index < 4; ++index) {
        labels[index] = static_cast<unsigned char>(input[rectangle.cells[index]]);
    }
    std::sort(labels.begin(), labels.end());
    auto unique_end = std::unique(labels.begin(), labels.end());
    if (unique_end == labels.begin() + 1) return false;
    unsigned char target = labels.front();
    output = input;
    for (char& raw : output) {
        unsigned char value = static_cast<unsigned char>(raw);
        if (std::find(labels.begin(), unique_end, value) != unique_end) {
            raw = static_cast<char>(target);
        }
    }
    output = normalize(output);
    return true;
}

class NautyCanonicalizer {
public:
    NautyCanonicalizer(unsigned rows, unsigned columns)
        : rows_(rows), columns_(columns), cells_(rows * columns) {
        nauty_check(WORDSIZE, 2, 256, NAUTYVERSIONID);
    }

    Partition operator()(const Partition& partition) {
        return canonical_oriented(partition, rows_, columns_);
    }

    uint64_t calls() const { return calls_; }

private:
    Partition canonical_oriented(const Partition& partition, unsigned rows, unsigned columns) {
        ++calls_;
        const unsigned blocks = block_count(partition);
        const bool allow_transpose = rows == columns;
        const unsigned row_start = 0;
        const unsigned column_start = rows;
        const unsigned cell_start = rows + columns;
        const unsigned block_start = cell_start + rows * columns;
        const int vertices = static_cast<int>(block_start + blocks);
        const int words = SETWORDSNEEDED(vertices);
        if (vertices > 256) throw std::runtime_error("nauty graph exceeds static probe bound");

        std::vector<graph> graph_data(static_cast<size_t>(words) * vertices);
        std::vector<graph> canonical_data(static_cast<size_t>(words) * vertices);
        EMPTYGRAPH(graph_data.data(), words, vertices);
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned column = 0; column < columns; ++column) {
                unsigned cell = row * columns + column;
                unsigned cell_vertex = cell_start + cell;
                unsigned block = static_cast<unsigned char>(partition[cell]);
                ADDONEEDGE(graph_data.data(), row_start + row, cell_vertex, words);
                ADDONEEDGE(graph_data.data(), column_start + column, cell_vertex, words);
                ADDONEEDGE(graph_data.data(), block_start + block, cell_vertex, words);
            }
        }

        std::vector<int> labels(vertices);
        std::vector<int> partition_cells(vertices, 1);
        std::vector<int> orbits(vertices);
        for (int vertex = 0; vertex < vertices; ++vertex) labels[vertex] = vertex;
        if (!allow_transpose) partition_cells[rows - 1] = 0;
        partition_cells[rows + columns - 1] = 0;
        partition_cells[cell_start + rows * columns - 1] = 0;
        partition_cells[vertices - 1] = 0;

        static DEFAULTOPTIONS_GRAPH(options);
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        statsblk stats{};
        densenauty(
            graph_data.data(), labels.data(), partition_cells.data(), orbits.data(),
            &options, &stats, words, vertices, canonical_data.data());
        if (stats.errstatus) throw std::runtime_error("nauty canonicalisation failed");

        std::array<int, 64> row_image{};
        std::array<int, 64> column_image{};
        row_image.fill(-1);
        column_image.fill(-1);
        if (allow_transpose) {
            std::array<uint64_t, 64> axis_adjacency{};
            for (unsigned cell_vertex = cell_start;
                 cell_vertex < cell_start + rows * columns; ++cell_vertex) {
                std::array<unsigned, 2> axes{};
                unsigned found = 0;
                for (unsigned neighbour = 0; neighbour < rows + columns; ++neighbour) {
                    if (ISELEMENT(GRAPHROW(canonical_data.data(), cell_vertex, words), neighbour)) {
                        if (found >= 2) throw std::runtime_error("canonical cell has too many axes");
                        axes[found++] = neighbour;
                    }
                }
                if (found != 2) throw std::runtime_error("canonical cell has wrong axis degree");
                axis_adjacency[axes[0]] |= uint64_t{1} << axes[1];
                axis_adjacency[axes[1]] |= uint64_t{1} << axes[0];
            }
            uint64_t first_side = 1;
            uint64_t frontier = 1;
            uint64_t visited = 1;
            bool take_first = false;
            while (frontier) {
                uint64_t following = 0;
                uint64_t scan = frontier;
                while (scan) {
                    unsigned axis = static_cast<unsigned>(__builtin_ctzll(scan));
                    following |= axis_adjacency[axis];
                    scan &= scan - 1;
                }
                following &= ~visited;
                visited |= following;
                take_first = !take_first;
                if (!take_first) first_side |= following;
                frontier = following;
            }
            unsigned next_row = 0;
            unsigned next_column = 0;
            for (unsigned axis = 0; axis < rows + columns; ++axis) {
                if (first_side & (uint64_t{1} << axis)) row_image[axis] = static_cast<int>(next_row++);
                else column_image[axis] = static_cast<int>(next_column++);
            }
            if (next_row != rows || next_column != columns) {
                throw std::runtime_error("canonical axis bipartition has wrong size");
            }
        } else {
            for (unsigned row = 0; row < rows; ++row) row_image[row] = static_cast<int>(row);
            for (unsigned column = 0; column < columns; ++column) {
                column_image[rows + column] = static_cast<int>(column);
            }
        }

        Partition output(rows * columns, static_cast<char>(0xff));
        for (unsigned cell_vertex = cell_start;
             cell_vertex < cell_start + rows * columns; ++cell_vertex) {
            unsigned row = rows;
            unsigned column = columns;
            unsigned block = blocks;
            for (unsigned neighbour = 0; neighbour < static_cast<unsigned>(vertices); ++neighbour) {
                if (!ISELEMENT(GRAPHROW(canonical_data.data(), cell_vertex, words), neighbour)) continue;
                if (neighbour < rows + columns) {
                    if (row_image[neighbour] >= 0) row = static_cast<unsigned>(row_image[neighbour]);
                    else if (column_image[neighbour] >= 0) {
                        column = static_cast<unsigned>(column_image[neighbour]);
                    }
                }
                else if (neighbour >= block_start) block = neighbour - block_start;
            }
            if (row == rows || column == columns || block == blocks) {
                throw std::runtime_error("invalid canonical incidence graph");
            }
            output[row * columns + column] = static_cast<char>(block);
        }
        if (std::find(output.begin(), output.end(), static_cast<char>(0xff)) != output.end()) {
            throw std::runtime_error("incomplete canonical partition");
        }
        return normalize(output);
    }

    unsigned rows_;
    unsigned columns_;
    unsigned cells_;
    uint64_t calls_ = 0;
};

struct Census {
    std::vector<Partition> states;
    uint64_t candidates = 0;
    bool capped = false;
};

Census enumerate_lattice(
    unsigned rows,
    unsigned columns,
    bool quotient,
    uint64_t state_cap,
    double time_cap
) {
    const unsigned cells = rows * columns;
    if (cells > 81) throw std::runtime_error("probe supports at most 81 cells");
    std::vector<Rectangle> generators = rectangles(rows, columns);
    Partition bottom(cells, '\0');
    for (unsigned cell = 0; cell < cells; ++cell) bottom[cell] = static_cast<char>(cell);

    NautyCanonicalizer canonicalizer(rows, columns);
    if (quotient) bottom = canonicalizer(bottom);
    std::unordered_set<Partition> seen;
    seen.reserve(static_cast<size_t>(std::min<uint64_t>(state_cap, 4'000'000)));
    seen.insert(bottom);
    std::unordered_map<Partition, Partition> raw_canonical_cache;
    uint64_t raw_cache_hits = 0;
    if (quotient) raw_canonical_cache.reserve(
        static_cast<size_t>(std::min<uint64_t>(state_cap * 4, 8'000'000)));
    std::vector<Partition> all{bottom};
    std::vector<Partition> frontier{bottom};
    auto started = std::chrono::steady_clock::now();
    uint64_t candidates = 0;
    uint64_t next_time_check = 65'536;
    bool capped = false;

    for (unsigned depth = 0; !frontier.empty(); ++depth) {
        std::vector<Partition> following;
        for (const Partition& state : frontier) {
            for (const Rectangle& rectangle : generators) {
                Partition child;
                if (!join_rectangle(state, rectangle, child)) continue;
                ++candidates;
                if (quotient) {
                    auto cached = raw_canonical_cache.find(child);
                    if (cached != raw_canonical_cache.end()) {
                        child = cached->second;
                        ++raw_cache_hits;
                    } else {
                        Partition raw = child;
                        child = canonicalizer(child);
                        raw_canonical_cache.emplace(std::move(raw), child);
                    }
                }
                if (seen.insert(child).second) {
                    following.push_back(child);
                    all.push_back(child);
                    if (all.size() >= state_cap) {
                        capped = true;
                        break;
                    }
                }
            }
            if (capped) break;
            if (candidates >= next_time_check && time_cap > 0) {
                next_time_check = candidates + 65'536;
                double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                if (seconds >= time_cap) {
                    capped = true;
                    break;
                }
            }
        }
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "depth=" << depth
                  << " frontier=" << frontier.size()
                  << " next=" << following.size()
                  << " total=" << all.size()
                  << " candidates=" << candidates
                  << " canonical_calls=" << canonicalizer.calls()
                  << " raw_cache=" << raw_canonical_cache.size()
                  << " raw_cache_hits=" << raw_cache_hits
                  << " seconds=" << seconds << std::endl;
        if (capped) break;
        frontier = std::move(following);
    }
    return {std::move(all), candidates, capped};
}

bool refines(const Partition& finer, const Partition& coarser) {
    std::array<unsigned char, 256> image{};
    image.fill(0xff);
    for (size_t cell = 0; cell < finer.size(); ++cell) {
        unsigned char source = static_cast<unsigned char>(finer[cell]);
        unsigned char target = static_cast<unsigned char>(coarser[cell]);
        if (image[source] == 0xff) image[source] = target;
        else if (image[source] != target) return false;
    }
    return true;
}

void evaluate_mobius(std::vector<Partition> states) {
    std::sort(states.begin(), states.end(), [](const Partition& left, const Partition& right) {
        unsigned left_blocks = block_count(left);
        unsigned right_blocks = block_count(right);
        return left_blocks != right_blocks ? left_blocks > right_blocks : left < right;
    });
    std::vector<uint64_t> mobius(states.size());
    uint64_t nonzero = 0;
    uint64_t answer = 0;
    for (size_t current = 0; current < states.size(); ++current) {
        uint64_t lower_sum = 0;
        for (size_t previous = 0; previous < current; ++previous) {
            if (!mobius[previous] || !refines(states[previous], states[current])) continue;
            lower_sum += mobius[previous];
            if (lower_sum >= kPrime) lower_sum -= kPrime;
        }
        mobius[current] = current == 0 ? 1 : (lower_sum ? kPrime - lower_sum : 0);
        if (mobius[current]) ++nonzero;
        answer += mul_mod(mobius[current], pow_mod(4, block_count(states[current])));
        if (answer >= kPrime) answer -= kPrime;
    }
    std::cout << "mobius_states=" << states.size()
              << " nonzero_mobius=" << nonzero
              << " T4_mod_prime=" << answer << '\n';
}

void self_test() {
    for (auto [rows, columns] : {
             std::pair{2U, 3U}, std::pair{3U, 3U}, std::pair{4U, 4U}}) {
        Census labelled = enumerate_lattice(rows, columns, false, 1'000'000, 0);
        Census quotient = enumerate_lattice(rows, columns, true, 1'000'000, 0);
        NautyCanonicalizer canonicalizer(rows, columns);
        std::unordered_set<Partition> independently_quotiented;
        for (const Partition& state : labelled.states) {
            independently_quotiented.insert(canonicalizer(state));
        }
        if (independently_quotiented.size() != quotient.states.size()) {
            throw std::runtime_error("orbit self-test mismatch");
        }
    }
    std::cout << "self_test=pass\n";
}

} // namespace

int main(int argc, char** argv) try {
    unsigned rows = 4;
    unsigned columns = 4;
    uint64_t state_cap = 10'000'000;
    double time_cap = 120;
    bool quotient = true;
    bool mobius = false;
    bool run_self_test = false;
    for (int index = 1; index < argc; ++index) {
        std::string_view option = argv[index];
        if (option == "--orbit") quotient = true;
        else if (option == "--labelled") quotient = false;
        else if (option == "--mobius") mobius = true;
        else if (option == "--self-test") run_self_test = true;
        else {
            if (++index >= argc) throw std::runtime_error("missing option value");
            std::string_view value = argv[index];
            if (option == "--rows") rows = static_cast<unsigned>(parse_u64(value));
            else if (option == "--columns") columns = static_cast<unsigned>(parse_u64(value));
            else if (option == "--state-cap") state_cap = parse_u64(value);
            else if (option == "--time-cap") time_cap = std::stod(std::string(value));
            else throw std::runtime_error("unknown option: " + std::string(option));
        }
    }
    if (run_self_test) {
        self_test();
        return 0;
    }
    if (mobius && quotient) throw std::runtime_error("Möbius evaluation currently requires --labelled");
    std::cout << "rows=" << rows << " columns=" << columns
              << " rectangles=" << rectangles(rows, columns).size()
              << " quotient=" << quotient << '\n';
    Census census = enumerate_lattice(rows, columns, quotient, state_cap, time_cap);
    std::cout << "status=" << (census.capped ? "capped" : "complete")
              << " states=" << census.states.size()
              << " candidates=" << census.candidates << '\n';
    if (mobius && !census.capped) evaluate_mobius(std::move(census.states));
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
