// Exact labelled second-class census for the dense-first 9x9 proposal.
//
// A C4-free binary matrix can use a pair of columns in at most one row.
// Represent the column pairs already used by a partial matrix as a 36-bit
// mask.  For a fixed first colour class A, this DP counts every C4-free B
// contained in the complement of A, classified by |B|.  It intentionally
// retains row/column labels: the result is an upper bound on the work after
// quotienting B by the stabilizer of A.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr unsigned kMaximumRows = 9;
constexpr unsigned kMaximumColumns = 9;
constexpr unsigned kEdgeBits = 7;
constexpr uint64_t kPairMaskBits = (uint64_t{1} << 36) - 1;

struct Choice {
    uint64_t pairs;
    uint8_t edges;
};

struct Options {
    unsigned rows = 9;
    unsigned columns = 9;
    unsigned minimum_edges = 17;
    unsigned maximum_edges = 29;
    uint64_t state_cap = 100'000'000;
    std::vector<uint16_t> first_rows;
};

[[noreturn]] void usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --first ROW0,...,ROW8 [--minimum N] [--maximum N]"
           " [--state-cap N]\n"
        << "Rows are hexadecimal bit masks, for example"
           " 181,086,030,....\n";
    std::exit(2);
}

uint64_t parse_u64(std::string_view text, int base = 10) {
    uint64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    auto result = std::from_chars(begin, end, value, base);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error("invalid integer: " + std::string(text));
    }
    return value;
}

std::vector<uint16_t> parse_rows(std::string_view text) {
    std::vector<uint16_t> rows;
    while (!text.empty()) {
        size_t comma = text.find(',');
        std::string_view field = text.substr(0, comma);
        rows.push_back(static_cast<uint16_t>(parse_u64(field, 16)));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return rows;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        std::string_view option = argv[index];
        if (index + 1 >= argc) usage(argv[0]);
        std::string_view value = argv[++index];
        if (option == "--first") options.first_rows = parse_rows(value);
        else if (option == "--minimum") options.minimum_edges = parse_u64(value);
        else if (option == "--maximum") options.maximum_edges = parse_u64(value);
        else if (option == "--state-cap") options.state_cap = parse_u64(value);
        else if (option == "--rows") options.rows = parse_u64(value);
        else if (option == "--columns") options.columns = parse_u64(value);
        else usage(argv[0]);
    }
    if (options.rows < 1 || options.rows > kMaximumRows ||
        options.columns < 1 || options.columns > kMaximumColumns ||
        options.first_rows.size() != options.rows ||
        options.minimum_edges > options.maximum_edges ||
        options.maximum_edges > options.rows * options.columns) {
        usage(argv[0]);
    }
    uint16_t valid_bits = (uint16_t{1} << options.columns) - 1;
    for (uint16_t row : options.first_rows) {
        if (row & ~valid_bits) usage(argv[0]);
    }
    return options;
}

std::array<std::array<int, kMaximumColumns>, kMaximumColumns>
pair_indices(unsigned columns) {
    std::array<std::array<int, kMaximumColumns>, kMaximumColumns> indices{};
    int next = 0;
    for (unsigned first = 0; first < columns; ++first) {
        for (unsigned second = first + 1; second < columns; ++second) {
            indices[first][second] = next++;
        }
    }
    return indices;
}

std::vector<Choice> choices_for_row(
    uint16_t allowed,
    unsigned columns,
    const std::array<std::array<int, kMaximumColumns>, kMaximumColumns>& indices
) {
    std::vector<Choice> choices;
    uint16_t subset = allowed;
    for (;;) {
        uint64_t pairs = 0;
        for (unsigned first = 0; first < columns; ++first) {
            if (!(subset & (uint16_t{1} << first))) continue;
            for (unsigned second = first + 1; second < columns; ++second) {
                if (subset & (uint16_t{1} << second)) {
                    pairs |= uint64_t{1} << indices[first][second];
                }
            }
        }
        choices.push_back({pairs, static_cast<uint8_t>(__builtin_popcount(subset))});
        if (!subset) break;
        subset = (subset - 1) & allowed;
    }
    std::sort(choices.begin(), choices.end(), [](Choice left, Choice right) {
        return left.edges > right.edges;
    });
    return choices;
}

uint64_t make_key(uint64_t pairs, unsigned edges) {
    return pairs | (uint64_t{edges} << 36);
}

unsigned key_edges(uint64_t key) {
    return static_cast<unsigned>(key >> 36);
}

uint64_t key_pairs(uint64_t key) {
    return key & kPairMaskBits;
}

}  // namespace

int main(int argc, char** argv) try {
    std::cout.setf(std::ios::unitbuf);
    Options options = parse_options(argc, argv);
    uint16_t full = (uint16_t{1} << options.columns) - 1;
    auto indices = pair_indices(options.columns);

    // Dense first rows leave fewer choices.  Processing them first is an exact
    // row relabelling and reduces intermediate state counts.
    std::sort(options.first_rows.begin(), options.first_rows.end(),
              [](uint16_t left, uint16_t right) {
                  return __builtin_popcount(left) > __builtin_popcount(right);
              });

    std::vector<std::vector<Choice>> row_choices;
    std::vector<unsigned> suffix_capacity(options.rows + 1, 0);
    for (uint16_t first_row : options.first_rows) {
        row_choices.push_back(choices_for_row(full ^ first_row, options.columns, indices));
    }
    for (unsigned row = options.rows; row-- > 0;) {
        unsigned maximum = row_choices[row].front().edges;
        suffix_capacity[row] = suffix_capacity[row + 1] + maximum;
    }

    using Counts = std::unordered_map<uint64_t, uint64_t>;
    Counts current;
    current.reserve(1 << 16);
    current.emplace(make_key(0, 0), 1);
    auto started = std::chrono::steady_clock::now();

    for (unsigned row = 0; row < options.rows; ++row) {
        Counts following;
        size_t reserve = std::min<uint64_t>(
            options.state_cap,
            std::max<uint64_t>(1 << 16, current.size() * 8));
        following.reserve(reserve);
        for (const auto& [key, multiplicity] : current) {
            uint64_t used_pairs = key_pairs(key);
            unsigned used_edges = key_edges(key);
            for (Choice choice : row_choices[row]) {
                unsigned edges = used_edges + choice.edges;
                if (edges > options.maximum_edges ||
                    edges + suffix_capacity[row + 1] < options.minimum_edges ||
                    (used_pairs & choice.pairs)) {
                    continue;
                }
                uint64_t next_key = make_key(used_pairs | choice.pairs, edges);
                auto [position, inserted] = following.try_emplace(next_key, 0);
                if (inserted && following.size() > options.state_cap) {
                    throw std::runtime_error(
                        "state cap exceeded after row " + std::to_string(row + 1));
                }
                uint64_t& count = position->second;
                if (UINT64_MAX - count < multiplicity) {
                    throw std::runtime_error("labelled count exceeds uint64_t");
                }
                count += multiplicity;
            }
        }
        current = std::move(following);
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "row=" << row + 1 << " states=" << current.size()
                  << " seconds=" << seconds << '\n';
    }

    std::vector<uint64_t> totals(options.maximum_edges + 1, 0);
    for (const auto& [key, multiplicity] : current) {
        totals[key_edges(key)] += multiplicity;
    }
    uint64_t total = 0;
    for (unsigned edges = options.minimum_edges; edges <= options.maximum_edges; ++edges) {
        total += totals[edges];
        std::cout << "edges=" << edges << " labelled_second_classes="
                  << totals[edges] << '\n';
    }
    std::cout << "total_labelled_second_classes=" << total << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
