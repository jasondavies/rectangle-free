#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>

#define PREFIX_BMMA_COST_CENSUS_NO_MAIN
#include "prefix_bmma_cost_census.cpp"

namespace {

struct NormalizedEntry {
    uint64_t mask;
    uint32_t weight;
    bool operator==(const NormalizedEntry& other) const {
        return mask == other.mask && weight == other.weight;
    }
};

struct Signature {
    uint64_t first = 0, second = 0;
    uint32_t entries = 0;
    bool operator==(const Signature& other) const {
        return first == other.first && second == other.second &&
               entries == other.entries;
    }
};

struct SignatureHash {
    size_t operator()(const Signature& value) const {
        return size_t(mix64(value.first ^
            (value.second + UINT64_C(0x9e3779b97f4a7c15)) ^ value.entries));
    }
};

struct DistributionSummary {
    Signature signature;
    uint32_t gcd = 0;
    uint32_t semantic_id = UINT32_MAX;
};

struct ExactClass {
    uint32_t representative = 0;
    std::vector<NormalizedEntry> entries;
    uint32_t entry_count = 0;
    uint32_t uses = 0;
};

struct AutomorphismStats {
    uint32_t original = 0;
    uint32_t structural = 0;
    uint32_t marginal_candidates = 0;
    uint32_t behavioral = 0;
    uint32_t exact_tests = 0;
    uint32_t entries = 0;
};

static uint64_t swap_token_planes_local(uint64_t mask) {
    constexpr uint64_t plane_mask = (UINT64_C(1) << PAIRS) - 1;
    return ((mask & plane_mask) << PAIRS) |
           ((mask >> PAIRS) & plane_mask);
}

static uint32_t canonical_half(uint32_t key) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    std::array<unsigned, HALF_COLUMNS> permutation{0, 1, 2, 3};
    uint32_t best = UINT32_MAX;
    do {
        std::array<uint8_t, ROWS> transformed{};
        for (unsigned row = 0; row < ROWS; row++)
            for (unsigned output = 0; output < HALF_COLUMNS; output++)
                transformed[row] |=
                    ((rows[row] >> permutation[output]) & 1U) << output;
        std::sort(transformed.begin(), transformed.end());
        uint32_t candidate = 0;
        for (uint8_t row : transformed) candidate = (candidate << 4) | row;
        best = std::min(best, candidate);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

static std::vector<uint32_t> enumerate_canonical_halves() {
    std::vector<uint32_t> keys;
    std::array<uint8_t, ROWS> rows{};
    auto visit = [&](auto&& self, unsigned position, unsigned minimum) -> void {
        if (position == rows.size()) {
            uint32_t key = 0;
            for (uint8_t row : rows) key = (key << HALF_COLUMNS) | row;
            keys.push_back(canonical_half(key));
            return;
        }
        for (unsigned value = minimum; value < (1U << HALF_COLUMNS); value++) {
            rows[position] = uint8_t(value);
            self(self, position + 1, value);
        }
    };
    visit(visit, 0, 0);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 25207)
        throw std::runtime_error("canonical 8x4 census is incomplete");
    return keys;
}

struct StructuralCore {
    uint32_t key = 0;
    uint8_t singleton_columns = 0;
};

static StructuralCore structural_core(uint32_t key, bool complement) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    if (complement)
        for (uint8_t& row : rows) row ^= 15U;
    uint8_t singleton_columns = 0;
    for (unsigned column = 0; column < HALF_COLUMNS; column++) {
        unsigned active = 0;
        for (uint8_t row : rows) active += (row >> column) & 1U;
        if (active > 1) continue;
        singleton_columns += active == 1;
        for (uint8_t& row : rows) row &= uint8_t(~(1U << column));
    }
    uint32_t core = 0;
    for (uint8_t row : rows) core = (core << HALF_COLUMNS) | row;
    return StructuralCore{canonical_half(core), singleton_columns};
}

static std::array<uint8_t, HALF_COLUMNS> column_multiset(uint32_t key) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    std::array<uint8_t, HALF_COLUMNS> columns{};
    for (unsigned column = 0; column < HALF_COLUMNS; column++)
        for (unsigned row = 0; row < ROWS; row++)
            columns[column] |= ((rows[row] >> column) & 1U) << row;
    std::sort(columns.begin(), columns.end());
    return columns;
}

static uint32_t permute_half_rows(
    uint32_t key, const std::array<unsigned, ROWS>& permutation) {
    std::array<uint8_t, ROWS> rows{};
    for (int row = ROWS - 1; row >= 0; row--) {
        rows[size_t(row)] = uint8_t(key & 15U);
        key >>= HALF_COLUMNS;
    }
    uint32_t result = 0;
    for (unsigned output = 0; output < ROWS; output++)
        result = (result << HALF_COLUMNS) | rows[permutation[output]];
    return result;
}

static unsigned pair_index(unsigned first, unsigned second) {
    if (first > second) std::swap(first, second);
    return unsigned(g_pair_index[first][second]);
}

static uint64_t permute_token_rows(
    uint64_t mask, const std::array<unsigned, ROWS>& permutation) {
    uint64_t result = 0;
    for (unsigned colour = 0; colour < 2; colour++)
        for (unsigned first = 0; first < ROWS; first++)
            for (unsigned second = first + 1; second < ROWS; second++) {
                unsigned destination = pair_index(first, second);
                unsigned source =
                    pair_index(permutation[first], permutation[second]);
                result |= ((mask >> (colour * PAIRS + source)) & 1U)
                          << (colour * PAIRS + destination);
            }
    return result;
}

struct EdgeInvariant {
    uint64_t weighted = 0;
    uint64_t both_weighted = 0;
    uint32_t supports = 0;
    bool operator==(const EdgeInvariant& other) const {
        return weighted == other.weighted &&
               both_weighted == other.both_weighted &&
               supports == other.supports;
    }
};

static bool exact_row_automorphism(
    const std::vector<FullWeightedEntry>& entries, uint32_t divisor,
    const std::array<unsigned, ROWS>& permutation) {
    for (const FullWeightedEntry& entry : entries) {
        uint64_t transformed = permute_token_rows(entry.mask, permutation);
        auto found = std::lower_bound(
            entries.begin(), entries.end(), transformed,
            [](const FullWeightedEntry& item, uint64_t mask) {
                return item.mask < mask;
            });
        if (found == entries.end() || found->mask != transformed ||
            found->weight / divisor != entry.weight / divisor)
            return false;
    }
    return true;
}

static AutomorphismStats automorphism_stats(uint32_t prefix) {
    std::vector<FullWeightedEntry> entries =
        build_full_weighted_distribution(prefix, false);
    uint32_t divisor = 0;
    for (const FullWeightedEntry& entry : entries)
        divisor = std::gcd(divisor, entry.weight);
    if (!divisor) divisor = 1;
    std::array<EdgeInvariant, PAIRS> invariants{};
    for (const FullWeightedEntry& entry : entries) {
        uint32_t weight = entry.weight / divisor;
        for (unsigned pair = 0; pair < PAIRS; pair++) {
            bool first = (entry.mask >> pair) & 1U;
            bool second = (entry.mask >> (PAIRS + pair)) & 1U;
            if (first) {
                invariants[pair].weighted += weight;
                invariants[pair].supports++;
            }
            if (first && second) invariants[pair].both_weighted += weight;
        }
    }
    StructuralCore core = structural_core(prefix, false);
    auto original_columns = column_multiset(prefix);
    auto core_columns = column_multiset(core.key);
    AutomorphismStats result;
    result.entries = entries.size();
    std::array<unsigned, ROWS> permutation{0, 1, 2, 3, 4, 5, 6, 7};
    do {
        bool original = column_multiset(permute_half_rows(prefix, permutation)) ==
                        original_columns;
        bool structural =
            column_multiset(permute_half_rows(core.key, permutation)) ==
            core_columns;
        result.original += original;
        result.structural += structural;
        bool marginal = true;
        for (unsigned first = 0; first < ROWS && marginal; first++)
            for (unsigned second = first + 1; second < ROWS; second++) {
                unsigned destination = pair_index(first, second);
                unsigned source =
                    pair_index(permutation[first], permutation[second]);
                if (!(invariants[destination] == invariants[source])) {
                    marginal = false;
                    break;
                }
            }
        if (!marginal) continue;
        result.marginal_candidates++;
        if (structural) {
            result.behavioral++;
        } else {
            result.exact_tests++;
            result.behavioral +=
                exact_row_automorphism(entries, divisor, permutation);
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (result.behavioral < result.structural ||
        result.structural < result.original)
        throw std::logic_error("automorphism subgroup invariant failed");
    return result;
}

static std::vector<NormalizedEntry> normalized_distribution(
    uint32_t prefix, bool complement, uint32_t* gcd_out = nullptr) {
    std::vector<FullWeightedEntry> full =
        build_full_weighted_distribution(prefix, complement);
    uint32_t divisor = 0;
    for (const FullWeightedEntry& entry : full)
        divisor = std::gcd(divisor, entry.weight);
    if (!divisor) {
        if (!full.empty())
            throw std::logic_error("nonempty distribution has zero gcd");
        if (gcd_out) *gcd_out = 0;
        return {};
    }
    std::vector<NormalizedEntry> result;
    result.reserve((full.size() + 1) / 2);
    uint64_t expanded = 0;
    for (const FullWeightedEntry& entry : full) {
        uint64_t swapped = swap_token_planes_local(entry.mask);
        if (entry.mask > swapped) continue;
        result.push_back(NormalizedEntry{entry.mask, entry.weight / divisor});
        expanded += entry.mask == swapped ? 1 : 2;
    }
    if (expanded != full.size())
        throw std::runtime_error("token-plane quotient invariant failed");
    if (gcd_out) *gcd_out = divisor;
    return result;
}

static Signature signature_of(const std::vector<NormalizedEntry>& entries) {
    uint64_t first = UINT64_C(0x6a09e667f3bcc909);
    uint64_t second = UINT64_C(0xbb67ae8584caa73b);
    uint64_t ordinal = 0;
    for (const NormalizedEntry& entry : entries) {
        uint64_t value = mix64(entry.mask ^
            (UINT64_C(0x9e3779b97f4a7c15) * entry.weight));
        first = mix64(first ^ value ^ ordinal);
        second += mix64(value + UINT64_C(0x3c6ef372fe94f82b) * ++ordinal);
        second = (second << 17) | (second >> 47);
    }
    return Signature{first, mix64(second ^ entries.size()),
                     uint32_t(entries.size())};
}

static DistributionSummary summarize(uint32_t prefix, bool complement) {
    uint32_t divisor = 0;
    std::vector<NormalizedEntry> entries =
        normalized_distribution(prefix, complement, &divisor);
    return DistributionSummary{signature_of(entries), divisor, UINT32_MAX};
}

}  // namespace

static std::vector<std::string> split_paths(const std::string& value) {
    std::vector<std::string> result;
    size_t begin = 0;
    for (;;) {
        size_t end = value.find(',', begin);
        result.push_back(value.substr(begin, end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            std::cerr << "usage: " << argv[0]
                      << " [AUTOMORPHISM_SAMPLES=128;0=all]"
                         " [SHARD[,SHARD...]] [SAMPLES_PER_SHARD=65536]\n";
            return 2;
        }
        initialise_tables();
        initialise_weighted_increments();
        double start = seconds_now();
        std::vector<uint32_t> prefixes = enumerate_canonical_halves();
        std::vector<DistributionSummary> summaries(2 * prefixes.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (long long half = 0; half < (long long)prefixes.size(); half++) {
            summaries[2 * size_t(half)] = summarize(prefixes[size_t(half)], false);
            summaries[2 * size_t(half) + 1] = summarize(prefixes[size_t(half)], true);
        }
        double fingerprint_seconds = seconds_now() - start;

        std::unordered_map<Signature, std::vector<uint32_t>, SignatureHash>
            candidate_groups;
        candidate_groups.reserve(summaries.size() * 2);
        for (uint32_t index = 0; index < summaries.size(); index++)
            candidate_groups[summaries[index].signature].push_back(index);
        uint64_t candidate_distributions = 0, candidate_group_count = 0;
        for (const auto& item : candidate_groups) {
            if (item.second.size() > 1) {
                candidate_group_count++;
                candidate_distributions += item.second.size();
            }
        }

        std::vector<ExactClass> exact_classes;
        exact_classes.reserve(candidate_groups.size());
        uint64_t hash_collision_groups = 0;
        double verify_start = seconds_now();
        for (const auto& item : candidate_groups) {
            if (item.second.size() == 1) {
                uint32_t index = item.second.front();
                uint32_t id = uint32_t(exact_classes.size());
                exact_classes.push_back(ExactClass{
                    index, {}, summaries[index].signature.entries, 1});
                summaries[index].semantic_id = id;
                continue;
            }
            std::vector<uint32_t> class_ids;
            for (uint32_t index : item.second) {
                std::vector<NormalizedEntry> entries = normalized_distribution(
                    prefixes[index / 2], bool(index & 1U));
                uint32_t found = UINT32_MAX;
                for (uint32_t class_id : class_ids) {
                    if (exact_classes[class_id].entries == entries) {
                        found = class_id;
                        break;
                    }
                }
                if (found == UINT32_MAX) {
                    found = uint32_t(exact_classes.size());
                    uint32_t entry_count = uint32_t(entries.size());
                    exact_classes.push_back(ExactClass{
                        index, std::move(entries), entry_count, 0});
                    class_ids.push_back(found);
                }
                summaries[index].semantic_id = found;
                exact_classes[found].uses++;
            }
            if (class_ids.size() > 1) hash_collision_groups++;
        }
        double verify_seconds = seconds_now() - verify_start;

        uint64_t total_entries = 0, unique_entries = 0;
        uint64_t duplicate_distributions = 0, duplicate_classes = 0;
        uint32_t maximum_class = 0;
        std::unordered_map<uint32_t, uint64_t> gcd_histogram;
        for (const DistributionSummary& summary : summaries) {
            total_entries += summary.signature.entries;
            gcd_histogram[summary.gcd]++;
        }
        std::unordered_set<uint32_t> zero_keys;
        uint64_t selected_zero = 0, complement_zero = 0;
        for (size_t half = 0; half < prefixes.size(); half++) {
            if (!summaries[2 * half].signature.entries) {
                zero_keys.insert(prefixes[half]);
                selected_zero++;
            }
            complement_zero +=
                !summaries[2 * half + 1].signature.entries;
        }
        for (const ExactClass& item : exact_classes) {
            unique_entries += item.entry_count;
            maximum_class = std::max(maximum_class, item.uses);
            if (item.uses > 1) {
                duplicate_classes++;
                duplicate_distributions += item.uses - 1;
            }
        }
        if (exact_classes.size() + duplicate_distributions != summaries.size())
            throw std::logic_error("exact class accounting failed");

        std::unordered_map<uint64_t, uint32_t> linked_classes;
        std::unordered_set<uint32_t> selected_classes, complement_classes;
        for (size_t half = 0; half < prefixes.size(); half++) {
            uint64_t key = (uint64_t(summaries[2 * half].semantic_id) << 32) |
                           summaries[2 * half + 1].semantic_id;
            linked_classes.emplace(key, uint32_t(linked_classes.size()));
            selected_classes.insert(summaries[2 * half].semantic_id);
            complement_classes.insert(summaries[2 * half + 1].semantic_id);
        }
        uint64_t selected_entries = 0, selected_unique_entries = 0;
        uint64_t complement_entries = 0, complement_unique_entries = 0;
        for (size_t half = 0; half < prefixes.size(); half++) {
            selected_entries += summaries[2 * half].signature.entries;
            complement_entries += summaries[2 * half + 1].signature.entries;
        }
        for (uint32_t id : selected_classes)
            selected_unique_entries += exact_classes[id].entry_count;
        for (uint32_t id : complement_classes)
            complement_unique_entries += exact_classes[id].entry_count;

        std::unordered_set<uint32_t> selected_cores, complement_cores;
        std::unordered_set<uint64_t> linked_cores;
        std::array<uint64_t, HALF_COLUMNS + 1> singleton_histogram{};
        for (uint32_t prefix : prefixes) {
            StructuralCore selected = structural_core(prefix, false);
            StructuralCore complement = structural_core(prefix, true);
            selected_cores.insert(selected.key);
            complement_cores.insert(complement.key);
            linked_cores.insert((uint64_t(selected.key) << 32) |
                                complement.key);
            singleton_histogram[selected.singleton_columns]++;
            singleton_histogram[complement.singleton_columns]++;
        }
        auto core_entries = [&](const std::unordered_set<uint32_t>& cores) {
            uint64_t result = 0;
            for (uint32_t core : cores) {
                auto found = std::lower_bound(prefixes.begin(), prefixes.end(),
                                              core);
                if (found == prefixes.end() || *found != core)
                    throw std::logic_error("structural core is not canonical");
                size_t index = size_t(found - prefixes.begin());
                result += summaries[2 * index].signature.entries;
            }
            return result;
        };
        uint64_t selected_core_entries = core_entries(selected_cores);
        uint64_t complement_core_entries = core_entries(complement_cores);

        size_t automorphism_samples =
            argc == 2 ? std::stoull(argv[1]) : 128;
        if (!automorphism_samples || automorphism_samples > prefixes.size())
            automorphism_samples = prefixes.size();
        std::vector<AutomorphismStats> automorphisms(automorphism_samples);
        double automorphism_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
        for (long long sample = 0;
             sample < (long long)automorphism_samples; sample++) {
            size_t index = size_t((U128(sample) * prefixes.size() +
                                   prefixes.size() / 2) /
                                  automorphism_samples);
            if (index >= prefixes.size()) index = prefixes.size() - 1;
            automorphisms[size_t(sample)] =
                automorphism_stats(prefixes[index]);
        }
        double automorphism_seconds = seconds_now() - automorphism_start;
        uint64_t original_aut = 0, structural_aut = 0, behavioral_aut = 0;
        uint64_t marginal_candidates = 0, exact_tests = 0;
        uint64_t structural_larger = 0, behavioral_larger = 0;
        uint64_t behavioral_larger_zero = 0, behavioral_larger_nonzero = 0;
        uint64_t behavioral_excess_zero = 0, behavioral_excess_nonzero = 0;
        uint32_t maximum_behavioral = 0;
        for (const AutomorphismStats& item : automorphisms) {
            original_aut += item.original;
            structural_aut += item.structural;
            behavioral_aut += item.behavioral;
            marginal_candidates += item.marginal_candidates;
            exact_tests += item.exact_tests;
            structural_larger += item.structural > item.original;
            behavioral_larger += item.behavioral > item.structural;
            if (item.behavioral > item.structural) {
                if (item.entries) {
                    behavioral_larger_nonzero++;
                    behavioral_excess_nonzero +=
                        item.behavioral - item.structural;
                } else {
                    behavioral_larger_zero++;
                    behavioral_excess_zero +=
                        item.behavioral - item.structural;
                }
            }
            maximum_behavioral = std::max(maximum_behavioral, item.behavioral);
        }
        std::vector<std::pair<uint32_t, uint64_t>> gcds(gcd_histogram.begin(),
                                                         gcd_histogram.end());
        std::sort(gcds.begin(), gcds.end());
        std::cout << std::setprecision(12)
                  << "BEHAVIORAL_DISTRIBUTION_TOTAL halves=" << prefixes.size()
                  << " distributions=" << summaries.size()
                  << " quotient_entries=" << total_entries
                  << " fingerprint_groups=" << candidate_groups.size()
                  << " candidate_groups=" << candidate_group_count
                  << " candidate_distributions=" << candidate_distributions
                  << " fingerprint_seconds=" << fingerprint_seconds << '\n';
        std::cout << "BEHAVIORAL_DISTRIBUTION_EXACT unique="
                  << exact_classes.size()
                  << " duplicate_distributions=" << duplicate_distributions
                  << " duplicate_classes=" << duplicate_classes
                  << " maximum_class=" << maximum_class
                  << " unique_entries=" << unique_entries
                  << " entry_ratio=" << double(unique_entries) / total_entries
                  << " hash_collision_groups=" << hash_collision_groups
                  << " verify_seconds=" << verify_seconds << " exact=OK\n";
        std::cout << "BEHAVIORAL_LINKED_EXACT unique=" << linked_classes.size()
                  << " duplicates=" << prefixes.size() - linked_classes.size()
                  << " ratio=" << double(linked_classes.size()) / prefixes.size()
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_COMPONENT component=selected unique="
                  << selected_classes.size()
                  << " entries=" << selected_entries
                  << " unique_entries=" << selected_unique_entries
                  << " entry_ratio="
                  << double(selected_unique_entries) / selected_entries
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_COMPONENT component=complement unique="
                  << complement_classes.size()
                  << " entries=" << complement_entries
                  << " unique_entries=" << complement_unique_entries
                  << " entry_ratio="
                  << double(complement_unique_entries) / complement_entries
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_STRUCTURAL component=selected unique="
                  << selected_cores.size()
                  << " unique_entries=" << selected_core_entries
                  << " entry_ratio="
                  << double(selected_core_entries) / selected_entries
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_STRUCTURAL component=complement unique="
                  << complement_cores.size()
                  << " unique_entries=" << complement_core_entries
                  << " entry_ratio="
                  << double(complement_core_entries) / complement_entries
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_STRUCTURAL_LINKED unique="
                  << linked_cores.size()
                  << " duplicates=" << prefixes.size() - linked_cores.size()
                  << " ratio=" << double(linked_cores.size()) / prefixes.size()
                  << " exact=OK\n";
        for (unsigned singleton = 0; singleton <= HALF_COLUMNS; singleton++)
            if (singleton_histogram[singleton])
                std::cout << "BEHAVIORAL_SINGLETON_COLUMNS count=" << singleton
                          << " distributions="
                          << singleton_histogram[singleton] << '\n';
        std::cout << "BEHAVIORAL_AUTOMORPHISM samples="
                  << automorphisms.size()
                  << " original_sum=" << original_aut
                  << " structural_sum=" << structural_aut
                  << " behavioral_sum=" << behavioral_aut
                  << " structural_larger=" << structural_larger
                  << " behavioral_larger=" << behavioral_larger
                  << " behavioral_larger_zero=" << behavioral_larger_zero
                  << " behavioral_larger_nonzero="
                  << behavioral_larger_nonzero
                  << " behavioral_excess_zero=" << behavioral_excess_zero
                  << " behavioral_excess_nonzero="
                  << behavioral_excess_nonzero
                  << " marginal_candidates=" << marginal_candidates
                  << " exact_tests=" << exact_tests
                  << " maximum_behavioral=" << maximum_behavioral
                  << " seconds=" << automorphism_seconds
                  << " exact=OK\n";
        std::cout << "BEHAVIORAL_ZERO_SOURCES selected=" << selected_zero
                  << " complement=" << complement_zero << " exact=OK\n";
        if (argc >= 3) {
            uint64_t samples_per_shard =
                argc == 4 ? std::stoull(argv[3]) : 65536;
            uint64_t sampled_records = 0, zero_records = 0;
            uint64_t selected_left_zero = 0, selected_right_zero = 0;
            uint64_t complement_left_zero = 0, complement_right_zero = 0;
            double zero_start = seconds_now();
            for (const std::string& path : split_paths(argv[2])) {
                std::vector<SampleRecord> records =
                    read_stride_sample(path, samples_per_shard);
                sampled_records += records.size();
#pragma omp parallel for reduction(+:zero_records,selected_left_zero,selected_right_zero,complement_left_zero,complement_right_zero)
                for (long long i = 0; i < (long long)records.size(); i++) {
                    uint32_t left = half_prefix(records[size_t(i)].key, 0);
                    uint32_t right =
                        half_prefix(records[size_t(i)].key, HALF_COLUMNS);
                    bool sl = zero_keys.count(canonical_half(left));
                    bool sr = zero_keys.count(canonical_half(right));
                    bool cl = zero_keys.count(canonical_half(~left));
                    bool cr = zero_keys.count(canonical_half(~right));
                    selected_left_zero += sl;
                    selected_right_zero += sr;
                    complement_left_zero += cl;
                    complement_right_zero += cr;
                    zero_records += sl || sr || cl || cr;
                }
            }
            std::cout << "BEHAVIORAL_ZERO_RECORDS records=" << sampled_records
                      << " zero_records=" << zero_records
                      << " ratio=" << double(zero_records) / sampled_records
                      << " selected_left=" << selected_left_zero
                      << " selected_right=" << selected_right_zero
                      << " complement_left=" << complement_left_zero
                      << " complement_right=" << complement_right_zero
                      << " seconds=" << seconds_now() - zero_start
                      << " exact=OK\n";
        }
        for (const auto& item : gcds)
            std::cout << "BEHAVIORAL_GCD value=" << item.first
                      << " distributions=" << item.second << '\n';
        unsigned shown = 0;
        for (const ExactClass& item : exact_classes) {
            if (item.uses <= 1) continue;
            std::cout << "BEHAVIORAL_DUPLICATE id="
                      << summaries[item.representative].semantic_id
                      << " uses=" << item.uses
                      << " entries=" << item.entry_count
                      << " half=" << item.representative / 2
                      << " component="
                      << ((item.representative & 1U) ? "complement" : "selected")
                      << '\n';
            if (++shown == 32) break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
