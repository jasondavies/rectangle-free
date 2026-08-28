#define GRID_ROWS 6
#define GRID_COLUMNS 12
#define LEFT_COLUMNS 6
#define RIGHT_COLUMNS 6
#define ORBIT_ROW_BITS 12
#define ORBIT_MAGIC "R6W1201"
#define TWCOLOUR_WIDE_ORBIT_RECORD 1

#include "../../src/gpu/twocolour_gpu_common.cuh"

#include <unordered_map>
#include <unordered_set>

namespace {

struct Counts {
    uint32_t selected = 0;
    uint32_t complement = 0;
};

PrefixKey extract(U128 key, uint16_t columns) {
    PrefixKey result = 0;
    for (int row = 0; row < ROWS; ++row) {
        const unsigned row_shift = COLUMNS * (ROWS - 1U - row);
        unsigned output = 0;
        PrefixKey pattern = 0;
        for (unsigned column = 0; column < COLUMNS; ++column) {
            if (!(columns & (uint16_t(1) << column))) continue;
            pattern |= PrefixKey((key >> (row_shift + column)) & 1U)
                       << output++;
        }
        result = (result << 6) | pattern;
    }
    return result;
}

std::vector<U128> read_sample(const char* path, uint64_t wanted) {
    std::ifstream input(path, std::ios::binary);
    char magic[8];
    uint32_t columns = 0;
    uint64_t count = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || std::memcmp(magic, ORBIT_MAGIC, 7) || columns != COLUMNS ||
        !wanted || wanted > count)
        throw std::runtime_error("invalid 6x12 sample");
    std::vector<U128> result(size_t(wanted), 0);
    for (uint64_t sample = 0; sample < wanted; ++sample) {
        const uint64_t index =
            (U128(sample * 2 + 1) * count) / (U128(2) * wanted);
        OrbitRecord record{};
        input.seekg(std::streamoff(20 + index * sizeof(record)));
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        result[size_t(sample)] =
            (U128(record.meta & WIDE_ORBIT_KEY_MASK) << 64) | record.low;
    }
    if (!input) throw std::runtime_error("truncated 6x12 sample");
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "Usage: %s SAMPLE_6X12.orbits [SAMPLES=64]\n",
                     argv[0]);
        return 2;
    }
    const uint64_t wanted = argc == 3 ? std::strtoull(argv[2], nullptr, 10)
                                      : 64;
    initialise_tables();
    std::vector<U128> records = read_sample(argv[1], wanted);
    std::vector<uint16_t> cuts;
    for (uint16_t mask = 0; mask < (1U << COLUMNS); ++mask)
        if ((mask & 1U) && __builtin_popcount(mask) == 6) cuts.push_back(mask);
    if (cuts.size() != 462) throw std::logic_error("bad 6+6 cut count");

    const uint16_t all_columns = (1U << COLUMNS) - 1U;
    std::vector<PrefixKey> canonical_keys;
    canonical_keys.reserve(records.size() * cuts.size() * 2);
    std::vector<PrefixKey> raw_halves(records.size() * cuts.size() * 2);
    double canonical_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long record = 0; record < (long long)records.size(); ++record) {
        for (size_t cut = 0; cut < cuts.size(); ++cut) {
            const PrefixKey first = extract(records[size_t(record)], cuts[cut]);
            const PrefixKey second =
                extract(records[size_t(record)], all_columns ^ cuts[cut]);
            const size_t offset = (size_t(record) * cuts.size() + cut) * 2;
            raw_halves[offset] = first;
            raw_halves[offset + 1] = second;
        }
    }
    std::vector<PrefixKey> canonical_halves(raw_halves.size());
#pragma omp parallel for schedule(dynamic, 16)
    for (long long index = 0; index < (long long)raw_halves.size(); ++index)
        canonical_halves[size_t(index)] =
            canonical_prefix(raw_halves[size_t(index)], 6).key;
    canonical_keys = canonical_halves;
    std::sort(canonical_keys.begin(), canonical_keys.end());
    canonical_keys.erase(std::unique(canonical_keys.begin(), canonical_keys.end()),
                         canonical_keys.end());
    const double canonical_seconds = seconds_now() - canonical_start;

    std::vector<Counts> counts(canonical_keys.size());
    double distribution_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)canonical_keys.size(); ++index) {
        counts[size_t(index)] = Counts{
            uint32_t(quotient_token_planes(build_distribution(
                         canonical_keys[size_t(index)], 6, false)).entries.size()),
            uint32_t(quotient_token_planes(build_distribution(
                         canonical_keys[size_t(index)], 6, true)).entries.size())};
    }
    const double distribution_seconds = seconds_now() - distribution_start;
    std::unordered_map<PrefixKey, uint32_t> index;
    index.reserve(canonical_keys.size() * 2);
    for (size_t item = 0; item < canonical_keys.size(); ++item)
        index.emplace(canonical_keys[item], uint32_t(item));

    std::vector<std::vector<uint64_t>> selected_costs(
        cuts.size(), std::vector<uint64_t>(records.size()));
    std::vector<std::vector<uint64_t>> complement_costs(
        cuts.size(), std::vector<uint64_t>(records.size()));
    std::vector<std::vector<PrefixKey>> execution_left(
        cuts.size(), std::vector<PrefixKey>(records.size()));
#pragma omp parallel for schedule(static)
    for (long long cut = 0; cut < (long long)cuts.size(); ++cut) {
        for (size_t record = 0; record < records.size(); ++record) {
            const size_t offset = (record * cuts.size() + size_t(cut)) * 2;
            const Counts& first = counts[index.at(canonical_halves[offset])];
            const Counts& second = counts[index.at(canonical_halves[offset + 1])];
            selected_costs[size_t(cut)][record] =
                uint64_t(first.selected) * second.selected;
            complement_costs[size_t(cut)][record] =
                uint64_t(first.complement) * second.complement;
            execution_left[size_t(cut)][record] =
                uint64_t(first.selected) + first.complement <=
                        uint64_t(second.selected) + second.complement
                    ? raw_halves[offset] : raw_halves[offset + 1];
        }
    }
    auto total_for = [&](size_t cut) {
        U128 result = 0;
        for (size_t record = 0; record < records.size(); ++record)
            result += selected_costs[cut][record] +
                      complement_costs[cut][record];
        return result;
    };
    const auto baseline_iterator =
        std::find(cuts.begin(), cuts.end(), uint16_t(0x03f));
    const size_t baseline = size_t(baseline_iterator - cuts.begin());
    const U128 baseline_cost = total_for(baseline);
    size_t best_fixed = 0;
    for (size_t cut = 1; cut < cuts.size(); ++cut)
        if (total_for(cut) < total_for(best_fixed)) best_fixed = cut;
    std::vector<uint64_t> oracle(records.size(), UINT64_MAX);
    std::vector<uint64_t> selected_oracle(records.size(), UINT64_MAX);
    std::vector<uint64_t> complement_oracle(records.size(), UINT64_MAX);
    std::vector<size_t> oracle_cut(records.size());
    for (size_t cut = 0; cut < cuts.size(); ++cut) {
        for (size_t record = 0; record < records.size(); ++record) {
            const uint64_t combined = selected_costs[cut][record] +
                                      complement_costs[cut][record];
            if (combined < oracle[record]) {
                oracle[record] = combined;
                oracle_cut[record] = cut;
            }
            selected_oracle[record] =
                std::min(selected_oracle[record], selected_costs[cut][record]);
            complement_oracle[record] = std::min(
                complement_oracle[record], complement_costs[cut][record]);
        }
    }
    const U128 oracle_cost =
        std::accumulate(oracle.begin(), oracle.end(), U128(0));
    U128 independent_oracle_cost = 0;
    for (size_t record = 0; record < records.size(); ++record)
        independent_oracle_cost += selected_oracle[record] +
                                   complement_oracle[record];

    std::vector<uint64_t> same_portfolio(records.size(), UINT64_MAX);
    std::vector<size_t> same_cuts;
    for (unsigned round = 0; round < 16; ++round) {
        size_t choice = 0;
        U128 choice_cost = U128(-1);
        for (size_t cut = 0; cut < cuts.size(); ++cut) {
            U128 candidate = 0;
            for (size_t record = 0; record < records.size(); ++record) {
                const uint64_t combined = selected_costs[cut][record] +
                                          complement_costs[cut][record];
                candidate += std::min(same_portfolio[record], combined);
            }
            if (candidate < choice_cost) choice_cost = candidate, choice = cut;
        }
        same_cuts.push_back(choice);
        for (size_t record = 0; record < records.size(); ++record) {
            const uint64_t combined = selected_costs[choice][record] +
                                      complement_costs[choice][record];
            same_portfolio[record] =
                std::min(same_portfolio[record], combined);
        }
        std::printf("COLUMN_SPLIT_6X12_PORTFOLIO size=%u added=0x%03x "
                    "ratio=%.12f\n", round + 1, cuts[choice],
                    double(choice_cost) / double(baseline_cost));
    }

    // One shared menu is allowed, but selected and complement contractions
    // independently choose their cheapest cut from that menu.
    std::vector<uint64_t> selected_portfolio(records.size(), UINT64_MAX);
    std::vector<uint64_t> complement_portfolio(records.size(), UINT64_MAX);
    std::vector<size_t> independent_cuts;
    for (unsigned round = 0; round < 16; ++round) {
        size_t choice = 0;
        U128 choice_cost = U128(-1);
        for (size_t cut = 0; cut < cuts.size(); ++cut) {
            U128 candidate = 0;
            for (size_t record = 0; record < records.size(); ++record) {
                candidate += std::min(selected_portfolio[record],
                                      selected_costs[cut][record]);
                candidate += std::min(complement_portfolio[record],
                                      complement_costs[cut][record]);
            }
            if (candidate < choice_cost) choice_cost = candidate, choice = cut;
        }
        independent_cuts.push_back(choice);
        for (size_t record = 0; record < records.size(); ++record) {
            selected_portfolio[record] = std::min(
                selected_portfolio[record], selected_costs[choice][record]);
            complement_portfolio[record] = std::min(
                complement_portfolio[record], complement_costs[choice][record]);
        }
        std::printf(
            "COLUMN_SPLIT_6X12_INDEPENDENT_PORTFOLIO size=%u added=0x%03x "
            "ratio=%.12f\n", round + 1, cuts[choice],
            double(choice_cost) / double(baseline_cost));
    }

    auto component_layout_stats = [&](size_t menu_size) {
        std::unordered_map<PrefixKey, uint32_t> selected_left;
        std::unordered_map<PrefixKey, uint32_t> selected_right;
        std::unordered_map<PrefixKey, uint32_t> complement_left;
        std::unordered_map<PrefixKey, uint32_t> complement_right;
        for (size_t record = 0; record < records.size(); ++record) {
            size_t selected_cut = independent_cuts[0];
            size_t complement_cut = independent_cuts[0];
            for (size_t item = 1; item < menu_size; ++item) {
                const size_t cut = independent_cuts[item];
                if (selected_costs[cut][record] <
                    selected_costs[selected_cut][record])
                    selected_cut = cut;
                if (complement_costs[cut][record] <
                    complement_costs[complement_cut][record])
                    complement_cut = cut;
            }
            auto add_component = [&](size_t cut, bool complement,
                                     auto& left, auto& right) {
                const size_t offset = (record * cuts.size() + cut) * 2;
                const Counts& first = counts[index.at(canonical_halves[offset])];
                const Counts& second =
                    counts[index.at(canonical_halves[offset + 1])];
                const uint32_t first_entries =
                    complement ? first.complement : first.selected;
                const uint32_t second_entries =
                    complement ? second.complement : second.selected;
                if (first_entries <= second_entries) {
                    left.emplace(raw_halves[offset], first_entries);
                    right.emplace(raw_halves[offset + 1], second_entries);
                } else {
                    left.emplace(raw_halves[offset + 1], second_entries);
                    right.emplace(raw_halves[offset], first_entries);
                }
            };
            add_component(selected_cut, false, selected_left, selected_right);
            add_component(complement_cut, true, complement_left,
                          complement_right);
        }
        auto entries = [](const auto& layouts) {
            U128 result = 0;
            for (const auto& item : layouts) result += item.second;
            return result;
        };
        const U128 left_entries = entries(selected_left) +
                                  entries(complement_left);
        const U128 right_entries = entries(selected_right) +
                                   entries(complement_right);
        std::printf(
            "COLUMN_SPLIT_6X12_INDEPENDENT_LAYOUT size=%zu "
            "selected_left=%zu complement_left=%zu selected_right=%zu "
            "complement_right=%zu left_entries=%s right_entries=%s\n",
            menu_size, selected_left.size(), complement_left.size(),
            selected_right.size(), complement_right.size(),
            u128_string(left_entries).c_str(),
            u128_string(right_entries).c_str());
    };
    component_layout_stats(1);
    component_layout_stats(4);
    component_layout_stats(8);
    component_layout_stats(16);
    std::unordered_set<PrefixKey> fixed_left;
    std::unordered_set<PrefixKey> oracle_left;
    for (size_t record = 0; record < records.size(); ++record) {
        fixed_left.insert(execution_left[baseline][record]);
        oracle_left.insert(execution_left[oracle_cut[record]][record]);
    }
    std::printf(
        "COLUMN_SPLIT_6X12 samples=%zu cuts=%zu canonical_halves=%zu "
        "baseline=0x%03x best_fixed=0x%03x best_fixed_ratio=%.12f "
        "oracle_ratio=%.12f independent_oracle_ratio=%.12f "
        "fixed_left=%zu oracle_left=%zu "
        "canonical_seconds=%.6f distribution_seconds=%.6f OK\n",
        records.size(), cuts.size(), canonical_keys.size(), cuts[baseline],
        cuts[best_fixed], double(total_for(best_fixed)) / double(baseline_cost),
        double(oracle_cost) / double(baseline_cost),
        double(independent_oracle_cost) / double(baseline_cost),
        fixed_left.size(), oracle_left.size(), canonical_seconds,
        distribution_seconds);
    return 0;
}
