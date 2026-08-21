#pragma once

// Historical CPU prefix-layout construction retained only for legacy probes.

struct PackedLayouts {
    std::vector<uint64_t> direct_masks;
    std::vector<uint32_t> direct_weights;
    std::vector<PrefixEntry> prefix_entries;
    std::vector<PrefixBucket> buckets;
};

static std::vector<PrefixPair> build_prefix_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory,
    PackedLayouts& layouts) {
    struct WorkItem {
        CanonicalRef reference;
        uint64_t entry_offset;
        uint32_t entry_count;
        std::vector<PrefixBucket> buckets;
    };
    struct UnorderedEntry {
        uint16_t prefix;
        PrefixSuffix suffix;
        uint32_t weight;
    };
    constexpr size_t prefix_count = size_t(1) << (2 * PREFIX_PAIR_COUNT);

    std::vector<WorkItem> work(keys.size() * 2);
    uint64_t entry_count = 0;
    for (size_t index = 0; index < keys.size(); index++) {
        const RawCanonicalPair& raw = lookup_raw(factory, keys[index]);
        const CanonicalRef references[2] = {raw.selected, raw.complement};
        for (int complement = 0; complement < 2; complement++) {
            WorkItem& item = work[index * 2 + complement];
            item.reference = references[complement];
            item.entry_offset = entry_count;
            item.entry_count = factory.descriptors[item.reference.distribution].count;
            entry_count += item.entry_count;
        }
    }
    if (entry_count > uint64_t(UINT32_MAX) + 1) {
        throw std::overflow_error("prefix entries exceed 32-bit address space");
    }
    layouts.direct_masks.resize(entry_count);
    layouts.direct_weights.resize(entry_count);
    layouts.prefix_entries.resize(entry_count);

#pragma omp parallel for schedule(dynamic, 8)
    for (long long work_index = 0; work_index < (long long)work.size(); work_index++) {
        WorkItem& item = work[size_t(work_index)];
        const CanonicalDescriptor& source =
            factory.descriptors[item.reference.distribution];
        std::vector<UnorderedEntry> unordered(source.count);
        std::vector<uint32_t> counts(prefix_count);
        for (uint32_t index = 0; index < source.count; index++) {
            const Entry& canonical = factory.entries[source.offset + index];
            uint64_t mask = transform_pair_mask(canonical.mask, item.reference.row_map);
            uint32_t weight = uint32_t(canonical.weight);
            if (weight & PREFIX_ENTRY_ORBIT_TWO) {
                throw std::overflow_error(
                    "distribution weight collides with orbit marker");
            }
            uint32_t prefix_weight = prefix_entry_weight_key(weight, mask);
            uint16_t prefix;
            PrefixSuffix suffix;
            split_pair_mask(mask, prefix, suffix);
            size_t destination = size_t(item.entry_offset) + index;
            layouts.direct_masks[destination] = mask;
            layouts.direct_weights[destination] = weight;
            unordered[index] = UnorderedEntry{prefix, suffix, prefix_weight};
            counts[prefix]++;
        }
        std::vector<uint32_t> positions(prefix_count);
        uint32_t running = 0;
        for (size_t prefix = 0; prefix < prefix_count; prefix++) {
            positions[prefix] = running;
            if (!counts[prefix]) continue;
            item.buckets.push_back(
                PrefixBucket{uint32_t(item.entry_offset + running), counts[prefix],
                             uint16_t(prefix), 0});
            running += counts[prefix];
        }
        for (const UnorderedEntry& entry : unordered) {
            uint32_t destination = positions[entry.prefix]++;
            layouts.prefix_entries[size_t(item.entry_offset) + destination] =
                PrefixEntry{entry.suffix, entry.weight};
        }
    }

    std::vector<PrefixPair> result(keys.size());
    for (size_t index = 0; index < keys.size(); index++) {
        PrefixDistribution* distributions[2] = {&result[index].selected,
                                                &result[index].complement};
        for (int complement = 0; complement < 2; complement++) {
            WorkItem& item = work[index * 2 + complement];
            if (layouts.buckets.size() + item.buckets.size() >
                size_t(UINT32_MAX) + 1) {
                throw std::overflow_error("prefix buckets exceed 32-bit address space");
            }
            PrefixDistribution& distribution = *distributions[complement];
            distribution.direct_offset = item.entry_offset;
            distribution.entry_count = item.entry_count;
            distribution.bucket_offset = uint32_t(layouts.buckets.size());
            distribution.bucket_count = uint32_t(item.buckets.size());
            layouts.buckets.insert(layouts.buckets.end(), item.buckets.begin(),
                                   item.buckets.end());
        }
    }
    return result;
}
