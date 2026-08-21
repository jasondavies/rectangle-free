#pragma once

// Geometry-neutral canonical device cache and adapter to the shared direct
// weight-class builder.  Both 7x9 and 8x8 use this as the sole path from a
// CanonicalFactory to a production prefix layout.

struct CanonicalWeightSpan {
    uint32_t offset;
    uint32_t count;
};

struct ProductionCanonicalDevice {
    DeviceBuffer<uint64_t> masks;
    DeviceBuffer<uint8_t> weight_ordinals;
    DeviceBuffer<uint32_t> class_weights;
    DeviceBuffer<uint8_t> class_orbit_sizes;
    std::vector<CanonicalWeightSpan> weight_spans;
    size_t class_weight_count = 0;
    size_t maximum_distribution_weights = 0;
    double weight_table_seconds = 0;
    size_t entry_count = 0;
};

static DeviceBuffer<uint64_t> upload_production_masks(
    const std::vector<Entry>& entries) {
    std::vector<uint64_t> masks(entries.size());
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < (long long)entries.size(); index++) {
        masks[size_t(index)] = entries[size_t(index)].mask;
    }
    return upload_buffer(masks);
}

static ProductionCanonicalDevice upload_production_canonical(
    const CanonicalFactory& factory) {
    ProductionCanonicalDevice result;
    result.entry_count = factory.entries.size();
    double start = seconds_now();
    result.weight_spans.resize(factory.descriptors.size());
    std::vector<uint32_t> values;
    std::vector<uint8_t> orbit_sizes;
    std::vector<uint8_t> ordinals(factory.entries.size());
    for (size_t descriptor_index = 0;
         descriptor_index < factory.descriptors.size(); descriptor_index++) {
        const CanonicalDescriptor& descriptor =
            factory.descriptors[descriptor_index];
        CanonicalWeightSpan span{uint32_t(values.size()), 0};
        for (uint32_t index = 0; index < descriptor.count; index++) {
            uint64_t wide = factory.entries[descriptor.offset + index].weight;
            if (!wide || wide > UINT32_MAX) {
                throw std::overflow_error(
                    "canonical weight does not fit exact uint32_t table");
            }
            uint32_t weight = uint32_t(wide);
            uint8_t orbit_size = uint8_t(token_plane_orbit_size(
                factory.entries[descriptor.offset + index].mask));
            uint32_t ordinal = 0;
            while (ordinal < span.count &&
                   (values[span.offset + ordinal] != weight ||
                    orbit_sizes[span.offset + ordinal] != orbit_size)) {
                ordinal++;
            }
            if (ordinal == span.count) {
                if (span.count == 32) {
                    throw std::overflow_error(
                        "canonical distribution exceeds 32 orbit-aware weight classes");
                }
                values.push_back(weight);
                orbit_sizes.push_back(orbit_size);
                span.count++;
            }
            ordinals[descriptor.offset + index] = uint8_t(ordinal);
        }
        result.maximum_distribution_weights = std::max<size_t>(
            result.maximum_distribution_weights, span.count);
        result.weight_spans[descriptor_index] = span;
    }
    result.class_weight_count = values.size();
    result.masks = upload_production_masks(factory.entries);
    result.weight_ordinals = upload_buffer(ordinals);
    result.class_weights = upload_buffer(values);
    result.class_orbit_sizes = upload_buffer(orbit_sizes);
    result.weight_table_seconds = seconds_now() - start;
    return result;
}

static DeviceWeightClassLayout build_direct_weight_class_layout_from_refs(
    const std::vector<std::array<CanonicalRef, 2>>& references,
    const CanonicalFactory& factory,
    const ProductionCanonicalDevice& canonical,
    DirectWeightClassWorkspace& workspace) {
    if (!canonical.class_weights ||
        canonical.weight_spans.size() != factory.descriptors.size()) {
        throw std::runtime_error("direct weight table is unavailable");
    }
    double start = seconds_now();
    std::vector<DirectWeightBuildDesc> descriptions(references.size() * 2);
    uint64_t total_entries = 0;
    for (size_t index = 0; index < references.size(); index++) {
        for (int complement = 0; complement < 2; complement++) {
            const CanonicalRef& reference = references[index][complement];
            const CanonicalDescriptor& source =
                factory.descriptors[reference.distribution];
            CanonicalWeightSpan weights =
                canonical.weight_spans[reference.distribution];
            if (weights.count > WEIGHT_CLASS_HASH_SLOTS ||
                total_entries + source.count > uint64_t(UINT32_MAX) + 1) {
                throw std::overflow_error(
                    "direct grouped layout exceeds exact production bounds");
            }
            descriptions[index * 2 + complement] = DirectWeightBuildDesc{
                source.offset, reference.row_map, uint32_t(total_entries),
                source.count, 0, 0, weights.offset, weights.count, 0};
            total_entries += source.count;
        }
    }
    return build_direct_weight_class_layout_from_descriptions(
        std::move(descriptions), references.size(), total_entries,
        canonical.class_weights, canonical.class_orbit_sizes, workspace, start,
        [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense) {
            histogram_direct_weight_prefixes<<<unsigned(references.size() * 2),
                                               THREADS>>>(
                canonical.masks, device_descriptions, dense);
        },
        [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense,
            DirectBucketAux* bucket_aux, uint32_t* candidates,
            uint32_t* failure) {
            histogram_direct_weight_classes<<<
                unsigned(references.size() * 2), THREADS>>>(
                canonical.masks, canonical.weight_ordinals,
                device_descriptions, dense, bucket_aux, candidates, failure);
        },
        [&](DirectWeightBuildDesc* device_descriptions, uint32_t* dense,
            DirectBucketAux* bucket_aux, uint32_t* candidates,
            PrefixSuffix* suffixes, uint32_t* failure) {
            scatter_direct_weight_classes<<<unsigned(references.size() * 2),
                                            THREADS>>>(
                canonical.masks, canonical.weight_ordinals,
                device_descriptions, dense, bucket_aux, candidates, suffixes,
                failure);
        });
}
