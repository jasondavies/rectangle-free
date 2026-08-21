#pragma once

// Host/device index construction for the rejected suffix-bitplane join.
struct DeviceSuffixBitplanes {
    uint32_t* offsets = nullptr;
    uint64_t* planes = nullptr;
    unsigned long long* weight_sums = nullptr;
    uint64_t plane_words = 0;
    uint64_t indexed_buckets = 0;
    uint64_t indexed_entries = 0;
    double plan_seconds = 0;
    double build_seconds = 0;
    double total_seconds = 0;
};

static DeviceSuffixBitplanes build_suffix_bitplanes(
    const DevicePrefixLayout& layout) {
    if (layout.host_buckets.size() != layout.bucket_count ||
        layout.bucket_count > UINT32_MAX) {
        throw std::runtime_error("suffix bitplane host buckets are unavailable");
    }
    double total_start = seconds_now();
    DeviceSuffixBitplanes result;
    std::vector<uint32_t> offsets(layout.bucket_count, UINT32_MAX);
    for (size_t index = 0; index < layout.host_buckets.size(); index++) {
        const PrefixBucket& bucket = layout.host_buckets[index];
        if (bucket.count < SUFFIX_BITPLANE_MIN_COUNT) continue;
        uint64_t words = (uint64_t(bucket.count) + 63) / 64;
        if (result.plane_words + 32 * words > UINT32_MAX) {
            throw std::overflow_error("suffix bitplane offsets exceed uint32_t");
        }
        offsets[index] = uint32_t(result.plane_words);
        result.plane_words += 32 * words;
        result.indexed_buckets++;
        result.indexed_entries += bucket.count;
    }
    result.plan_seconds = seconds_now() - total_start;
    result.offsets = upload_vector(offsets);
    CUDA_CHECK(cudaMalloc(&result.weight_sums,
                          layout.bucket_count *
                              sizeof(unsigned long long)));
    if (result.plane_words) {
        CUDA_CHECK(cudaMalloc(&result.planes,
                              result.plane_words * sizeof(uint64_t)));
    }
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    build_prefix_bucket_bitplanes<<<unsigned(layout.bucket_count), THREADS>>>(
        layout.entries, layout.buckets, result.offsets,
        uint32_t(layout.bucket_count), result.planes, result.weight_sums);
    CUDA_CHECK(cudaGetLastError());
    result.build_seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    result.total_seconds = seconds_now() - total_start;
    return result;
}

static void free_suffix_bitplanes(DeviceSuffixBitplanes& index) {
    CUDA_CHECK(cudaFree(index.weight_sums));
    CUDA_CHECK(cudaFree(index.planes));
    CUDA_CHECK(cudaFree(index.offsets));
    index.weight_sums = nullptr;
    index.planes = nullptr;
    index.offsets = nullptr;
}
