#pragma once

// Seven-row device layout and packed 7x5 canonical-cache pipeline.
// Generic prefix/suffix algebra lives in twocolour_prefix_algebra.cuh.

#include "twocolour_prefix_algebra.cuh"

#include "gpu_memory_policy.hpp"
#include "../common/sha256.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#if GRID_ROWS != 7
#error "the packed prefix core is specific to seven-row production"
#endif

constexpr uint64_t PACKED_STREAM_STAGING_BYTES = UINT64_C(1024) << 20;
constexpr size_t PACKED_STREAM_DESCRIPTION_CAP = 8192;
#ifndef PACKED_PREFETCH_BYTES
#define PACKED_PREFETCH_BYTES (UINT64_C(4608) << 20)
#endif

struct PrefixEntry {
    PrefixSuffix suffix;
    uint32_t weight;
};

// The compact left-layout conversion carries the size-two support-orbit marker
// in the high weight bit and removes it when WeightClassMeta is emitted.
constexpr uint32_t PREFIX_ENTRY_ORBIT_TWO = UINT32_C(1) << 31;

static __host__ __device__ uint32_t prefix_entry_weight_key(
    uint32_t weight, uint64_t mask) {
    return weight | (token_plane_orbit_size(mask) == 2
                         ? PREFIX_ENTRY_ORBIT_TWO : 0U);
}

struct GpuBucketBuildDesc {
    uint64_t source_offset;
    uint64_t row_map;
    uint32_t destination_offset;
    uint32_t count;
};

static_assert(sizeof(PrefixEntry) == 8,
              "seven-row prefix entries must remain compact");
static_assert(sizeof(GpuBucketBuildDesc) == 24,
              "GPU bucket build descriptor must be compact");

__global__ void histogram_prefix_distributions(
    const Entry* __restrict__ canonical,
    const GpuBucketBuildDesc* __restrict__ descriptions,
    uint32_t* __restrict__ dense_counts) {
    __shared__ uint32_t histogram[PREFIX_BUCKET_COUNT];
    for (uint32_t prefix = threadIdx.x; prefix < PREFIX_BUCKET_COUNT;
         prefix += blockDim.x) {
        histogram[prefix] = 0;
    }
    __syncthreads();
    const GpuBucketBuildDesc description = descriptions[blockIdx.x];
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        uint64_t mask = transform_pair_mask(
            canonical[description.source_offset + index].mask, description.row_map);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        atomicAdd(&histogram[prefix], 1U);
    }
    __syncthreads();
    for (uint32_t prefix = threadIdx.x; prefix < PREFIX_BUCKET_COUNT;
         prefix += blockDim.x) {
        dense_counts[size_t(blockIdx.x) * PREFIX_BUCKET_COUNT + prefix] =
            histogram[prefix];
    }
}

__global__ void scatter_prefix_distributions(
    const Entry* __restrict__ canonical,
    const GpuBucketBuildDesc* __restrict__ descriptions,
    const uint32_t* __restrict__ dense_offsets,
    PrefixEntry* __restrict__ output) {
    __shared__ uint32_t positions[PREFIX_BUCKET_COUNT];
    for (uint32_t prefix = threadIdx.x; prefix < PREFIX_BUCKET_COUNT;
         prefix += blockDim.x) {
        positions[prefix] =
            dense_offsets[size_t(blockIdx.x) * PREFIX_BUCKET_COUNT + prefix];
    }
    __syncthreads();
    const GpuBucketBuildDesc description = descriptions[blockIdx.x];
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        const Entry canonical_entry = canonical[description.source_offset + index];
        uint64_t mask = transform_pair_mask(canonical_entry.mask, description.row_map);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        uint32_t destination = description.destination_offset +
                               atomicAdd(&positions[prefix], 1U);
        uint32_t weight = uint32_t(canonical_entry.weight);
        weight = prefix_entry_weight_key(weight, mask);
        output[destination] = PrefixEntry{suffix, weight};
    }
}
constexpr uint64_t PACKED_CANONICAL_MASK = (UINT64_C(1) << (2 * PAIRS)) - 1;
constexpr unsigned PACKED_CANONICAL_WEIGHT_SHIFT = 2 * PAIRS;
static_assert(2 * PAIRS == 42, "packed canonical entries require seven rows");







struct DevicePrefixLayout {
    PrefixEntry* entries = nullptr;
    PrefixBucket* buckets = nullptr;
    std::vector<PrefixPair> pairs;
    size_t entry_count = 0;
    size_t bucket_count = 0;
    double histogram_seconds = 0;
    double scatter_seconds = 0;
    double metadata_seconds = 0;
    double plan_seconds = 0;
    double upload_seconds = 0;
    double source_gather_seconds = 0;
    double source_upload_seconds = 0;
    uint64_t source_entries = 0;
    uint64_t source_chunks = 0;
    double total_seconds = 0;
};

static DevicePrefixLayout build_device_prefix_layout(
    const std::vector<PrefixKey>& keys, const CanonicalFactory& factory) {
    double total_start = seconds_now();
    DevicePrefixLayout result;
    result.pairs.resize(keys.size());
    std::vector<GpuBucketBuildDesc> descriptions(keys.size() * 2);
    uint64_t total_entries = 0;
    for (size_t index = 0; index < keys.size(); index++) {
        const RawCanonicalPair& raw = lookup_raw(factory, keys[index]);
        const CanonicalRef references[2] = {raw.selected, raw.complement};
        for (int complement = 0; complement < 2; complement++) {
            const CanonicalRef& reference = references[complement];
            const CanonicalDescriptor& source =
                factory.descriptors[reference.distribution];
            if (total_entries + source.count > uint64_t(UINT32_MAX) + 1) {
                throw std::overflow_error("GPU prefix layout exceeds 32-bit offsets");
            }
            descriptions[index * 2 + complement] =
                GpuBucketBuildDesc{source.offset, reference.row_map,
                                   uint32_t(total_entries), source.count};
            total_entries += source.count;
        }
    }
    result.entry_count = size_t(total_entries);
    result.plan_seconds = seconds_now() - total_start;

    double upload_start = seconds_now();
    Entry* device_canonical = upload_vector(factory.entries);
    GpuBucketBuildDesc* device_descriptions = upload_vector(descriptions);
    size_t dense_count = descriptions.size() * size_t(PREFIX_BUCKET_COUNT);
    uint32_t* device_dense = nullptr;
    CUDA_CHECK(cudaMalloc(&device_dense, dense_count * sizeof(uint32_t)));
    result.upload_seconds = seconds_now() - upload_start;
    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));
    CUDA_CHECK(cudaEventRecord(event_start));
    histogram_prefix_distributions<<<unsigned(descriptions.size()), THREADS>>>(
        device_canonical, device_descriptions, device_dense);
    CUDA_CHECK(cudaGetLastError());
    result.histogram_seconds = elapsed_kernel(event_start, event_end);

    double metadata_start = seconds_now();
    std::vector<uint32_t> dense_counts(dense_count);
    CUDA_CHECK(cudaMemcpy(dense_counts.data(), device_dense,
                          dense_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> dense_offsets(dense_count);
    std::vector<PrefixBucket> buckets;
    for (size_t description_index = 0; description_index < descriptions.size();
         description_index++) {
        const GpuBucketBuildDesc& description = descriptions[description_index];
        PrefixDistribution& distribution = description_index & 1
                                               ? result.pairs[description_index / 2].complement
                                               : result.pairs[description_index / 2].selected;
        distribution.direct_offset = description.destination_offset;
        distribution.entry_count = description.count;
        if (buckets.size() > UINT32_MAX) {
            throw std::overflow_error("GPU prefix bucket offset exceeds uint32_t");
        }
        distribution.bucket_offset = uint32_t(buckets.size());
        uint32_t running = 0;
        size_t dense_base = description_index * size_t(PREFIX_BUCKET_COUNT);
        for (uint32_t prefix = 0; prefix < PREFIX_BUCKET_COUNT; prefix++) {
            dense_offsets[dense_base + prefix] = running;
            uint32_t count = dense_counts[dense_base + prefix];
            if (!count) continue;
            buckets.push_back(PrefixBucket{description.destination_offset + running,
                                            count, uint16_t(prefix), 0});
            running += count;
        }
        if (running != description.count) {
            throw std::runtime_error("GPU prefix histogram count mismatch");
        }
        distribution.bucket_count = uint32_t(buckets.size() -
                                              distribution.bucket_offset);
    }
    result.bucket_count = buckets.size();
    CUDA_CHECK(cudaMemcpy(device_dense, dense_offsets.data(),
                          dense_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    result.buckets = upload_vector(buckets);
    result.metadata_seconds = seconds_now() - metadata_start;

    CUDA_CHECK(cudaMalloc(&result.entries, result.entry_count * sizeof(PrefixEntry)));
    CUDA_CHECK(cudaEventRecord(event_start));
    scatter_prefix_distributions<<<unsigned(descriptions.size()), THREADS>>>(
        device_canonical, device_descriptions, device_dense, result.entries);
    CUDA_CHECK(cudaGetLastError());
    result.scatter_seconds = elapsed_kernel(event_start, event_end);
    CUDA_CHECK(cudaEventDestroy(event_end));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaFree(device_dense));
    CUDA_CHECK(cudaFree(device_descriptions));
    CUDA_CHECK(cudaFree(device_canonical));
    result.total_seconds = seconds_now() - total_start;
    return result;
}
static void free_device_prefix_layout(DevicePrefixLayout& layout) {
    CUDA_CHECK(cudaFree(layout.buckets));
    CUDA_CHECK(cudaFree(layout.entries));
    layout.buckets = nullptr;
    layout.entries = nullptr;
}
struct PackedWeightSpan {
    uint32_t offset = 0;
    uint32_t count = 0;
};

class ReadOnlyMapping {
  public:
    ReadOnlyMapping() = default;
    ~ReadOnlyMapping() { reset(); }
    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping(ReadOnlyMapping&& other) noexcept { swap(other); }
    ReadOnlyMapping& operator=(ReadOnlyMapping&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void open_file(const char* path) {
        reset();
        descriptor_ = ::open(path, O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error(std::string("cannot open packed cache: ") +
                                     std::strerror(errno));
        }
        struct stat status {};
        if (fstat(descriptor_, &status) || status.st_size <= 0) {
            int saved = errno;
            reset();
            throw std::runtime_error(std::string("cannot stat packed cache: ") +
                                     std::strerror(saved));
        }
        size_ = size_t(status.st_size);
        data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, descriptor_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            int saved = errno;
            reset();
            throw std::runtime_error(std::string("cannot map packed cache: ") +
                                     std::strerror(saved));
        }
    }
    void reset() {
        if (data_) munmap(data_, size_);
        if (descriptor_ >= 0) ::close(descriptor_);
        data_ = nullptr;
        descriptor_ = -1;
        size_ = 0;
    }
    const uint8_t* bytes() const {
        return static_cast<const uint8_t*>(data_);
    }
    size_t size() const { return size_; }
    explicit operator bool() const { return data_ != nullptr; }

  private:
    void swap(ReadOnlyMapping& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(descriptor_, other.descriptor_);
        std::swap(size_, other.size_);
    }
    void* data_ = nullptr;
    int descriptor_ = -1;
    size_t size_ = 0;
};

constexpr uint32_t PACKED_CACHE_ENDIAN = UINT32_C(0x01020304);
constexpr uint32_t PACKED_CACHE_TOKEN_PLANE_QUOTIENT = 1;
constexpr uint32_t PACKED_CACHE_CHECKSUM_BLOCK_BYTES = 4U << 20;
constexpr size_t PACKED_CACHE_SHA256_HEX_BYTES = 64;
constexpr size_t PACKED_CACHE_HEADER_BYTES = 256;
constexpr size_t PACKED_CACHE_DISTRIBUTIONS = 136758;
constexpr size_t PACKED_CACHE_ENTRIES = UINT64_C(2370316739);

struct PackedCacheArtifactHeader {
    char magic[8];
    uint32_t header_bytes;
    uint32_t endian;
    uint32_t rows;
    uint32_t columns;
    uint32_t token_plane_quotient;
    uint32_t checksum_block_bytes;
    uint64_t distribution_count;
    uint64_t entry_count;
    uint64_t class_count;
    uint64_t catalog_bytes;
    uint64_t checksum_count;
    uint64_t entry_offset;
    uint64_t file_bytes;
    char source_sha256[PACKED_CACHE_SHA256_HEX_BYTES];
    char artifact_sha256[PACKED_CACHE_SHA256_HEX_BYTES];
    uint8_t reserved[40];
};
static_assert(sizeof(PackedCacheArtifactHeader) == PACKED_CACHE_HEADER_BYTES,
              "packed cache header must remain stable");

using PackedCacheChecksum =
    std::array<char, PACKED_CACHE_SHA256_HEX_BYTES>;

struct PackedUniversalCache {
    std::unique_ptr<uint64_t[]> owned_entries;
    ReadOnlyMapping mapping;
    const uint64_t* host_entries = nullptr;
    PinnedBuffer<uint64_t> host_prefetch;
    DeviceBuffer<uint64_t> device_prefetch;
    DeviceBuffer<uint64_t> device_entries;
    size_t prefetch_bytes = 0;
    size_t memory_reserve_bytes = 0;
    bool device_resident = false;
    CudaStream prefetch_stream;
    std::vector<PrefixKey> keys;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> counts;
    std::vector<PackedWeightSpan> weight_spans;
    std::vector<uint32_t> class_weights;
    std::vector<uint8_t> class_orbit_sizes;
    std::vector<PackedCacheChecksum> entry_checksums;
    std::unique_ptr<std::atomic<uint8_t>[]> checksum_states;
    size_t checksum_count = 0;
    DeviceBuffer<uint32_t> device_class_weights;
    DeviceBuffer<uint8_t> device_class_orbit_sizes;
    size_t maximum_distribution_weights = 0;
    size_t entry_count = 0;
    size_t bytes = 0;
    size_t free_bytes_before = 0;
    size_t free_bytes_after = 0;
    double count_seconds = 0;
    double allocation_seconds = 0;
    double pack_seconds = 0;
    double upload_seconds = 0;
    double validation_seconds = 0;
    double total_seconds = 0;
    std::string source_sha256;
    std::string artifact_sha256;
};

static bool validate_packed_cache_range(const PackedUniversalCache& cache,
                                        uint64_t entry_offset,
                                        uint64_t entry_count);
static bool validate_packed_cache_all(const PackedUniversalCache& cache);


static PrefixKey compact_orbit_7x5_key(uint64_t orbit_key) {
    PrefixKey result = 0;
    for (int row = 0; row < ROWS; row++) {
        unsigned shift = 8 * (ROWS - 1U - row);
        result = (result << 5) | PrefixKey((orbit_key >> shift) & 31U);
    }
    return result;
}

static void initialise_packed_cache_device(PackedUniversalCache& cache) {
    size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&cache.free_bytes_before, &total_device_bytes));
    double upload_start = seconds_now();
    cache.memory_reserve_bytes = gpu_memory_policy::reserve_bytes(
        size_t(4) << 30);
    // A maximum 32-bit-offset right layout occupies about 24 GiB at the
    // production six-byte estimate.  Eight more GiB cover persistent left
    // layouts and builder metadata.  On a 96-GiB Blackwell this selects full
    // residency; on a 48-GiB L40S it retains the measured streamed policy.
    constexpr size_t resident_recurring_headroom = size_t(32) << 30;
    cache.device_resident = gpu_memory_policy::prefer_resident_cache(
        cache.free_bytes_before, cache.bytes, cache.memory_reserve_bytes,
        resident_recurring_headroom);
    if (cache.device_resident) {
        if (cache.checksum_count && !validate_packed_cache_all(cache)) {
            throw std::runtime_error(
                "resident packed cache entry checksum mismatch");
        }
        cache.device_entries.reserve(cache.entry_count);
        CUDA_CHECK(cudaMemcpy(cache.device_entries.get(), cache.host_entries,
                              cache.bytes, cudaMemcpyHostToDevice));
    } else {
        cache.prefetch_bytes = size_t(PACKED_PREFETCH_BYTES);
        if (!cache.prefetch_bytes ||
            cache.prefetch_bytes % sizeof(uint64_t)) {
            throw std::runtime_error("invalid packed prefetch size");
        }
        cache.host_prefetch.reserve(cache.prefetch_bytes / sizeof(uint64_t));
        cache.device_prefetch.reserve(cache.prefetch_bytes / sizeof(uint64_t));
        cache.prefetch_stream.create(cudaStreamNonBlocking);
    }
    cache.device_class_weights = upload_buffer(cache.class_weights);
    cache.device_class_orbit_sizes = upload_buffer(cache.class_orbit_sizes);
    cache.upload_seconds = seconds_now() - upload_start;
    CUDA_CHECK(cudaMemGetInfo(&cache.free_bytes_after, &total_device_bytes));
}

static PackedUniversalCache build_packed_universal_cache_from_orbits(
    const char* path, bool initialise_device) {
    double total_start = seconds_now();
    PackedUniversalCache cache;
    cache.source_sha256 = sha256_file(path);

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open universal 7x5 orbit file");
    char magic[8];
    uint32_t columns = 0;
    uint64_t records = 0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    input.read(reinterpret_cast<char*>(&records), sizeof(records));
    if (!input || std::memcmp(magic, "R7ORB01", 7) || columns != 5 ||
        records != PACKED_CACHE_DISTRIBUTIONS) {
        throw std::runtime_error("invalid complete 7x5 orbit file");
    }
    cache.keys.reserve(records);
    U128 labelled_weight = 0;
    for (uint64_t index = 0; index < records; index++) {
        OrbitRecord record{};
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        if (!input) throw std::runtime_error("truncated complete 7x5 orbit file");
        cache.keys.push_back(
            canonical_prefix(compact_orbit_7x5_key(record.key), 5).key);
        labelled_weight += record.weight;
    }
    char trailing;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("trailing complete 7x5 orbit data");
    }
    std::sort(cache.keys.begin(), cache.keys.end());
    if (std::adjacent_find(cache.keys.begin(), cache.keys.end()) !=
            cache.keys.end() ||
        labelled_weight != (U128(1) << 35)) {
        throw std::runtime_error("universal 7x5 orbit validation failed");
    }

    cache.counts.resize(cache.keys.size());
    cache.weight_spans.resize(cache.keys.size());
    std::vector<std::array<uint32_t, 32>> distribution_weights(
        cache.keys.size());
    std::vector<std::array<uint8_t, 32>> distribution_orbit_sizes(
        cache.keys.size());
    std::vector<uint8_t> distribution_weight_counts(cache.keys.size());
    double count_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)cache.keys.size(); index++) {
        Distribution distribution =
            build_distribution(cache.keys[size_t(index)], 5, false);
        distribution = quotient_token_planes(std::move(distribution));
        cache.counts[size_t(index)] = uint32_t(distribution.entries.size());
        uint8_t weight_count = 0;
        for (const Entry& entry : distribution.entries) {
            if (entry.weight > UINT32_MAX) {
                weight_count = UINT8_MAX;
                break;
            }
            uint8_t ordinal = 0;
            while (ordinal < weight_count &&
                   (distribution_weights[size_t(index)][ordinal] !=
                        entry.weight
                    || distribution_orbit_sizes[size_t(index)][ordinal] !=
                           token_plane_orbit_size(entry.mask)
                   )) ordinal++;
            if (ordinal == weight_count) {
                if (weight_count == 32) {
                    weight_count = UINT8_MAX;
                    break;
                }
                distribution_weights[size_t(index)][weight_count++] =
                    uint32_t(entry.weight);
                distribution_orbit_sizes[size_t(index)][weight_count - 1] =
                    uint8_t(token_plane_orbit_size(entry.mask));
            }
        }
        distribution_weight_counts[size_t(index)] = weight_count;
    }
    cache.count_seconds = seconds_now() - count_start;
    for (size_t index = 0; index < cache.keys.size(); index++) {
        uint32_t count = distribution_weight_counts[index];
        if (count == UINT8_MAX) {
            throw std::overflow_error(
                "packed distribution exceeds 32 weight classes");
        }
        cache.weight_spans[index] =
            PackedWeightSpan{uint32_t(cache.class_weights.size()), count};
        cache.maximum_distribution_weights = std::max<size_t>(
            cache.maximum_distribution_weights, count);
        cache.class_weights.insert(cache.class_weights.end(),
                                   distribution_weights[index].begin(),
                                   distribution_weights[index].begin() + count);
        cache.class_orbit_sizes.insert(
            cache.class_orbit_sizes.end(),
            distribution_orbit_sizes[index].begin(),
            distribution_orbit_sizes[index].begin() + count);
    }
    cache.offsets.resize(cache.keys.size());
    uint64_t total_entries = 0;
    for (size_t index = 0; index < cache.keys.size(); index++) {
        cache.offsets[index] = total_entries;
        total_entries += cache.counts[index];
    }
    if (total_entries != PACKED_CACHE_ENTRIES) {
        throw std::runtime_error("universal packed entry census changed");
    }
    cache.entry_count = size_t(total_entries);
    cache.bytes = cache.entry_count * sizeof(uint64_t);

    double allocation_start = seconds_now();
    std::unique_ptr<uint64_t[]> packed(new uint64_t[cache.entry_count]);
    cache.allocation_seconds = seconds_now() - allocation_start;
    std::atomic<unsigned> packing_failure{0};
    double pack_start = seconds_now();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long index = 0; index < (long long)cache.keys.size(); index++) {
        Distribution distribution =
            build_distribution(cache.keys[size_t(index)], 5, false);
        distribution = quotient_token_planes(std::move(distribution));
        if (distribution.entries.size() != cache.counts[size_t(index)]) {
            packing_failure.fetch_or(1U, std::memory_order_relaxed);
            continue;
        }
        size_t destination = size_t(cache.offsets[size_t(index)]);
        for (const Entry& entry : distribution.entries) {
            if (entry.mask & ~PACKED_CANONICAL_MASK) {
                packing_failure.fetch_or(2U, std::memory_order_relaxed);
            }
            if (entry.weight > 255) {
                packing_failure.fetch_or(4U, std::memory_order_relaxed);
            }
            const PackedWeightSpan span = cache.weight_spans[size_t(index)];
            uint32_t ordinal = 0;
            while (ordinal < span.count &&
                   (cache.class_weights[span.offset + ordinal] != entry.weight
                    || cache.class_orbit_sizes[span.offset + ordinal] !=
                           token_plane_orbit_size(entry.mask)
                   )) {
                ordinal++;
            }
            if (ordinal == span.count) {
                packing_failure.fetch_or(8U, std::memory_order_relaxed);
            }
            packed[destination++] =
                (entry.mask & PACKED_CANONICAL_MASK) |
                (uint64_t(ordinal) << PACKED_CANONICAL_WEIGHT_SHIFT);
        }
    }
    cache.pack_seconds = seconds_now() - pack_start;
    if (packing_failure.load(std::memory_order_relaxed)) {
        throw std::runtime_error("universal canonical packing bound failed");
    }

    cache.owned_entries = std::move(packed);
    cache.host_entries = cache.owned_entries.get();
    if (initialise_device) initialise_packed_cache_device(cache);
    cache.total_seconds = seconds_now() - total_start;
    std::printf(
        "PACKED_CACHE distributions=%zu entries=%zu bytes=%zu "
        "free_before=%zu free_after=%zu count_seconds=%.6f "
        "allocation_seconds=%.6f pack_seconds=%.6f upload_seconds=%.6f "
        "total_seconds=%.6f mode=%s max_distribution_weights=%zu "
        "token_plane_quotient=%d exact=OK\n",
        cache.keys.size(), cache.entry_count, cache.bytes,
        cache.free_bytes_before, cache.free_bytes_after, cache.count_seconds,
        cache.allocation_seconds, cache.pack_seconds, cache.upload_seconds,
        cache.total_seconds,
        initialise_device ? "rebuilt-host-stream" : "rebuilt-host-only"
        , cache.maximum_distribution_weights
        , 1
    );
    return cache;
}

static std::string sha256_memory(const void* data, size_t bytes) {
    Sha256 hash;
    hash.update(data, bytes);
    return hash.finish_hex();
}

static uint64_t packed_cache_catalog_bytes(uint64_t distributions,
                                           uint64_t classes) {
    return distributions *
               (sizeof(PrefixKey) + sizeof(uint64_t) + sizeof(uint32_t) +
                sizeof(PackedWeightSpan)) +
           classes * (sizeof(uint32_t) + sizeof(uint8_t));
}

static uint64_t packed_cache_checksum_count(uint64_t entry_count) {
    uint64_t bytes = entry_count * sizeof(uint64_t);
    return (bytes + PACKED_CACHE_CHECKSUM_BLOCK_BYTES - 1) /
           PACKED_CACHE_CHECKSUM_BLOCK_BYTES;
}

static uint64_t packed_cache_entry_offset(uint64_t catalog_bytes,
                                          uint64_t checksum_count) {
    uint64_t unaligned = PACKED_CACHE_HEADER_BYTES + catalog_bytes +
                         checksum_count * sizeof(PackedCacheChecksum);
    return (unaligned + 4095) & ~UINT64_C(4095);
}

static std::vector<PackedCacheChecksum> packed_cache_entry_checksums(
    const uint64_t* entries, size_t entry_count) {
    size_t bytes = entry_count * sizeof(uint64_t);
    size_t count = size_t(packed_cache_checksum_count(entry_count));
    std::vector<PackedCacheChecksum> result(count);
#pragma omp parallel for schedule(static)
    for (long long block = 0; block < (long long)count; block++) {
        size_t offset = size_t(block) * PACKED_CACHE_CHECKSUM_BLOCK_BYTES;
        size_t length = std::min<size_t>(
            PACKED_CACHE_CHECKSUM_BLOCK_BYTES, bytes - offset);
        std::string digest =
            sha256_memory(reinterpret_cast<const uint8_t*>(entries) + offset,
                          length);
        std::memcpy(result[size_t(block)].data(), digest.data(),
                    PACKED_CACHE_SHA256_HEX_BYTES);
    }
    return result;
}

static void packed_cache_hash_catalog(
    Sha256& hash, const PackedUniversalCache& cache,
    const std::vector<PackedCacheChecksum>& checksums) {
    static constexpr char domain[] = "rectangle-free-packed-7x5-v1";
    hash.update(domain, sizeof(domain) - 1);
    hash.update(cache.source_sha256);
    const uint64_t distributions = cache.keys.size();
    const uint64_t entries = cache.entry_count;
    const uint64_t classes = cache.class_weights.size();
    hash.update(&distributions, sizeof(distributions));
    hash.update(&entries, sizeof(entries));
    hash.update(&classes, sizeof(classes));
    hash.update(cache.keys.data(), cache.keys.size() * sizeof(PrefixKey));
    hash.update(cache.offsets.data(), cache.offsets.size() * sizeof(uint64_t));
    hash.update(cache.counts.data(), cache.counts.size() * sizeof(uint32_t));
    hash.update(cache.weight_spans.data(),
                cache.weight_spans.size() * sizeof(PackedWeightSpan));
    hash.update(cache.class_weights.data(),
                cache.class_weights.size() * sizeof(uint32_t));
    hash.update(cache.class_orbit_sizes.data(),
                cache.class_orbit_sizes.size() * sizeof(uint8_t));
    hash.update(checksums.data(), checksums.size() * sizeof(PackedCacheChecksum));
}

static std::string packed_cache_artifact_digest(
    const PackedUniversalCache& cache,
    const std::vector<PackedCacheChecksum>& checksums) {
    Sha256 hash;
    packed_cache_hash_catalog(hash, cache, checksums);
    return hash.finish_hex();
}

static void packed_cache_copy_catalog(uint8_t*& destination,
                                      const PackedUniversalCache& cache) {
    auto copy = [&](const void* source, size_t bytes) {
        std::memcpy(destination, source, bytes);
        destination += bytes;
    };
    copy(cache.keys.data(), cache.keys.size() * sizeof(PrefixKey));
    copy(cache.offsets.data(), cache.offsets.size() * sizeof(uint64_t));
    copy(cache.counts.data(), cache.counts.size() * sizeof(uint32_t));
    copy(cache.weight_spans.data(),
         cache.weight_spans.size() * sizeof(PackedWeightSpan));
    copy(cache.class_weights.data(),
         cache.class_weights.size() * sizeof(uint32_t));
    copy(cache.class_orbit_sizes.data(),
         cache.class_orbit_sizes.size() * sizeof(uint8_t));
}

static void write_packed_universal_cache(
    PackedUniversalCache& cache, const char* output_path) {
    if (!cache.host_entries || cache.keys.size() != PACKED_CACHE_DISTRIBUTIONS ||
        cache.entry_count != PACKED_CACHE_ENTRIES ||
        cache.source_sha256.size() != PACKED_CACHE_SHA256_HEX_BYTES) {
        throw std::runtime_error("invalid packed cache build for publication");
    }
    double checksum_start = seconds_now();
    std::vector<PackedCacheChecksum> checksums =
        packed_cache_entry_checksums(cache.host_entries, cache.entry_count);
    cache.artifact_sha256 = packed_cache_artifact_digest(cache, checksums);
    double checksum_seconds = seconds_now() - checksum_start;

    PackedCacheArtifactHeader header{};
    std::memcpy(header.magic, "R7PCK01", 8);
    header.header_bytes = PACKED_CACHE_HEADER_BYTES;
    header.endian = PACKED_CACHE_ENDIAN;
    header.rows = GRID_ROWS;
    header.columns = RIGHT_COLUMNS;
    header.token_plane_quotient = PACKED_CACHE_TOKEN_PLANE_QUOTIENT;
    header.checksum_block_bytes = PACKED_CACHE_CHECKSUM_BLOCK_BYTES;
    header.distribution_count = cache.keys.size();
    header.entry_count = cache.entry_count;
    header.class_count = cache.class_weights.size();
    header.catalog_bytes = packed_cache_catalog_bytes(
        header.distribution_count, header.class_count);
    header.checksum_count = checksums.size();
    header.entry_offset = packed_cache_entry_offset(
        header.catalog_bytes, header.checksum_count);
    header.file_bytes = header.entry_offset + cache.bytes;
    std::memcpy(header.source_sha256, cache.source_sha256.data(),
                PACKED_CACHE_SHA256_HEX_BYTES);
    std::memcpy(header.artifact_sha256, cache.artifact_sha256.data(),
                PACKED_CACHE_SHA256_HEX_BYTES);

    std::string temporary = std::string(output_path) + ".tmp." +
                            std::to_string(getpid());
    int descriptor = ::open(temporary.c_str(),
                            O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0666);
    if (descriptor < 0) {
        throw std::runtime_error(std::string("cannot create packed cache: ") +
                                 std::strerror(errno));
    }
    void* mapped = nullptr;
    auto fail = [&](const std::string& message) {
        if (mapped) munmap(mapped, size_t(header.file_bytes));
        ::close(descriptor);
        ::unlink(temporary.c_str());
        throw std::runtime_error(message);
    };
    if (ftruncate(descriptor, off_t(header.file_bytes))) {
        fail(std::string("cannot size packed cache: ") + std::strerror(errno));
    }
    mapped = mmap(nullptr, size_t(header.file_bytes), PROT_READ | PROT_WRITE,
                  MAP_SHARED, descriptor, 0);
    if (mapped == MAP_FAILED) {
        mapped = nullptr;
        fail(std::string("cannot map packed cache output: ") +
             std::strerror(errno));
    }
    double write_start = seconds_now();
    uint8_t* bytes = static_cast<uint8_t*>(mapped);
    std::memcpy(bytes, &header, sizeof(header));
    uint8_t* catalog = bytes + sizeof(header);
    packed_cache_copy_catalog(catalog, cache);
    std::memcpy(catalog, checksums.data(),
                checksums.size() * sizeof(PackedCacheChecksum));
    std::memcpy(bytes + header.entry_offset, cache.host_entries, cache.bytes);
    if (msync(mapped, size_t(header.file_bytes), MS_SYNC)) {
        fail(std::string("cannot flush packed cache: ") + std::strerror(errno));
    }
    if (munmap(mapped, size_t(header.file_bytes))) {
        mapped = nullptr;
        fail(std::string("cannot unmap packed cache: ") + std::strerror(errno));
    }
    mapped = nullptr;
    if (fsync(descriptor)) {
        fail(std::string("cannot sync packed cache: ") + std::strerror(errno));
    }
    if (::close(descriptor)) {
        descriptor = -1;
        ::unlink(temporary.c_str());
        throw std::runtime_error(std::string("cannot close packed cache: ") +
                                 std::strerror(errno));
    }
    descriptor = -1;
    if (::link(temporary.c_str(), output_path)) {
        ::unlink(temporary.c_str());
        throw std::runtime_error(std::string("cannot publish packed cache: ") +
                                 std::strerror(errno));
    }
    if (::unlink(temporary.c_str())) {
        throw std::runtime_error(std::string("cannot retire cache temporary: ") +
                                 std::strerror(errno));
    }
    double write_seconds = seconds_now() - write_start;
    std::printf(
        "PACKED_CACHE_WRITE path=%s bytes=%llu checksums=%zu "
        "checksum_seconds=%.6f write_seconds=%.6f source_sha256=%s "
        "artifact_sha256=%s exact=OK\n",
        output_path, (unsigned long long)header.file_bytes, checksums.size(),
        checksum_seconds, write_seconds, cache.source_sha256.c_str(),
        cache.artifact_sha256.c_str());
}

static PackedCacheArtifactHeader packed_cache_read_header(const char* path) {
    int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::runtime_error(std::string("cannot open packed cache: ") +
                                 std::strerror(errno));
    }
    PackedCacheArtifactHeader header{};
    ssize_t received = pread(descriptor, &header, sizeof(header), 0);
    struct stat status {};
    int stat_result = fstat(descriptor, &status);
    int close_result = ::close(descriptor);
    if (received != ssize_t(sizeof(header)) || stat_result || close_result) {
        throw std::runtime_error("cannot read packed cache header");
    }
    uint64_t expected_catalog = packed_cache_catalog_bytes(
        header.distribution_count, header.class_count);
    uint64_t expected_checksums =
        packed_cache_checksum_count(header.entry_count);
    uint64_t expected_entry_offset = packed_cache_entry_offset(
        expected_catalog, expected_checksums);
    uint64_t expected_file_bytes = expected_entry_offset +
                                   header.entry_count * sizeof(uint64_t);
    auto valid_digest = [](const char* digest) {
        for (size_t index = 0; index < PACKED_CACHE_SHA256_HEX_BYTES; index++) {
            char value = digest[index];
            if (!((value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f'))) return false;
        }
        return true;
    };
    bool reserved_zero = std::all_of(
        std::begin(header.reserved), std::end(header.reserved),
        [](uint8_t value) { return value == 0; });
    if (std::memcmp(header.magic, "R7PCK01", 8) ||
        header.header_bytes != PACKED_CACHE_HEADER_BYTES ||
        header.endian != PACKED_CACHE_ENDIAN || header.rows != GRID_ROWS ||
        header.columns != RIGHT_COLUMNS ||
        header.token_plane_quotient != PACKED_CACHE_TOKEN_PLANE_QUOTIENT ||
        header.checksum_block_bytes != PACKED_CACHE_CHECKSUM_BLOCK_BYTES ||
        header.distribution_count != PACKED_CACHE_DISTRIBUTIONS ||
        header.entry_count != PACKED_CACHE_ENTRIES ||
        header.class_count > PACKED_CACHE_DISTRIBUTIONS * 32ULL ||
        header.catalog_bytes != expected_catalog ||
        header.checksum_count != expected_checksums ||
        header.entry_offset != expected_entry_offset ||
        header.file_bytes != expected_file_bytes ||
        uint64_t(status.st_size) != header.file_bytes || !reserved_zero ||
        !valid_digest(header.source_sha256) ||
        !valid_digest(header.artifact_sha256)) {
        throw std::runtime_error("invalid packed cache artifact header");
    }
    return header;
}

static bool is_packed_cache_artifact(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open packed cache input");
    char magic[8]{};
    input.read(magic, sizeof(magic));
    return input.gcount() == sizeof(magic) &&
           !std::memcmp(magic, "R7PCK01", sizeof(magic));
}

// Checkpoint identity is the canonical orbit source, not its physical cache
// representation.  A result therefore remains reusable when a campaign moves
// between a rebuilt cache and the equivalent mapped artifact.
static std::string packed_cache_identity_sha256(const char* path) {
    if (!is_packed_cache_artifact(path)) return sha256_file(path);
    PackedCacheArtifactHeader header = packed_cache_read_header(path);
    return std::string(header.source_sha256,
                       PACKED_CACHE_SHA256_HEX_BYTES);
}

static PackedUniversalCache load_packed_universal_cache(const char* path) {
    double total_start = seconds_now();
    PackedCacheArtifactHeader header = packed_cache_read_header(path);
    PackedUniversalCache cache;
    cache.mapping.open_file(path);
    cache.entry_count = size_t(header.entry_count);
    cache.bytes = cache.entry_count * sizeof(uint64_t);
    cache.source_sha256.assign(header.source_sha256,
                               PACKED_CACHE_SHA256_HEX_BYTES);
    cache.artifact_sha256.assign(header.artifact_sha256,
                                 PACKED_CACHE_SHA256_HEX_BYTES);
    cache.keys.resize(size_t(header.distribution_count));
    cache.offsets.resize(size_t(header.distribution_count));
    cache.counts.resize(size_t(header.distribution_count));
    cache.weight_spans.resize(size_t(header.distribution_count));
    cache.class_weights.resize(size_t(header.class_count));
    cache.class_orbit_sizes.resize(size_t(header.class_count));

    const uint8_t* cursor = cache.mapping.bytes() + sizeof(header);
    auto copy = [&](void* destination, size_t bytes) {
        std::memcpy(destination, cursor, bytes);
        cursor += bytes;
    };
    copy(cache.keys.data(), cache.keys.size() * sizeof(PrefixKey));
    copy(cache.offsets.data(), cache.offsets.size() * sizeof(uint64_t));
    copy(cache.counts.data(), cache.counts.size() * sizeof(uint32_t));
    copy(cache.weight_spans.data(),
         cache.weight_spans.size() * sizeof(PackedWeightSpan));
    copy(cache.class_weights.data(),
         cache.class_weights.size() * sizeof(uint32_t));
    copy(cache.class_orbit_sizes.data(),
         cache.class_orbit_sizes.size() * sizeof(uint8_t));
    std::vector<PackedCacheChecksum> checksums(size_t(header.checksum_count));
    copy(checksums.data(), checksums.size() * sizeof(PackedCacheChecksum));
    if (cursor != cache.mapping.bytes() + sizeof(header) +
                      header.catalog_bytes +
                      header.checksum_count * sizeof(PackedCacheChecksum)) {
        throw std::runtime_error("packed cache catalog size mismatch");
    }
    cache.host_entries = reinterpret_cast<const uint64_t*>(
        cache.mapping.bytes() + header.entry_offset);

    uint64_t running_entries = 0;
    uint64_t running_classes = 0;
    bool metadata_valid =
        std::is_sorted(cache.keys.begin(), cache.keys.end()) &&
        std::adjacent_find(cache.keys.begin(), cache.keys.end()) ==
            cache.keys.end() &&
        cache.class_weights.size() == cache.class_orbit_sizes.size();
    for (size_t index = 0; metadata_valid && index < cache.keys.size(); index++) {
        PackedWeightSpan span = cache.weight_spans[index];
        metadata_valid = cache.offsets[index] == running_entries &&
                         span.offset == running_classes && span.count <= 32 &&
                         running_entries + cache.counts[index] <=
                             cache.entry_count &&
                         running_classes + span.count <=
                             cache.class_weights.size();
        running_entries += cache.counts[index];
        running_classes += span.count;
        cache.maximum_distribution_weights = std::max<size_t>(
            cache.maximum_distribution_weights, span.count);
    }
    for (size_t index = 0; metadata_valid &&
                           index < cache.class_weights.size(); index++) {
        metadata_valid = cache.class_weights[index] != 0 &&
                         (cache.class_orbit_sizes[index] == 1 ||
                          cache.class_orbit_sizes[index] == 2);
    }
    metadata_valid = metadata_valid && running_entries == cache.entry_count &&
                     running_classes == cache.class_weights.size();
    std::string computed_artifact =
        packed_cache_artifact_digest(cache, checksums);
    if (!metadata_valid || computed_artifact != cache.artifact_sha256) {
        throw std::runtime_error("packed cache catalog validation failed");
    }

    cache.entry_checksums = std::move(checksums);
    cache.checksum_count = cache.entry_checksums.size();
    cache.checksum_states =
        std::make_unique<std::atomic<uint8_t>[]>(cache.checksum_count);
    for (size_t index = 0; index < cache.checksum_count; index++) {
        cache.checksum_states[index].store(0, std::memory_order_relaxed);
    }
    cache.validation_seconds = seconds_now() - total_start;
    initialise_packed_cache_device(cache);
    cache.total_seconds = seconds_now() - total_start;
    std::printf(
        "PACKED_CACHE distributions=%zu entries=%zu bytes=%zu "
        "free_before=%zu free_after=%zu validation_seconds=%.6f "
        "upload_seconds=%.6f total_seconds=%.6f mode=%s prefetch_bytes=%zu "
        "memory_reserve_bytes=%zu "
        "max_distribution_weights=%zu artifact_sha256=%s "
        "token_plane_quotient=1 exact=OK\n",
        cache.keys.size(), cache.entry_count, cache.bytes,
        cache.free_bytes_before, cache.free_bytes_after,
        cache.validation_seconds, cache.upload_seconds, cache.total_seconds,
        cache.device_resident ? "device-resident" : "mapped-host-stream",
        cache.prefetch_bytes, cache.memory_reserve_bytes,
        cache.maximum_distribution_weights, cache.artifact_sha256.c_str());
    return cache;
}

static PackedUniversalCache load_or_build_packed_universal_cache(
    const char* path) {
    return is_packed_cache_artifact(path)
        ? load_packed_universal_cache(path)
        : build_packed_universal_cache_from_orbits(path, true);
}

static bool validate_packed_cache_range(const PackedUniversalCache& cache,
                                        uint64_t entry_offset,
                                        uint64_t entry_count) {
    if (!entry_count) return true;
    if (entry_offset > cache.entry_count ||
        entry_count > cache.entry_count - entry_offset) {
        return false;
    }
    // A freshly reconstructed cache is already held in owned memory and has
    // no serialized payload to authenticate.  Checksums apply only to mapped
    // artifacts.
    if (!cache.checksum_count && cache.owned_entries) return true;
    if (!cache.checksum_states) return false;
    uint64_t first_byte = entry_offset * sizeof(uint64_t);
    uint64_t end_byte = (entry_offset + entry_count) * sizeof(uint64_t);
    size_t first_block = size_t(first_byte /
                                PACKED_CACHE_CHECKSUM_BLOCK_BYTES);
    size_t last_block = size_t((end_byte - 1) /
                               PACKED_CACHE_CHECKSUM_BLOCK_BYTES);
    for (size_t block = first_block; block <= last_block; block++) {
        uint8_t state = cache.checksum_states[block].load(
            std::memory_order_acquire);
        if (state == 0) {
            uint8_t expected = 0;
            if (cache.checksum_states[block].compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel)) {
                size_t byte_offset =
                    block * size_t(PACKED_CACHE_CHECKSUM_BLOCK_BYTES);
                size_t length = std::min<size_t>(
                    PACKED_CACHE_CHECKSUM_BLOCK_BYTES,
                    cache.bytes - byte_offset);
                std::string digest = sha256_memory(
                    reinterpret_cast<const uint8_t*>(cache.host_entries) +
                        byte_offset,
                    length);
                bool valid = !std::memcmp(
                    digest.data(), cache.entry_checksums[block].data(),
                    PACKED_CACHE_SHA256_HEX_BYTES);
                cache.checksum_states[block].store(valid ? 2 : 3,
                                                   std::memory_order_release);
                state = valid ? 2 : 3;
            } else {
                state = expected;
            }
        }
        while (state == 1) {
            std::this_thread::yield();
            state = cache.checksum_states[block].load(
                std::memory_order_acquire);
        }
        if (state != 2) return false;
    }
    return true;
}

static bool validate_packed_cache_all(const PackedUniversalCache& cache) {
    std::atomic<bool> failure{false};
    const uint64_t entries_per_block =
        PACKED_CACHE_CHECKSUM_BLOCK_BYTES / sizeof(uint64_t);
#pragma omp parallel for schedule(static)
    for (long long block = 0;
         block < (long long)cache.checksum_count; block++) {
        uint64_t offset = uint64_t(block) * entries_per_block;
        uint64_t count = std::min<uint64_t>(entries_per_block,
                                            cache.entry_count - offset);
        if (!validate_packed_cache_range(cache, offset, count)) {
            failure.store(true, std::memory_order_relaxed);
        }
    }
    return !failure.load(std::memory_order_relaxed);
}

static size_t packed_distribution_index(const PackedUniversalCache& cache,
                                        PrefixKey canonical_key) {
    auto found = std::lower_bound(cache.keys.begin(), cache.keys.end(), canonical_key);
    if (found == cache.keys.end() || *found != canonical_key) {
        throw std::runtime_error("universal packed canonical lookup failed");
    }
    return size_t(found - cache.keys.begin());
}


struct HostStreamBuildDesc {
    size_t distribution;
    uint64_t row_map;
    uint32_t destination_offset;
    uint32_t count;
    uint32_t logical_index;
};

struct HostStreamCopy {
    uint64_t source_offset;
    uint64_t destination_offset;
    uint32_t count;
};

struct PackedLayoutSourceRef {
    size_t distribution;
    uint32_t row_map;
};

struct PrefetchedPackedChunk {
    std::vector<GpuBucketBuildDesc> descriptions;
    std::vector<uint32_t> logical_indices;
};

struct PrefetchedPackedLayout {
    size_t pair_count = 0;
    size_t entry_count = 0;
    uint64_t source_entries = 0;
    std::vector<std::array<PackedLayoutSourceRef, 2>> references;
    std::vector<PrefetchedPackedChunk> chunks;
    cudaEvent_t upload_start = nullptr;
    cudaEvent_t upload_end = nullptr;
    double plan_seconds = 0;
    double gather_seconds = 0;
};

static PrefetchedPackedLayout prefetch_host_packed_layout(
    const std::vector<std::array<PackedLayoutSourceRef, 2>>& references,
    const PackedUniversalCache& cache) {
    if (!cache.host_entries || !cache.host_prefetch ||
        !cache.device_prefetch || !cache.prefetch_stream) {
        throw std::runtime_error("packed pipeline prefetch is not initialised");
    }
    constexpr size_t description_cap = PACKED_STREAM_DESCRIPTION_CAP;
    const uint64_t chunk_source_cap =
        uint64_t(PACKED_STREAM_STAGING_BYTES) / sizeof(uint64_t);
    double plan_start = seconds_now();
    PrefetchedPackedLayout result;
    result.pair_count = references.size();
    result.references = references;
    std::vector<HostStreamBuildDesc> plan;
    plan.reserve(references.size() * 2);
    uint64_t output_entries = 0;
    for (size_t index = 0; index < references.size(); index++) {
        for (int complement = 0; complement < 2; complement++) {
            const PackedLayoutSourceRef& reference =
                references[index][complement];
            if (reference.distribution >= cache.counts.size()) {
                throw std::runtime_error(
                    "prefetched source reference is out of range");
            }
            uint32_t count = cache.counts[reference.distribution];
            if (output_entries + count > uint64_t(UINT32_MAX) + 1) {
                throw std::overflow_error(
                    "prefetched layout exceeds 32-bit offsets");
            }
            plan.push_back(HostStreamBuildDesc{
                reference.distribution, reference.row_map,
                uint32_t(output_entries), count,
                uint32_t(index * 2 + size_t(complement))});
            output_entries += count;
        }
    }
    result.entry_count = size_t(output_entries);
    std::vector<size_t> order(plan.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (plan[lhs].distribution != plan[rhs].distribution) {
            return plan[lhs].distribution < plan[rhs].distribution;
        }
        return plan[lhs].logical_index < plan[rhs].logical_index;
    });

    std::vector<HostStreamCopy> copies;
    copies.reserve(plan.size());
    size_t cursor = 0;
    while (cursor < order.size()) {
        size_t end = cursor;
        uint64_t chunk_entries = 0;
        size_t last_distribution = SIZE_MAX;
        while (end < order.size() && end - cursor < description_cap) {
            const HostStreamBuildDesc& candidate = plan[order[end]];
            uint64_t added = candidate.distribution == last_distribution
                                 ? 0
                                 : cache.counts[candidate.distribution];
            if (end != cursor && chunk_entries + added > chunk_source_cap) break;
            if (added > chunk_source_cap) {
                throw std::runtime_error(
                    "one canonical distribution exceeds prefetch chunk cap");
            }
            chunk_entries += added;
            last_distribution = candidate.distribution;
            end++;
        }
        if (end == cursor) throw std::runtime_error("empty prefetch chunk");
        uint64_t chunk_base = result.source_entries;
        PrefetchedPackedChunk chunk;
        chunk.descriptions.reserve(end - cursor);
        chunk.logical_indices.reserve(end - cursor);
        last_distribution = SIZE_MAX;
        uint64_t distribution_offset = 0;
        for (size_t position = cursor; position < end; position++) {
            const HostStreamBuildDesc& item = plan[order[position]];
            if (item.distribution != last_distribution) {
                distribution_offset = result.source_entries;
                copies.push_back(HostStreamCopy{
                    cache.offsets[item.distribution], distribution_offset,
                    cache.counts[item.distribution]});
                result.source_entries += cache.counts[item.distribution];
                last_distribution = item.distribution;
            }
            chunk.descriptions.push_back(GpuBucketBuildDesc{
                distribution_offset, item.row_map, item.destination_offset,
                item.count});
            chunk.logical_indices.push_back(item.logical_index);
        }
        if (result.source_entries - chunk_base != chunk_entries) {
            throw std::runtime_error("prefetch source planning mismatch");
        }
        result.chunks.push_back(std::move(chunk));
        cursor = end;
    }
    if (result.source_entries > cache.prefetch_bytes / sizeof(uint64_t)) {
        throw std::runtime_error("packed batch exceeds prefetch capacity");
    }
    result.plan_seconds = seconds_now() - plan_start;

    double gather_start = seconds_now();
    std::atomic<bool> checksum_failure{false};
#pragma omp parallel for schedule(static)
    for (long long copy_index = 0;
         copy_index < (long long)copies.size(); copy_index++) {
        const HostStreamCopy& copy = copies[size_t(copy_index)];
        if (!validate_packed_cache_range(cache, copy.source_offset,
                                         copy.count)) {
            checksum_failure.store(true, std::memory_order_relaxed);
        } else {
            std::memcpy(cache.host_prefetch.get() + copy.destination_offset,
                        cache.host_entries + copy.source_offset,
                        size_t(copy.count) * sizeof(uint64_t));
        }
    }
    if (checksum_failure.load(std::memory_order_relaxed)) {
        throw std::runtime_error("packed cache entry checksum mismatch");
    }
    result.gather_seconds = seconds_now() - gather_start;

    CUDA_CHECK(cudaEventCreate(&result.upload_start));
    CUDA_CHECK(cudaEventCreate(&result.upload_end));
    CUDA_CHECK(cudaEventRecord(result.upload_start,
                               cache.prefetch_stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        cache.device_prefetch.get(), cache.host_prefetch.get(),
        size_t(result.source_entries) * sizeof(uint64_t),
        cudaMemcpyHostToDevice, cache.prefetch_stream.get()));
    CUDA_CHECK(cudaEventRecord(result.upload_end,
                               cache.prefetch_stream.get()));
    return result;
}
