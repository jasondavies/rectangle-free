#define PREFIX_PRODUCTION_NO_MAIN
#include "../../twocolour_8x8_prefix_solve.cu"

#include <mma.h>

using BmmaMask = uint4;

struct DeviceBmmaEntries {
    BmmaMask* masks = nullptr;
    uint64_t* suffixes = nullptr;
    uint32_t* weights = nullptr;
    size_t count = 0;
};

__global__ void pack_bmma_entries(
    const PrefixEntry* __restrict__ source,
    BmmaMask* __restrict__ masks,
    uint64_t* __restrict__ suffixes,
    uint32_t* __restrict__ weights,
    size_t count) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        uint64_t suffix = source[index].suffix;
        masks[index] = make_uint4(uint32_t(suffix), uint32_t(suffix >> 32),
                                  0, 0);
        suffixes[index] = suffix;
        weights[index] = source[index].weight;
    }
}

static DeviceBmmaEntries build_bmma_entries(const DevicePrefixLayout& source,
                                             double& pack_seconds) {
    DeviceBmmaEntries result;
    result.count = source.entry_count;
    CUDA_CHECK(cudaMalloc(&result.masks, result.count * sizeof(BmmaMask)));
    CUDA_CHECK(cudaMalloc(&result.suffixes, result.count * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&result.weights, result.count * sizeof(uint32_t)));
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    unsigned blocks = unsigned(std::min<size_t>(65535, (result.count + 255) / 256));
    pack_bmma_entries<<<blocks, 256>>>(source.entries, result.masks,
                                      result.suffixes, result.weights,
                                      result.count);
    CUDA_CHECK(cudaGetLastError());
    pack_seconds += elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return result;
}

static void free_bmma_entries(DeviceBmmaEntries& entries) {
    CUDA_CHECK(cudaFree(entries.weights));
    CUDA_CHECK(cudaFree(entries.suffixes));
    CUDA_CHECK(cudaFree(entries.masks));
    entries = DeviceBmmaEntries{};
}

__global__ void soa_prefix_disjoint_joins(
    const uint64_t* __restrict__ left_suffixes,
    const uint32_t* __restrict__ left_weights,
    const uint64_t* __restrict__ right_suffixes,
    const uint32_t* __restrict__ right_weights,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    __shared__ unsigned long long warp_partial[8];
    __shared__ uint64_t warp_suffixes[THREADS];
    __shared__ uint32_t warp_weights[THREADS];
    __shared__ uint32_t next_task;
    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();
    uint32_t task_base = 0;
    if (!lane) task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end =
            min(task_count, task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t left_bucket_index = task_base / join.right_bucket_count;
        uint32_t right_bucket_index =
            task_base - left_bucket_index * join.right_bucket_count;
        uint32_t cached_left_bucket_index = UINT32_MAX;
        PrefixBucket left_bucket{};
        for (uint32_t task = task_base; task < task_end; task++) {
            if (left_bucket_index != cached_left_bucket_index) {
                left_bucket =
                    left_buckets[join.left_bucket_offset + left_bucket_index];
                cached_left_bucket_index = left_bucket_index;
            }
            PrefixBucket right_bucket =
                right_buckets[join.right_bucket_offset + right_bucket_index];
            if (!(left_bucket.prefix & right_bucket.prefix)) {
                const uint64_t* lhs_suffixes = left_suffixes;
                const uint32_t* lhs_weights = left_weights;
                const uint64_t* rhs_suffixes = right_suffixes;
                const uint32_t* rhs_weights = right_weights;
                PrefixBucket lhs = left_bucket;
                PrefixBucket rhs = right_bucket;
                if (rhs.count < lhs.count) {
                    lhs_suffixes = right_suffixes;
                    lhs_weights = right_weights;
                    rhs_suffixes = left_suffixes;
                    rhs_weights = left_weights;
                    PrefixBucket temporary = lhs;
                    lhs = rhs;
                    rhs = temporary;
                }
                for (uint32_t left_base = 0; left_base < lhs.count;
                     left_base += 32) {
                    uint32_t left_index = left_base + lane;
                    bool left_valid = left_index < lhs.count;
                    uint64_t left_suffix = left_valid
                        ? lhs_suffixes[lhs.entry_offset + left_index]
                        : 0;
                    uint32_t left_weight = left_valid
                        ? lhs_weights[lhs.entry_offset + left_index]
                        : 0;
                    for (uint32_t right_base = 0; right_base < rhs.count;
                         right_base += 32) {
                        uint32_t right_index = right_base + lane;
                        if (right_index < rhs.count) {
                            warp_suffixes[warp * 32 + lane] =
                                rhs_suffixes[rhs.entry_offset + right_index];
                            warp_weights[warp * 32 + lane] =
                                rhs_weights[rhs.entry_offset + right_index];
                        } else {
                            warp_suffixes[warp * 32 + lane] = 0;
                            warp_weights[warp * 32 + lane] = 0;
                        }
                        __syncwarp();
                        uint32_t count =
                            min(uint32_t(32), rhs.count - right_base);
#pragma unroll
                        for (uint32_t offset = 0; offset < count; offset++) {
                            if (left_valid &&
                                !(left_suffix &
                                  warp_suffixes[warp * 32 + offset])) {
                                sum += uint64_t(left_weight) *
                                       uint64_t(warp_weights[
                                           warp * 32 + offset]);
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            right_bucket_index++;
            if (right_bucket_index == join.right_bucket_count) {
                right_bucket_index = 0;
                left_bucket_index++;
            }
        }
        if (!lane) {
            task_base = atomicAdd(&next_task, uint32_t(PREFIX_TASK_CHUNK));
        }
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!lane) results[blockIdx.x] = sum;
    }
}

__global__ void bmma_prefix_disjoint_joins(
    const BmmaMask* __restrict__ left_masks,
    const uint32_t* __restrict__ left_weights,
    const BmmaMask* __restrict__ right_masks,
    const uint32_t* __restrict__ right_weights,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results) {
    namespace wmma = nvcuda::wmma;
    namespace experimental = nvcuda::wmma::experimental;
    __shared__ unsigned long long warp_partial[8];
    __shared__ uint32_t next_task;
    __shared__ __align__(32) unsigned left_tiles[8][32];
    __shared__ __align__(32) unsigned right_tiles[8][32];
    __shared__ __align__(32) int conflict_tiles[8][64];

    const PrefixJoinDesc join = joins[blockIdx.x];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5;
    const uint32_t task_count =
        join.left_bucket_count * join.right_bucket_count;
    unsigned long long sum = 0;
    if (!threadIdx.x) next_task = 0;
    __syncthreads();

    uint32_t task = 0;
    if (!lane) task = atomicAdd(&next_task, 1U);
    task = __shfl_sync(UINT32_MAX, task, 0);
    while (task < task_count) {
        uint32_t left_bucket_index = task / join.right_bucket_count;
        uint32_t right_bucket_index =
            task - left_bucket_index * join.right_bucket_count;
        PrefixBucket left =
            left_buckets[join.left_bucket_offset + left_bucket_index];
        PrefixBucket right =
            right_buckets[join.right_bucket_offset + right_bucket_index];
        if (!(left.prefix & right.prefix)) {
            for (uint32_t left_base = 0; left_base < left.count;
                 left_base += 8) {
                uint32_t left_count = min(uint32_t(8), left.count - left_base);
                for (uint32_t right_base = 0; right_base < right.count;
                     right_base += 8) {
                    uint32_t right_count =
                        min(uint32_t(8), right.count - right_base);
                    if (lane < 8) {
                        BmmaMask left_mask = lane < left_count
                            ? left_masks[left.entry_offset + left_base + lane]
                            : make_uint4(0, 0, 0, 0);
                        BmmaMask right_mask = lane < right_count
                            ? right_masks[right.entry_offset + right_base + lane]
                            : make_uint4(0, 0, 0, 0);
                        reinterpret_cast<BmmaMask*>(left_tiles[warp])[lane] =
                            left_mask;
                        reinterpret_cast<BmmaMask*>(right_tiles[warp])[lane] =
                            right_mask;
                    }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a, 8, 8, 128,
                                   experimental::precision::b1,
                                   wmma::row_major> left_fragment;
                    wmma::fragment<wmma::matrix_b, 8, 8, 128,
                                   experimental::precision::b1,
                                   wmma::col_major> right_fragment;
                    wmma::fragment<wmma::accumulator, 8, 8, 128, int>
                        conflict_fragment;
                    wmma::fill_fragment(conflict_fragment, 0);
                    wmma::load_matrix_sync(left_fragment, left_tiles[warp], 128);
                    wmma::load_matrix_sync(right_fragment, right_tiles[warp], 128);
                    wmma::bmma_sync(
                        conflict_fragment, left_fragment, right_fragment,
                        conflict_fragment, experimental::bmmaBitOpAND,
                        experimental::bmmaAccumulateOpPOPC);
                    wmma::store_matrix_sync(conflict_tiles[warp],
                                            conflict_fragment, 8,
                                            wmma::mem_row_major);
                    __syncwarp();
                    for (uint32_t output = lane; output < 64; output += 32) {
                        uint32_t row = output >> 3;
                        uint32_t column = output & 7U;
                        if (row < left_count && column < right_count &&
                            conflict_tiles[warp][output] == 0) {
                            sum += uint64_t(left_weights[
                                       left.entry_offset + left_base + row]) *
                                   uint64_t(right_weights[
                                       right.entry_offset + right_base + column]);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        if (!lane) task = atomicAdd(&next_task, 1U);
        task = __shfl_sync(UINT32_MAX, task, 0);
    }

#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < 8 ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!lane) results[blockIdx.x] = sum;
    }
}

static double time_scalar_prefix(
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    prefix_disjoint_joins<<<unsigned(join_count), THREADS>>>(
        left.entries, right.entries, left.buckets, right.buckets, joins,
        results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static double time_bmma_prefix(
    const DeviceBmmaEntries& left_entries,
    const DeviceBmmaEntries& right_entries,
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    bmma_prefix_disjoint_joins<<<unsigned(join_count), THREADS>>>(
        left_entries.masks, left_entries.weights, right_entries.masks,
        right_entries.weights, left.buckets, right.buckets, joins, results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

static double time_soa_prefix(
    const DeviceBmmaEntries& left_entries,
    const DeviceBmmaEntries& right_entries,
    const DevicePrefixLayout& left, const DevicePrefixLayout& right,
    const PrefixJoinDesc* joins, unsigned long long* results,
    size_t join_count) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    soa_prefix_disjoint_joins<<<unsigned(join_count), THREADS>>>(
        left_entries.suffixes, left_entries.weights, right_entries.suffixes,
        right_entries.weights, left.buckets, right.buckets, joins, results);
    CUDA_CHECK(cudaGetLastError());
    double seconds = elapsed_kernel(start, end);
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    return seconds;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr,
                     "Usage: %s ORBITS [START=0] [END=8192] [REPEATS=5]\n",
                     argv[0]);
        return 2;
    }
    std::string path = argv[1];
    uint64_t start_record = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0;
    uint64_t end_record = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 8192;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
    if (end_record <= start_record || repeats < 1) return 2;

    initialise_tables();
    U128 labelled_weight = 0;
    uint64_t records = 0;
    std::vector<Edge> edges = read_edges(path, start_record, end_record, 0, 0,
                                         labelled_weight, records);
    std::vector<PrefixKey> left_keys = unique_lefts(edges);
    std::vector<PrefixKey> right_keys = unique_rights(edges);
    std::vector<PrefixKey> all_keys = left_keys;
    all_keys.insert(all_keys.end(), right_keys.begin(), right_keys.end());
    CanonicalFactory factory =
        build_canonical_factory(std::move(all_keys), LEFT_COLUMNS);
    ProductionCanonicalDevice canonical = upload_production_canonical(factory);
    DevicePrefixLayout left =
        build_sparse_device_prefix_layout(left_keys, factory, canonical);
    DevicePrefixLayout right =
        build_sparse_device_prefix_layout(right_keys, factory, canonical);

    std::unordered_map<PrefixKey, uint32_t> left_index;
    std::unordered_map<PrefixKey, uint32_t> right_index;
    left_index.reserve(left_keys.size() * 2);
    right_index.reserve(right_keys.size() * 2);
    for (size_t index = 0; index < left_keys.size(); index++) {
        left_index.emplace(left_keys[index], uint32_t(index));
    }
    for (size_t index = 0; index < right_keys.size(); index++) {
        right_index.emplace(right_keys[index], uint32_t(index));
    }
    std::vector<PrefixJoinDesc> joins;
    std::vector<uint64_t> direct_work;
    joins.reserve(edges.size() * 2);
    direct_work.reserve(edges.size() * 2);
    U128 comparisons = 0;
    for (const Edge& edge : edges) {
        const PrefixPair& left_pair = left.pairs[left_index.at(edge.left)];
        const PrefixPair& right_pair = right.pairs[right_index.at(edge.right)];
        const PrefixDistribution lhs[2] = {
            left_pair.selected, left_pair.complement};
        const PrefixDistribution rhs[2] = {
            right_pair.selected, right_pair.complement};
        for (int component = 0; component < 2; component++) {
            joins.push_back(PrefixJoinDesc{
                lhs[component].bucket_offset, rhs[component].bucket_offset,
                lhs[component].bucket_count, rhs[component].bucket_count});
            uint64_t work = uint64_t(lhs[component].entry_count) *
                            rhs[component].entry_count;
            direct_work.push_back(work);
            comparisons += work;
        }
    }
    schedule_prefix_heavy_first(joins, direct_work);
    PrefixJoinDesc* device_joins = upload_vector(joins);
    unsigned long long* scalar_results = nullptr;
    unsigned long long* bmma_results = nullptr;
    unsigned long long* soa_results = nullptr;
    CUDA_CHECK(cudaMalloc(&scalar_results,
                          joins.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&bmma_results,
                          joins.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&soa_results,
                          joins.size() * sizeof(unsigned long long)));

    double pack_seconds = 0;
    DeviceBmmaEntries left_bmma = build_bmma_entries(left, pack_seconds);
    DeviceBmmaEntries right_bmma = build_bmma_entries(right, pack_seconds);
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf(
        "BMMA_INPUT records=%llu joins=%zu left_entries=%zu "
        "right_entries=%zu comparisons=%s pack_seconds=%.6f free_bytes=%zu\n",
        (unsigned long long)records, joins.size(), left.entry_count,
        right.entry_count, u128_string(comparisons).c_str(), pack_seconds,
        free_bytes);

    // Untimed exact warm-up catches layout/API errors before measurement.
    time_scalar_prefix(left, right, device_joins, scalar_results, joins.size());
    time_bmma_prefix(left_bmma, right_bmma, left, right, device_joins,
                     bmma_results, joins.size());
    time_soa_prefix(left_bmma, right_bmma, left, right, device_joins,
                    soa_results, joins.size());
    std::vector<unsigned long long> host_scalar(joins.size());
    std::vector<unsigned long long> host_bmma(joins.size());
    std::vector<unsigned long long> host_soa(joins.size());
    CUDA_CHECK(cudaMemcpy(host_scalar.data(), scalar_results,
                          host_scalar.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_bmma.data(), bmma_results,
                          host_bmma.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_soa.data(), soa_results,
                          host_soa.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    if (host_scalar != host_bmma || host_scalar != host_soa) {
        size_t mismatch = 0;
        while (mismatch < host_scalar.size() &&
               host_scalar[mismatch] == host_bmma[mismatch] &&
               host_scalar[mismatch] == host_soa[mismatch]) mismatch++;
        std::fprintf(stderr,
                     "probe mismatch join=%zu scalar=%llu bmma=%llu soa=%llu\n",
                     mismatch, host_scalar[mismatch], host_bmma[mismatch],
                     host_soa[mismatch]);
        return 1;
    }

    std::vector<double> scalar_times;
    std::vector<double> bmma_times;
    std::vector<double> soa_times;
    for (int repeat = 0; repeat < repeats; repeat++) {
        double scalar = time_scalar_prefix(left, right, device_joins,
                                           scalar_results, joins.size());
        double bmma = time_bmma_prefix(left_bmma, right_bmma, left, right,
                                       device_joins, bmma_results, joins.size());
        double soa = time_soa_prefix(left_bmma, right_bmma, left, right,
                                     device_joins, soa_results, joins.size());
        scalar_times.push_back(scalar);
        bmma_times.push_back(bmma);
        soa_times.push_back(soa);
        std::printf(
            "BMMA_REPEAT index=%d scalar=%.6f bmma=%.6f soa=%.6f "
            "bmma_speedup=%.6f soa_speedup=%.6f\n",
            repeat, scalar, bmma, soa, scalar / bmma, scalar / soa);
    }
    std::sort(scalar_times.begin(), scalar_times.end());
    std::sort(bmma_times.begin(), bmma_times.end());
    std::sort(soa_times.begin(), soa_times.end());
    double scalar_median = scalar_times[scalar_times.size() / 2];
    double bmma_median = bmma_times[bmma_times.size() / 2];
    double soa_median = soa_times[soa_times.size() / 2];
    std::printf(
        "BMMA_RESULT exact=OK scalar_median=%.6f bmma_median=%.6f "
        "soa_median=%.6f bmma_speedup=%.6f soa_speedup=%.6f "
        "pack_seconds=%.6f\n",
        scalar_median, bmma_median, soa_median,
        scalar_median / bmma_median, scalar_median / soa_median,
        pack_seconds);

    free_bmma_entries(right_bmma);
    free_bmma_entries(left_bmma);
    CUDA_CHECK(cudaFree(soa_results));
    CUDA_CHECK(cudaFree(bmma_results));
    CUDA_CHECK(cudaFree(scalar_results));
    CUDA_CHECK(cudaFree(device_joins));
    free_device_prefix_layout(right);
    free_device_prefix_layout(left);
    free_production_canonical(canonical);
    return 0;
}
