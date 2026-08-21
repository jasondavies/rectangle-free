#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct alignas(16) Entry {
    uint64_t mask;
    uint64_t weight;
};

struct Join {
    uint64_t lhs_offset;
    uint64_t rhs_offset;
    uint64_t lhs_count;
    uint64_t rhs_count;
    uint64_t expected;
};

struct Dataset {
    std::vector<Entry> entries;
    std::vector<Join> joins;
    uint64_t comparisons = 0;
    uint64_t kernels = 0;
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                             \
        if (error_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(error_));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

template <int THREADS>
__global__ void disjoint_join_tiled(const Entry* __restrict__ lhs,
                                    uint64_t lhs_count,
                                    const Entry* __restrict__ rhs,
                                    uint64_t rhs_count,
                                    unsigned long long* result) {
    __shared__ Entry rhs_tile[THREADS];
    __shared__ unsigned long long partial[THREADS];
    uint64_t lhs_index = (uint64_t)blockIdx.x * THREADS + threadIdx.x;
    uint64_t lhs_mask = lhs_index < lhs_count ? lhs[lhs_index].mask : 0;
    uint64_t lhs_weight = lhs_index < lhs_count ? lhs[lhs_index].weight : 0;
    unsigned long long sum = 0;
    uint64_t tile_count = (rhs_count + THREADS - 1U) / THREADS;
    for (uint64_t tile = blockIdx.y; tile < tile_count; tile += gridDim.y) {
        uint64_t rhs_index = tile * THREADS + threadIdx.x;
        rhs_tile[threadIdx.x] = rhs_index < rhs_count ? rhs[rhs_index] : Entry{0, 0};
        __syncthreads();
        if (lhs_index < lhs_count) {
            uint64_t available = rhs_count - tile * THREADS;
            int count = available < THREADS ? (int)available : THREADS;
#pragma unroll 4
            for (int j = 0; j < count; j++) {
                Entry other = rhs_tile[j];
                if ((lhs_mask & other.mask) == 0) sum += lhs_weight * other.weight;
            }
        }
        __syncthreads();
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = THREADS / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(result, partial[0]);
}

static uint64_t read_u64(std::ifstream& input) {
    uint64_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated dataset");
    return value;
}

static Dataset read_dataset(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, "T4GPU01", 7)) {
        throw std::runtime_error("bad dataset magic");
    }
    Dataset dataset;
    dataset.kernels = read_u64(input);
    dataset.joins.reserve((size_t)dataset.kernels * 2U);
    for (uint64_t kernel = 0; kernel < dataset.kernels; kernel++) {
        uint64_t counts[4];
        for (uint64_t& count : counts) count = read_u64(input);
        uint64_t expected[2] = {read_u64(input), read_u64(input)};
        uint64_t comparisons = read_u64(input);
        dataset.comparisons += comparisons;
        uint64_t offsets[4];
        for (int distribution = 0; distribution < 4; distribution++) {
            offsets[distribution] = dataset.entries.size();
            size_t old_size = dataset.entries.size();
            dataset.entries.resize(old_size + (size_t)counts[distribution]);
            input.read(reinterpret_cast<char*>(dataset.entries.data() + old_size),
                       (std::streamsize)(counts[distribution] * sizeof(Entry)));
            if (!input) throw std::runtime_error("truncated distribution");
        }
        dataset.joins.push_back(
            Join{offsets[0], offsets[1], counts[0], counts[1], expected[0]});
        dataset.joins.push_back(
            Join{offsets[2], offsets[3], counts[2], counts[3], expected[1]});
    }
    return dataset;
}

template <int THREADS>
static void launch_join(const Entry* entries, const Join& join,
                        unsigned long long* result, cudaStream_t stream) {
    uint64_t lhs_blocks = (join.lhs_count + THREADS - 1U) / THREADS;
    uint64_t rhs_tiles = (join.rhs_count + THREADS - 1U) / THREADS;
    uint64_t stripes = (2048U + lhs_blocks - 1U) / lhs_blocks;
    stripes = std::max<uint64_t>(1, std::min<uint64_t>({stripes, rhs_tiles, 32}));
    dim3 grid((unsigned)lhs_blocks, (unsigned)stripes);
    disjoint_join_tiled<THREADS><<<grid, THREADS, 0, stream>>>(
        entries + join.lhs_offset, join.lhs_count,
        entries + join.rhs_offset, join.rhs_count, result);
}

template <int THREADS>
static double run_once(const Dataset& dataset, const Entry* device_entries,
                       unsigned long long* device_results,
                       std::vector<cudaStream_t>& streams,
                       std::vector<unsigned long long>& results) {
    CUDA_CHECK(cudaMemset(device_results, 0,
                          dataset.joins.size() * sizeof(device_results[0])));
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < dataset.joins.size(); i++) {
        launch_join<THREADS>(device_entries, dataset.joins[i], device_results + i,
                             streams[i % streams.size()]);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(results.data(), device_results,
                          results.size() * sizeof(results[0]), cudaMemcpyDeviceToHost));
    return std::chrono::duration<double>(end - start).count();
}

template <int THREADS>
static void benchmark(const Dataset& dataset, const Entry* device_entries,
                      unsigned long long* device_results, int stream_count,
                      int repeats) {
    std::vector<cudaStream_t> streams((size_t)stream_count);
    for (cudaStream_t& stream : streams) CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<unsigned long long> results(dataset.joins.size());
    run_once<THREADS>(dataset, device_entries, device_results, streams, results);
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i] != dataset.joins[i].expected) {
            std::fprintf(stderr,
                         "validation failed join=%zu got=%llu expected=%llu\n", i,
                         results[i],
                         (unsigned long long)dataset.joins[i].expected);
            std::exit(1);
        }
    }
    std::vector<double> times;
    for (int repeat = 0; repeat < repeats; repeat++) {
        double seconds = run_once<THREADS>(dataset, device_entries, device_results,
                                           streams, results);
        times.push_back(seconds);
    }
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i] != dataset.joins[i].expected) {
            std::fprintf(stderr, "repeat validation failed join=%zu\n", i);
            std::exit(1);
        }
    }
    double best = *std::min_element(times.begin(), times.end());
    double mean = 0;
    for (double seconds : times) mean += seconds;
    mean /= times.size();
    double comparisons_per_second = (double)dataset.comparisons / best;
    double seconds_per_kernel = best / dataset.kernels;
    double projected_gpu_hours = seconds_per_kernel * 7343033248.0 / 3600.0;
    std::printf(
        "RESULT threads=%d streams=%d repeats=%d best_seconds=%.9f mean_seconds=%.9f "
        "comparisons_per_second=%.3f billion_comparisons_per_second=%.3f "
        "seconds_per_kernel=%.9f projected_gpu_hours=%.1f\n",
        THREADS, stream_count, repeats, best, mean, comparisons_per_second,
        comparisons_per_second / 1e9, seconds_per_kernel, projected_gpu_hours);
    for (cudaStream_t stream : streams) CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "Usage: %s DATASET [THREADS=256] [STREAMS=1] [REPEATS=3]\n",
                     argv[0]);
        return 2;
    }
    int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    int streams = argc > 3 ? std::atoi(argv[3]) : 1;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 3;
    if ((threads != 128 && threads != 256 && threads != 512) ||
        streams < 1 || streams > 32 || repeats < 1) return 2;
    Dataset dataset = read_dataset(argv[1]);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf(
        "DEVICE name=%s cc=%d.%d sm=%d memory_gib=%.2f clock_mhz=%.0f\n",
        properties.name, properties.major, properties.minor,
        properties.multiProcessorCount,
        properties.totalGlobalMem / 1073741824.0,
        properties.clockRate / 1000.0);
    std::printf(
        "DATASET kernels=%llu joins=%zu entries=%zu bytes=%zu comparisons=%llu\n",
        (unsigned long long)dataset.kernels, dataset.joins.size(),
        dataset.entries.size(), dataset.entries.size() * sizeof(Entry),
        (unsigned long long)dataset.comparisons);
    Entry* device_entries = nullptr;
    unsigned long long* device_results = nullptr;
    auto upload_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&device_entries, dataset.entries.size() * sizeof(Entry)));
    CUDA_CHECK(cudaMalloc(&device_results,
                          dataset.joins.size() * sizeof(device_results[0])));
    CUDA_CHECK(cudaMemcpy(device_entries, dataset.entries.data(),
                          dataset.entries.size() * sizeof(Entry), cudaMemcpyHostToDevice));
    auto upload_end = std::chrono::steady_clock::now();
    std::printf("UPLOAD seconds=%.9f\n",
                std::chrono::duration<double>(upload_end - upload_start).count());
    auto repeated_upload_start = std::chrono::steady_clock::now();
    constexpr int upload_repeats = 5;
    for (int repeat = 0; repeat < upload_repeats; repeat++) {
        CUDA_CHECK(cudaMemcpy(device_entries, dataset.entries.data(),
                              dataset.entries.size() * sizeof(Entry),
                              cudaMemcpyHostToDevice));
    }
    auto repeated_upload_end = std::chrono::steady_clock::now();
    double repeated_upload_seconds =
        std::chrono::duration<double>(repeated_upload_end - repeated_upload_start).count() /
        upload_repeats;
    std::printf("H2D bytes=%zu mean_seconds=%.9f gib_per_second=%.3f seconds_per_kernel=%.9f\n",
                dataset.entries.size() * sizeof(Entry), repeated_upload_seconds,
                dataset.entries.size() * sizeof(Entry) / repeated_upload_seconds /
                    1073741824.0,
                repeated_upload_seconds / dataset.kernels);
    if (threads == 128) benchmark<128>(dataset, device_entries, device_results, streams, repeats);
    if (threads == 256) benchmark<256>(dataset, device_entries, device_results, streams, repeats);
    if (threads == 512) benchmark<512>(dataset, device_entries, device_results, streams, repeats);
    CUDA_CHECK(cudaFree(device_results));
    CUDA_CHECK(cudaFree(device_entries));
    return 0;
}
