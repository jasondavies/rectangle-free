// GPU 4+5-row meet-in-the-middle gate for the dense-first 9x9 proposal.
//
// For a fixed C4-free first colour class A, enumerate partial second colour
// classes B independently in two row halves.  A partial B is represented by
// the 36 column pairs it has used; two halves combine iff those masks are
// disjoint.  Equal (pair-mask, edge-count) states are weighted by their exact
// multiplicity.  The CUDA join returns the labelled B count by cardinality.

#include <cuda_runtime.h>

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

constexpr unsigned kRows = 9;
constexpr unsigned kColumns = 9;
constexpr unsigned kColumnPairs = 36;
constexpr uint64_t kPairMask = (uint64_t{1} << kColumnPairs) - 1;
constexpr unsigned kWarpsPerBlock = 8;

struct Choice {
    uint64_t pairs;
    uint8_t edges;
};

struct Entry {
    uint64_t pairs;
    uint32_t weight;
    uint8_t edges;
};

struct DeviceFamily {
    uint64_t* pairs = nullptr;
    uint32_t* weights = nullptr;
    std::array<uint64_t, 83> offsets{};
    uint64_t size = 0;
};

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

uint64_t parse_u64(std::string_view text, int base = 10) {
    uint64_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
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
    if (rows.size() != kRows) throw std::runtime_error("--first requires nine rows");
    for (uint16_t row : rows) {
        if (row & ~uint16_t{0x1ff}) throw std::runtime_error("first row exceeds nine bits");
    }
    return rows;
}

std::array<std::array<int, kColumns>, kColumns> pair_indices() {
    std::array<std::array<int, kColumns>, kColumns> indices{};
    int next = 0;
    for (unsigned first = 0; first < kColumns; ++first) {
        for (unsigned second = first + 1; second < kColumns; ++second) {
            indices[first][second] = next++;
        }
    }
    return indices;
}

std::vector<Choice> row_choices(
    uint16_t allowed,
    const std::array<std::array<int, kColumns>, kColumns>& indices
) {
    std::vector<Choice> output;
    uint16_t subset = allowed;
    for (;;) {
        uint64_t pairs = 0;
        for (unsigned first = 0; first < kColumns; ++first) {
            if (!(subset & (uint16_t{1} << first))) continue;
            for (unsigned second = first + 1; second < kColumns; ++second) {
                if (subset & (uint16_t{1} << second)) {
                    pairs |= uint64_t{1} << indices[first][second];
                }
            }
        }
        output.push_back({pairs, static_cast<uint8_t>(__builtin_popcount(subset))});
        if (!subset) break;
        subset = (subset - 1) & allowed;
    }
    return output;
}

uint64_t state_key(uint64_t pairs, unsigned edges) {
    return pairs | (uint64_t{edges} << kColumnPairs);
}

std::vector<Entry> build_half(
    const std::vector<uint16_t>& first_rows,
    unsigned begin,
    unsigned end,
    unsigned maximum_edges
) {
    auto indices = pair_indices();
    using Counts = std::unordered_map<uint64_t, uint32_t>;
    Counts current;
    current.emplace(0, 1);
    auto started = std::chrono::steady_clock::now();
    for (unsigned row = begin; row < end; ++row) {
        std::vector<Choice> choices = row_choices(uint16_t{0x1ff} ^ first_rows[row], indices);
        Counts following;
        following.reserve(std::max<size_t>(65536, current.size() * 8));
        for (const auto& [key, multiplicity] : current) {
            uint64_t used_pairs = key & kPairMask;
            unsigned used_edges = key >> kColumnPairs;
            for (Choice choice : choices) {
                unsigned edges = used_edges + choice.edges;
                if (edges > maximum_edges || (used_pairs & choice.pairs)) continue;
                uint64_t next = state_key(used_pairs | choice.pairs, edges);
                uint32_t& count = following[next];
                if (UINT32_MAX - count < multiplicity) {
                    throw std::runtime_error("partial multiplicity exceeds uint32_t");
                }
                count += multiplicity;
            }
        }
        current = std::move(following);
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "half=" << begin << ':' << end << " rows=" << row - begin + 1
                  << " states=" << current.size() << " seconds=" << seconds << '\n';
    }
    std::vector<Entry> output;
    output.reserve(current.size());
    for (const auto& [key, multiplicity] : current) {
        output.push_back({
            key & kPairMask,
            multiplicity,
            static_cast<uint8_t>(key >> kColumnPairs),
        });
    }
    std::sort(output.begin(), output.end(), [](const Entry& left, const Entry& right) {
        if (left.edges != right.edges) return left.edges < right.edges;
        return left.pairs < right.pairs;
    });
    return output;
}

DeviceFamily upload_family(const std::vector<Entry>& entries) {
    DeviceFamily device;
    device.size = entries.size();
    std::vector<uint64_t> pairs(entries.size());
    std::vector<uint32_t> weights(entries.size());
    unsigned previous = 0;
    for (size_t index = 0; index < entries.size(); ++index) {
        while (previous <= entries[index].edges) {
            device.offsets[previous++] = index;
        }
        pairs[index] = entries[index].pairs;
        weights[index] = entries[index].weight;
    }
    while (previous < device.offsets.size()) device.offsets[previous++] = entries.size();
    cuda_check(cudaMalloc(&device.pairs, pairs.size() * sizeof(pairs[0])), "cudaMalloc pairs");
    cuda_check(cudaMalloc(&device.weights, weights.size() * sizeof(weights[0])), "cudaMalloc weights");
    cuda_check(cudaMemcpy(device.pairs, pairs.data(), pairs.size() * sizeof(pairs[0]),
                          cudaMemcpyHostToDevice), "copy pairs");
    cuda_check(cudaMemcpy(device.weights, weights.data(), weights.size() * sizeof(weights[0]),
                          cudaMemcpyHostToDevice), "copy weights");
    return device;
}

__global__ void disjoint_join(
    const uint64_t* left_pairs,
    const uint32_t* left_weights,
    uint64_t left_count,
    const uint64_t* right_pairs,
    const uint32_t* right_weights,
    uint64_t right_count,
    unsigned long long* result
) {
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    uint64_t left_index = uint64_t{blockIdx.x} * kWarpsPerBlock + warp;
    unsigned long long sum = 0;
    if (left_index < left_count) {
        uint64_t left_mask = left_pairs[left_index];
        uint64_t left_weight = left_weights[left_index];
        for (uint64_t right = lane; right < right_count; right += 32) {
            if (!(left_mask & right_pairs[right])) {
                sum += left_weight * uint64_t{right_weights[right]};
            }
        }
    }
    for (unsigned distance = 16; distance; distance >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, distance);
    }
    __shared__ unsigned long long warp_sums[kWarpsPerBlock];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long block_sum = 0;
        for (unsigned index = 0; index < kWarpsPerBlock; ++index) {
            block_sum += warp_sums[index];
        }
        atomicAdd(result, block_sum);
    }
}

}  // namespace

int main(int argc, char** argv) try {
    std::cout.setf(std::ios::unitbuf);
    std::vector<uint16_t> first_rows;
    unsigned minimum = 18;
    unsigned maximum = 29;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) throw std::runtime_error("missing option value");
        std::string_view option = argv[index];
        std::string_view value = argv[++index];
        if (option == "--first") first_rows = parse_rows(value);
        else if (option == "--minimum") minimum = parse_u64(value);
        else if (option == "--maximum") maximum = parse_u64(value);
        else throw std::runtime_error("unknown option: " + std::string(option));
    }
    if (first_rows.size() != kRows || minimum > maximum || maximum > 81) {
        throw std::runtime_error("invalid arguments");
    }
    std::sort(first_rows.begin(), first_rows.end(), [](uint16_t left, uint16_t right) {
        return __builtin_popcount(left) > __builtin_popcount(right);
    });
    // Spread the three dense rows across the 5- and 4-row halves.  Keeping all
    // dense rows in the shorter half makes the other family needlessly large.
    std::vector<uint16_t> split_rows;
    for (unsigned index : {0U, 2U, 4U, 6U, 8U, 1U, 3U, 5U, 7U}) {
        split_rows.push_back(first_rows[index]);
    }

    std::vector<Entry> left = build_half(split_rows, 0, 5, maximum);
    std::vector<Entry> right = build_half(split_rows, 5, 9, maximum);
    if (left.size() > right.size()) std::swap(left, right);
    std::cout << "left_states=" << left.size() << " right_states=" << right.size() << '\n';
    DeviceFamily device_left = upload_family(left);
    DeviceFamily device_right = upload_family(right);
    unsigned long long* device_result = nullptr;
    cuda_check(cudaMalloc(&device_result, sizeof(*device_result)), "cudaMalloc result");

    unsigned long long grand_total = 0;
    unsigned long long comparisons = 0;
    double gpu_seconds = 0;
    for (unsigned total = minimum; total <= maximum; ++total) {
        unsigned long long total_count = 0;
        for (unsigned left_edges = 0; left_edges <= total; ++left_edges) {
            unsigned right_edges = total - left_edges;
            uint64_t left_begin = device_left.offsets[left_edges];
            uint64_t left_end = device_left.offsets[left_edges + 1];
            uint64_t right_begin = device_right.offsets[right_edges];
            uint64_t right_end = device_right.offsets[right_edges + 1];
            uint64_t left_count = left_end - left_begin;
            uint64_t right_count = right_end - right_begin;
            if (!left_count || !right_count) continue;
            comparisons += left_count * right_count;
            cuda_check(cudaMemset(device_result, 0, sizeof(*device_result)), "clear result");
            cudaEvent_t start, finish;
            cuda_check(cudaEventCreate(&start), "create start event");
            cuda_check(cudaEventCreate(&finish), "create finish event");
            cuda_check(cudaEventRecord(start), "record start");
            uint64_t blocks = (left_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
            disjoint_join<<<static_cast<unsigned>(blocks), 256>>>(
                device_left.pairs + left_begin,
                device_left.weights + left_begin,
                left_count,
                device_right.pairs + right_begin,
                device_right.weights + right_begin,
                right_count,
                device_result);
            cuda_check(cudaGetLastError(), "launch disjoint join");
            cuda_check(cudaEventRecord(finish), "record finish");
            cuda_check(cudaEventSynchronize(finish), "wait for join");
            float milliseconds = 0;
            cuda_check(cudaEventElapsedTime(&milliseconds, start, finish), "time join");
            gpu_seconds += milliseconds / 1000.0;
            unsigned long long value = 0;
            cuda_check(cudaMemcpy(&value, device_result, sizeof(value),
                                  cudaMemcpyDeviceToHost), "copy result");
            total_count += value;
            cudaEventDestroy(start);
            cudaEventDestroy(finish);
        }
        grand_total += total_count;
        std::cout << "edges=" << total << " labelled_second_classes=" << total_count << '\n';
    }
    std::cout << "total_labelled_second_classes=" << grand_total
              << " comparisons=" << comparisons
              << " gpu_seconds=" << gpu_seconds
              << " comparisons_per_second=" << comparisons / gpu_seconds << '\n';

    cudaFree(device_result);
    cudaFree(device_left.pairs);
    cudaFree(device_left.weights);
    cudaFree(device_right.pairs);
    cudaFree(device_right.weights);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
