#ifndef TWCOLOUR_WEIGHT_CLASS_JOIN_CUH
#define TWCOLOUR_WEIGHT_CLASS_JOIN_CUH

#include <thrust/device_ptr.h>
#include <thrust/scan.h>


static_assert(sizeof(PrefixSuffix) <= sizeof(uint64_t),
              "grouped suffix storage supports masks through 64 bits");

static constexpr const char* WEIGHT_CLASS_JOIN_FINGERPRINT =
    "weight_class_arch_native_dual_plane_v1";

// Exact production representation for the architecture-native join. Physical
// prefix buckets retain their order and prefix, but their spans index
// WeightClassMeta records. Each class record in turn names a contiguous run of
// equal-weight suffix entries.
struct WeightClassMeta {
    uint32_t entry_offset;
    uint32_t count;
    uint32_t weight;
    // Common support-orbit size under token-plane exchange (1 or 2).
    uint32_t orbit_size;
};
static_assert(sizeof(WeightClassMeta) == 16,
              "weight-class metadata must remain aligned");

struct DeviceWeightClassLayout {
    // Entries in a weight class all inherit their exact weight from the
    // corresponding WeightClassMeta.  Keeping a second per-entry copy wastes
    // half of the 16-byte PrefixEntry stride on 8x8, so grouped layouts retain
    // only their suffix masks.
    DeviceBuffer<PrefixSuffix> suffixes;
    DeviceBuffer<PrefixBucket> buckets;
    DeviceBuffer<WeightClassMeta> classes;
    std::vector<PrefixPair> pairs;
    size_t entry_count = 0;
    size_t bucket_count = 0;
    size_t class_count = 0;
    size_t maximum_classes = 0;
    size_t candidate_slots = 0;
    size_t fixed_candidate_slots = 0;
    double build_seconds = 0;
    double plan_seconds = 0;
    double upload_seconds = 0;
    double histogram_seconds = 0;
    double metadata_seconds = 0;
    double scatter_seconds = 0;
    double source_gather_seconds = 0;
    double source_upload_seconds = 0;
    uint64_t source_entries = 0;
    uint64_t source_chunks = 0;
    double total_seconds = 0;
};

constexpr unsigned WEIGHT_CLASS_HASH_SLOTS = 32;

static __device__ __forceinline__ unsigned weight_class_hash_slot(
    uint32_t weight) {
    return (weight * UINT32_C(2654435761)) &
           (WEIGHT_CLASS_HASH_SLOTS - 1);
}

static __device__ __forceinline__ unsigned weight_class_find_or_insert(
    uint32_t* weights, uint32_t weight) {
    unsigned slot = weight_class_hash_slot(weight);
#pragma unroll
    for (unsigned probe = 0; probe < WEIGHT_CLASS_HASH_SLOTS; probe++) {
        uint32_t previous = atomicCAS(&weights[slot], 0U, weight);
        if (!previous || previous == weight) return slot;
        slot = (slot + 1) & (WEIGHT_CLASS_HASH_SLOTS - 1);
    }
    return UINT32_MAX;
}


static void free_weight_class_layout(DeviceWeightClassLayout& layout) {
    layout.classes.reset();
    layout.buckets.reset();
    layout.suffixes.reset();
    layout = DeviceWeightClassLayout{};
}

struct PtxFragmentA {
    uint32_t bits0;
    uint32_t bits1;
    bool valid0;
    bool valid1;
};

struct PtxFragmentB {
    uint32_t bits;
    bool valid0;
    bool valid1;
};

static __device__ __forceinline__ uint32_t weight_class_suffix_word(
    uint64_t suffix, unsigned word) {
    return word == 0 ? uint32_t(suffix)
                     : word == 1 ? uint32_t(suffix >> 32) : 0;
}

static __device__ __forceinline__ void weight_class_inline_bmma_16x8(
    uint32_t a0, uint32_t a1, uint32_t b, uint32_t& d0, uint32_t& d1,
    uint32_t& d2, uint32_t& d3) {
    uint32_t zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b), "r"(zero), "r"(zero),
          "r"(zero), "r"(zero));
}

static __device__ __forceinline__ PtxFragmentA load_weight_class_ptx_a(
    const PrefixSuffix* __restrict__ suffixes, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane, bool swap_planes) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    const uint32_t row0 = group;
    const uint32_t row1 = group + 8;
    PtxFragmentA fragment{};
    fragment.valid0 = row0 < count;
    fragment.valid1 = row1 < count;
    uint64_t suffix0 = fragment.valid0
        ? suffixes[offset + base + row0] : 0;
    uint64_t suffix1 = fragment.valid1
        ? suffixes[offset + base + row1] : 0;
    if (swap_planes) {
        suffix0 = swap_suffix_token_planes(suffix0);
        suffix1 = swap_suffix_token_planes(suffix1);
    }
    fragment.bits0 = weight_class_suffix_word(suffix0, word);
    fragment.bits1 = weight_class_suffix_word(suffix1, word);
    return fragment;
}

static __device__ __forceinline__ PtxFragmentB load_weight_class_ptx_b(
    const PrefixSuffix* __restrict__ suffixes, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane, bool swap_planes) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    const uint32_t output0 = 2 * word;
    const uint32_t output1 = output0 + 1;
    PtxFragmentB fragment{};
    uint64_t suffix = group < count
        ? suffixes[offset + base + group] : 0;
    if (swap_planes) suffix = swap_suffix_token_planes(suffix);
    fragment.bits = weight_class_suffix_word(suffix, word);
    fragment.valid0 = output0 < count;
    fragment.valid1 = output1 < count;
    return fragment;
}

static __device__ __forceinline__ unsigned long long
weight_class_predicate_join(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane, bool swap_right = false) {
    bool swap_left = false;
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    if (reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count)) {
        const PrefixSuffix* temporary_suffixes = left_suffixes;
        left_suffixes = right_suffixes;
        right_suffixes = temporary_suffixes;
        PrefixBucket temporary_bucket = left;
        left = right;
        right = temporary_bucket;
        swap_left = swap_right;
        swap_right = false;
    }
    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < left.count; left_base += 16) {
        uint32_t left_count = min(uint32_t(16), left.count - left_base);
        PtxFragmentA a = load_weight_class_ptx_a(
            left_suffixes, left.entry_offset, left_count, left_base, lane,
            swap_left);
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 8) {
            uint32_t right_count = min(uint32_t(8),
                                       right.count - right_base);
            PtxFragmentB b = load_weight_class_ptx_b(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, swap_right);
            uint32_t d0, d1, d2, d3;
            weight_class_inline_bmma_16x8(
                a.bits0, a.bits1, b.bits, d0, d1, d2, d3);
            sum += a.valid0 && b.valid0 && d0 == 0;
            sum += a.valid0 && b.valid1 && d1 == 0;
            sum += a.valid1 && b.valid0 && d2 == 0;
            sum += a.valid1 && b.valid1 && d3 == 0;
        }
    }
    return sum;
}

static __device__ __forceinline__ unsigned long long
weight_class_fragment_count(PtxFragmentA a, PtxFragmentB b) {
    uint32_t d0, d1, d2, d3;
    weight_class_inline_bmma_16x8(
        a.bits0, a.bits1, b.bits, d0, d1, d2, d3);
    unsigned long long sum = 0;
    sum += a.valid0 && b.valid0 && d0 == 0;
    sum += a.valid0 && b.valid1 && d1 == 0;
    sum += a.valid1 && b.valid0 && d2 == 0;
    sum += a.valid1 && b.valid1 && d3 == 0;
    return sum;
}

// Evaluate both token-plane orientations in one traversal when both physical
// prefixes are compatible.  The lower-padding orientation is unchanged; the
// operand unaffected by token-plane exchange is loaded once per tile pair.
static __device__ __forceinline__ unsigned long long
weight_class_predicate_join_dual(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane) {
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    const bool reverse = reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count);
    unsigned long long sum = 0;
    if (!reverse) {
        for (uint32_t left_base = 0; left_base < left.count;
             left_base += 16) {
            uint32_t left_count = min(uint32_t(16), left.count - left_base);
            PtxFragmentA a = load_weight_class_ptx_a(
                left_suffixes, left.entry_offset, left_count, left_base, lane,
                false);
            for (uint32_t right_base = 0; right_base < right.count;
                 right_base += 8) {
                uint32_t right_count = min(uint32_t(8),
                                           right.count - right_base);
                PtxFragmentB b = load_weight_class_ptx_b(
                    right_suffixes, right.entry_offset, right_count,
                    right_base, lane, false);
                sum += weight_class_fragment_count(a, b);
                b = load_weight_class_ptx_b(
                    right_suffixes, right.entry_offset, right_count,
                    right_base, lane, true);
                sum += weight_class_fragment_count(a, b);
            }
        }
    } else {
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 16) {
            uint32_t right_count = min(uint32_t(16),
                                       right.count - right_base);
            PtxFragmentA a = load_weight_class_ptx_a(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, false);
            PtxFragmentA swapped_a = load_weight_class_ptx_a(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, true);
            for (uint32_t left_base = 0; left_base < left.count;
                 left_base += 8) {
                uint32_t left_count = min(uint32_t(8),
                                          left.count - left_base);
                PtxFragmentB b = load_weight_class_ptx_b(
                    left_suffixes, left.entry_offset, left_count, left_base,
                    lane, false);
                sum += weight_class_fragment_count(a, b);
                sum += weight_class_fragment_count(swapped_a, b);
            }
        }
    }
    return sum;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200

// Spread eight logical bits into eight FP4 E2M1 nibbles.  E2M1 code 0x2 is
// exactly +1.0, while zero remains 0x0.
static __device__ __forceinline__ uint32_t weight_class_fp4_byte(
    uint32_t bits) {
    bits &= 0xffU;
    bits = (bits | (bits << 12)) & 0x000f000fU;
    bits = (bits | (bits << 6)) & 0x03030303U;
    bits = (bits | (bits << 3)) & 0x11111111U;
    return bits << 1;
}

struct WeightClassFp4A {
    uint32_t bits0;
    uint32_t bits1;
    uint32_t bits2;
    uint32_t bits3;
    bool valid0;
    bool valid1;
};

struct WeightClassFp4B {
    uint32_t bits0;
    uint32_t bits1;
    bool valid0;
    bool valid1;
};

static __device__ __forceinline__ WeightClassFp4A
load_weight_class_fp4_a(
    const PrefixSuffix* __restrict__ suffixes, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane, bool swap_planes) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    const uint32_t row0 = group;
    const uint32_t row1 = group + 8;
    WeightClassFp4A fragment{};
    fragment.valid0 = row0 < count;
    fragment.valid1 = row1 < count;
    uint64_t suffix0 = fragment.valid0
        ? uint64_t(suffixes[offset + base + row0]) : 0;
    uint64_t suffix1 = fragment.valid1
        ? uint64_t(suffixes[offset + base + row1]) : 0;
    if (swap_planes) {
        suffix0 = swap_suffix_token_planes(suffix0);
        suffix1 = swap_suffix_token_planes(suffix1);
    }
    fragment.bits0 = weight_class_fp4_byte(uint32_t(suffix0 >> (8 * word)));
    fragment.bits1 = weight_class_fp4_byte(uint32_t(suffix1 >> (8 * word)));
    fragment.bits2 = weight_class_fp4_byte(
        uint32_t(suffix0 >> (8 * (word + 4))));
    fragment.bits3 = weight_class_fp4_byte(
        uint32_t(suffix1 >> (8 * (word + 4))));
    return fragment;
}

static __device__ __forceinline__ WeightClassFp4B
load_weight_class_fp4_b(
    const PrefixSuffix* __restrict__ suffixes, uint32_t offset,
    uint32_t count, uint32_t base, unsigned lane, bool swap_planes) {
    const unsigned group = lane >> 2;
    const unsigned word = lane & 3U;
    const uint32_t output0 = 2 * word;
    const uint32_t output1 = output0 + 1;
    WeightClassFp4B fragment{};
    uint64_t suffix = group < count
        ? uint64_t(suffixes[offset + base + group]) : 0;
    if (swap_planes) suffix = swap_suffix_token_planes(suffix);
    fragment.bits0 = weight_class_fp4_byte(uint32_t(suffix >> (8 * word)));
    fragment.bits1 = weight_class_fp4_byte(
        uint32_t(suffix >> (8 * (word + 4))));
    fragment.valid0 = output0 < count;
    fragment.valid1 = output1 < count;
    return fragment;
}

static __device__ __forceinline__ unsigned long long
weight_class_fp4_fragment_count(
    WeightClassFp4A a, WeightClassFp4B b, unsigned lane) {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    // NVFP4 UE4M3 encodes 1.0 as 0x38.  The 4X scale ABI consumes A
    // selectors from lanes 0/1 of each four-lane group and B from lane 0.
    const unsigned thread_in_group = lane & 3U;
    const uint32_t unit_scales = UINT32_C(0x38383838);
    const uint32_t scale_a = thread_in_group < 2 ? unit_scales : 0;
    const uint32_t scale_b = thread_in_group == 0 ? unit_scales : 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."
        "scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0,0}, {%11}, {0,0};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a.bits0), "r"(a.bits1), "r"(a.bits2), "r"(a.bits3),
          "r"(b.bits0), "r"(b.bits1), "r"(scale_a), "r"(scale_b));
    unsigned long long sum = 0;
    sum += a.valid0 && b.valid0 && d0 == 0.0f;
    sum += a.valid0 && b.valid1 && d1 == 0.0f;
    sum += a.valid1 && b.valid0 && d2 == 0.0f;
    sum += a.valid1 && b.valid1 && d3 == 0.0f;
    return sum;
}

static __device__ __forceinline__ unsigned long long
weight_class_predicate_join_fp4(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane, bool swap_right = false) {
    bool swap_left = false;
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    if (reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count)) {
        const PrefixSuffix* temporary_suffixes = left_suffixes;
        left_suffixes = right_suffixes;
        right_suffixes = temporary_suffixes;
        PrefixBucket temporary_bucket = left;
        left = right;
        right = temporary_bucket;
        swap_left = swap_right;
        swap_right = false;
    }
    unsigned long long sum = 0;
    for (uint32_t left_base = 0; left_base < left.count; left_base += 16) {
        uint32_t left_count = min(uint32_t(16), left.count - left_base);
        WeightClassFp4A a = load_weight_class_fp4_a(
            left_suffixes, left.entry_offset, left_count, left_base, lane,
            swap_left);
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 8) {
            uint32_t right_count = min(uint32_t(8),
                                       right.count - right_base);
            WeightClassFp4B b = load_weight_class_fp4_b(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, swap_right);
            sum += weight_class_fp4_fragment_count(a, b, lane);
        }
    }
    return sum;
}

static __device__ __forceinline__ unsigned long long
weight_class_predicate_join_fp4_dual(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane) {
    uint64_t forward_tiles = uint64_t((left.count + 15) / 16) *
                             ((right.count + 7) / 8);
    uint64_t reverse_tiles = uint64_t((right.count + 15) / 16) *
                             ((left.count + 7) / 8);
    const bool reverse = reverse_tiles < forward_tiles ||
        (reverse_tiles == forward_tiles && right.count < left.count);
    unsigned long long sum = 0;
    if (!reverse) {
        for (uint32_t left_base = 0; left_base < left.count;
             left_base += 16) {
            uint32_t left_count = min(uint32_t(16), left.count - left_base);
            WeightClassFp4A a = load_weight_class_fp4_a(
                left_suffixes, left.entry_offset, left_count, left_base,
                lane, false);
            for (uint32_t right_base = 0; right_base < right.count;
                 right_base += 8) {
                uint32_t right_count = min(uint32_t(8),
                                           right.count - right_base);
                WeightClassFp4B b = load_weight_class_fp4_b(
                    right_suffixes, right.entry_offset, right_count,
                    right_base, lane, false);
                sum += weight_class_fp4_fragment_count(a, b, lane);
                b = load_weight_class_fp4_b(
                    right_suffixes, right.entry_offset, right_count,
                    right_base, lane, true);
                sum += weight_class_fp4_fragment_count(a, b, lane);
            }
        }
    } else {
        for (uint32_t right_base = 0; right_base < right.count;
             right_base += 16) {
            uint32_t right_count = min(uint32_t(16),
                                       right.count - right_base);
            WeightClassFp4A a = load_weight_class_fp4_a(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, false);
            WeightClassFp4A swapped_a = load_weight_class_fp4_a(
                right_suffixes, right.entry_offset, right_count, right_base,
                lane, true);
            for (uint32_t left_base = 0; left_base < left.count;
                 left_base += 8) {
                uint32_t left_count = min(uint32_t(8),
                                          left.count - left_base);
                WeightClassFp4B b = load_weight_class_fp4_b(
                    left_suffixes, left.entry_offset, left_count, left_base,
                    lane, false);
                sum += weight_class_fp4_fragment_count(a, b, lane);
                sum += weight_class_fp4_fragment_count(swapped_a, b, lane);
            }
        }
    }
    return sum;
}

#endif

static __device__ __forceinline__ unsigned long long
weight_class_selected_join(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane, bool swap_right = false) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    return weight_class_predicate_join_fp4(
        left_suffixes, left, right_suffixes, right, lane, swap_right);
#else
    return weight_class_predicate_join(
        left_suffixes, left, right_suffixes, right, lane, swap_right);
#endif
}

static __device__ __forceinline__ unsigned long long
weight_class_selected_join_dual(
    const PrefixSuffix* __restrict__ left_suffixes, PrefixBucket left,
    const PrefixSuffix* __restrict__ right_suffixes, PrefixBucket right,
    unsigned lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    return weight_class_predicate_join_fp4_dual(
        left_suffixes, left, right_suffixes, right, lane);
#else
    return weight_class_predicate_join_dual(
        left_suffixes, left, right_suffixes, right, lane);
#endif
}

static __device__ __forceinline__ unsigned long long
weight_class_bucket_pair_join(
    const PrefixSuffix* __restrict__ left_suffixes,
    const PrefixSuffix* __restrict__ right_suffixes,
    const WeightClassMeta* __restrict__ left_classes,
    const WeightClassMeta* __restrict__ right_classes,
    PrefixBucket left_bucket, PrefixBucket right_bucket,
    bool forward_prefix, bool swapped_prefix, unsigned lane) {
    unsigned long long sum = 0;
    for (uint32_t a = 0; a < left_bucket.count; a++) {
        WeightClassMeta left_class =
            left_classes[left_bucket.entry_offset + a];
        PrefixBucket left_entries_bucket{
            left_class.entry_offset, left_class.count, 0, 0};
        for (uint32_t b = 0; b < right_bucket.count; b++) {
            WeightClassMeta right_class =
                right_classes[right_bucket.entry_offset + b];
            PrefixBucket right_entries_bucket{
                right_class.entry_offset, right_class.count, 0, 0};
            unsigned long long compatible = 0;
            if (forward_prefix && swapped_prefix &&
                right_class.orbit_size == 2) {
                compatible = weight_class_selected_join_dual(
                    left_suffixes, left_entries_bucket,
                    right_suffixes, right_entries_bucket, lane);
            } else {
                if (forward_prefix) {
                    compatible += weight_class_selected_join(
                        left_suffixes, left_entries_bucket,
                        right_suffixes, right_entries_bucket, lane);
                }
                if (right_class.orbit_size == 2 && swapped_prefix) {
                    compatible += weight_class_selected_join(
                        left_suffixes, left_entries_bucket,
                        right_suffixes, right_entries_bucket, lane, true);
                }
            }
            sum += compatible * uint64_t(left_class.orbit_size) *
                   uint64_t(left_class.weight) *
                   uint64_t(right_class.weight);
        }
    }
    return sum;
}

__global__ void weight_class_prefix_joins(
    const PrefixSuffix* __restrict__ left_suffixes,
    const PrefixSuffix* __restrict__ right_suffixes,
    const PrefixBucket* __restrict__ left_buckets,
    const PrefixBucket* __restrict__ right_buckets,
    const WeightClassMeta* __restrict__ left_classes,
    const WeightClassMeta* __restrict__ right_classes,
    const PrefixJoinDesc* __restrict__ joins,
    unsigned long long* __restrict__ results
    ) {
    constexpr unsigned warps_per_block = THREADS / 32;
    __shared__ unsigned long long warp_partial[warps_per_block];
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
    if (!lane) task_base = atomicAdd(&next_task,
                                     uint32_t(PREFIX_TASK_CHUNK));
    task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    while (task_base < task_count) {
        uint32_t task_end = min(task_count,
                                task_base + uint32_t(PREFIX_TASK_CHUNK));
        uint32_t li = task_base / join.right_bucket_count;
        uint32_t ri = task_base - li * join.right_bucket_count;
        for (uint32_t task = task_base; task < task_end; task++) {
            PrefixBucket left_bucket =
                left_buckets[join.left_bucket_offset + li];
            PrefixBucket right_bucket =
                right_buckets[join.right_bucket_offset + ri];
            const bool forward_prefix =
                !(left_bucket.prefix & right_bucket.prefix);
            const bool swapped_prefix =
                !(left_bucket.prefix &
                  swap_prefix_token_planes(right_bucket.prefix));
            if (forward_prefix || swapped_prefix)
                sum += weight_class_bucket_pair_join(
                    left_suffixes, right_suffixes, left_classes,
                    right_classes, left_bucket, right_bucket,
                    forward_prefix, swapped_prefix, lane);
            if (++ri == join.right_bucket_count) {
                ri = 0;
                li++;
            }
        }
        if (!lane) task_base = atomicAdd(&next_task,
                                         uint32_t(PREFIX_TASK_CHUNK));
        task_base = __shfl_sync(UINT32_MAX, task_base, 0);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (!lane) warp_partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = lane < warps_per_block ? warp_partial[lane] : 0;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(UINT32_MAX, sum, offset);
        }
        if (!lane) results[blockIdx.x] = sum;
    }
}

// Direct grouped-layout producer.  A row permutation changes suffix masks but
// not weights, so each labelled description inherits the exact, small weight
// alphabet of its canonical source distribution.  Histogramming by its weight
// ordinal builds the final class runs without first materialising an ordinary
// prefix layout.
struct DirectWeightBuildDesc {
    uint64_t source_offset;
    uint32_t row_map;
    uint32_t destination_offset;
    uint32_t count;
    uint32_t bucket_base;
    uint32_t bucket_count;
    uint32_t weight_offset;
    uint32_t weight_count;
    uint32_t candidate_base;
};

struct DirectBucketAux {
    uint32_t weight_offset;
    uint32_t weight_count;
    uint32_t candidate_offset;
};
static_assert(sizeof(DirectBucketAux) == 12,
              "direct bucket scratch metadata must remain compact");

struct DirectWeightClassWorkspace {
    DeviceBuffer<uint32_t> dense;
    DeviceBuffer<uint32_t> occupied_counts;
    DeviceBuffer<uint32_t> failure;
    DeviceBuffer<DirectWeightBuildDesc> descriptions;
    DeviceBuffer<DirectBucketAux> bucket_aux;
    DeviceBuffer<uint32_t> candidates;
    DeviceBuffer<uint32_t> class_offsets;
    DeviceBuffer<uint32_t> maximum;

    void reset() {
        dense.reset();
        occupied_counts.reset();
        failure.reset();
        descriptions.reset();
        bucket_aux.reset();
        candidates.reset();
        class_offsets.reset();
        maximum.reset();
    }
};

static __device__ __forceinline__ void initialise_shared_pair_permutation(
    uint32_t row_map, uint64_t* images) {
    for (unsigned source = threadIdx.x; source < 2 * PAIRS;
         source += blockDim.x) {
        unsigned pair = source % PAIRS;
        unsigned first = 0;
        while (pair >= unsigned(ROWS - first - 1)) {
            pair -= unsigned(ROWS - first - 1);
            first++;
        }
        unsigned second = first + 1 + pair;
        unsigned image_first = (row_map >> (4 * first)) & 15U;
        unsigned image_second = (row_map >> (4 * second)) & 15U;
        unsigned destination =
            (source / PAIRS) * PAIRS +
            unsigned(pair_number(int(image_first), int(image_second)));
        images[source] = UINT64_C(1) << destination;
    }
    __syncthreads();
}

static __device__ __forceinline__ uint64_t transform_pair_mask_shared(
    uint64_t mask, const uint64_t* images) {
    uint64_t transformed = 0;
    while (mask) {
        unsigned source = unsigned(__ffsll(mask) - 1);
        transformed |= images[source];
        mask &= mask - 1;
    }
    return transformed;
}

__global__ void histogram_direct_weight_prefixes(
    const CanonicalDeviceMask* __restrict__ canonical_masks,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    uint32_t* __restrict__ dense_counts) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        uint64_t mask = canonical_masks[description.source_offset + index];
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        atomicAdd(&dense_counts[dense_base + prefix], 1U);
    }
}

#if GRID_ROWS == 7
__global__ void histogram_direct_weight_prefixes_packed(
    const uint64_t* __restrict__ canonical,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    uint32_t* __restrict__ dense_counts) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        uint64_t packed = canonical[description.source_offset + index];
        uint64_t mask = packed & PACKED_CANONICAL_MASK;
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        atomicAdd(&dense_counts[dense_base + prefix], 1U);
    }
}
#endif

__global__ void count_direct_occupied_prefixes(
    const uint32_t* __restrict__ dense_counts,
    uint32_t* __restrict__ occupied_counts) {
    __shared__ uint32_t partial[THREADS];
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    uint32_t count = 0;
    for (uint32_t prefix = threadIdx.x; prefix < PREFIX_BUCKET_COUNT;
         prefix += blockDim.x) {
        count += dense_counts[dense_base + prefix] != 0;
    }
    partial[threadIdx.x] = count;
    __syncthreads();
    for (unsigned offset = THREADS / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) partial[threadIdx.x] +=
            partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (!threadIdx.x) occupied_counts[blockIdx.x] = partial[0];
}

__global__ void build_direct_prefix_metadata(
    const DirectWeightBuildDesc* __restrict__ descriptions,
    uint32_t* __restrict__ dense_bucket_map,
    PrefixBucket* __restrict__ buckets,
    DirectBucketAux* __restrict__ bucket_aux,
    uint32_t* __restrict__ failure) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    unsigned lane = threadIdx.x & 31U;
    uint32_t running = 0;
    uint32_t bucket_ordinal = 0;
    for (uint32_t base = 0; base < PREFIX_BUCKET_COUNT; base += 32) {
        uint32_t prefix = base + lane;
        uint32_t count = prefix < PREFIX_BUCKET_COUNT
            ? dense_bucket_map[dense_base + prefix] : 0;
        unsigned occupied = __ballot_sync(UINT32_MAX, count != 0);
        uint32_t inclusive = count;
#pragma unroll
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            uint32_t previous = __shfl_up_sync(UINT32_MAX, inclusive, offset);
            if (lane >= offset) inclusive += previous;
        }
        if (count) {
            unsigned lower_lanes = lane ? (UINT32_C(1) << lane) - 1U : 0U;
            uint32_t ordinal = bucket_ordinal + __popc(occupied & lower_lanes);
            uint32_t bucket_index = description.bucket_base + ordinal;
            buckets[bucket_index] = PrefixBucket{
                description.destination_offset + running + inclusive - count,
                count, uint16_t(prefix), 0};
            bucket_aux[bucket_index] = DirectBucketAux{
                description.weight_offset, description.weight_count,
                description.candidate_base +
                    ordinal * description.weight_count};
            dense_bucket_map[dense_base + prefix] = bucket_index;
        }
        running += __shfl_sync(UINT32_MAX, inclusive, 31);
        bucket_ordinal += __popc(occupied);
    }
    if (!lane && (running != description.count ||
                  bucket_ordinal != description.bucket_count)) {
        atomicExch(failure, 1U);
    }
}

__global__ void histogram_direct_weight_classes(
    const CanonicalDeviceMask* __restrict__ canonical_masks,
    const uint8_t* __restrict__ canonical_weight_ordinals,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    const uint32_t* __restrict__ dense_bucket_map,
    const DirectBucketAux* __restrict__ bucket_aux,
    uint32_t* __restrict__ candidate_counts,
    uint32_t* __restrict__ failure) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        size_t source = description.source_offset + index;
        uint64_t mask = canonical_masks[source];
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        uint32_t ordinal = canonical_weight_ordinals[source];
        if (ordinal >= description.weight_count) {
            atomicExch(failure, 1U);
            continue;
        }
        uint32_t bucket = dense_bucket_map[dense_base + prefix];
        atomicAdd(&candidate_counts[
                      bucket_aux[bucket].candidate_offset + ordinal],
                  1U);
    }
}

#if GRID_ROWS == 7
__global__ void histogram_direct_weight_classes_packed(
    const uint64_t* __restrict__ canonical,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    const uint32_t* __restrict__ dense_bucket_map,
    const DirectBucketAux* __restrict__ bucket_aux,
    uint32_t* __restrict__ candidate_counts,
    uint32_t* __restrict__ failure) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        uint64_t packed = canonical[description.source_offset + index];
        uint64_t mask = packed & PACKED_CANONICAL_MASK;
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        uint32_t ordinal = uint32_t(packed >> PACKED_CANONICAL_WEIGHT_SHIFT);
        if (ordinal >= description.weight_count) {
            atomicExch(failure, 1U);
            continue;
        }
        uint32_t bucket = dense_bucket_map[dense_base + prefix];
        atomicAdd(&candidate_counts[
                      bucket_aux[bucket].candidate_offset + ordinal],
                  1U);
    }
}
#endif

__global__ void count_direct_classes(
    const uint32_t* __restrict__ candidate_counts,
    const DirectBucketAux* __restrict__ bucket_aux, size_t bucket_count,
    uint32_t* __restrict__ class_offsets, uint32_t* __restrict__ maximum) {
    for (size_t bucket = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         bucket < bucket_count;
         bucket += size_t(blockDim.x) * gridDim.x) {
        uint32_t count = 0;
        for (uint32_t ordinal = 0;
             ordinal < bucket_aux[bucket].weight_count; ordinal++) {
            count += candidate_counts[
                bucket_aux[bucket].candidate_offset + ordinal] != 0;
        }
        class_offsets[bucket] = count;
        atomicMax(maximum, count);
    }
}

__global__ void build_direct_weight_classes(
    PrefixBucket* __restrict__ buckets,
    const DirectBucketAux* __restrict__ bucket_aux, size_t bucket_count,
    uint32_t* __restrict__ candidate_positions,
    const uint32_t* __restrict__ class_weights,
    const uint8_t* __restrict__ class_orbit_sizes,
    const uint32_t* __restrict__ class_offsets,
    WeightClassMeta* __restrict__ classes) {
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31U;
    constexpr unsigned warps_per_block = THREADS / 32;
    size_t bucket_index = size_t(blockIdx.x) * warps_per_block + warp;
    if (bucket_index >= bucket_count) return;
    PrefixBucket physical = buckets[bucket_index];
    DirectBucketAux aux = bucket_aux[bucket_index];
    size_t candidate_index = aux.candidate_offset + lane;
    uint32_t count = lane < aux.weight_count
        ? candidate_positions[candidate_index] : 0;
    unsigned occupied = __ballot_sync(UINT32_MAX, count != 0);
    uint32_t inclusive = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        uint32_t previous = __shfl_up_sync(UINT32_MAX, inclusive, offset);
        if (lane >= offset) inclusive += previous;
    }
    uint32_t local_base = inclusive - count;
    if (lane < aux.weight_count) {
        candidate_positions[candidate_index] =
            physical.entry_offset + local_base;
    }
    if (count) {
        unsigned ordinal = __popc(
            occupied & ((UINT32_C(1) << lane) - 1U));
        classes[class_offsets[bucket_index] + ordinal] = WeightClassMeta{
            physical.entry_offset + local_base, count,
            class_weights[aux.weight_offset + lane],
            class_orbit_sizes
                ? uint32_t(class_orbit_sizes[aux.weight_offset + lane]) : 0U};
    }
    if (!lane) {
        buckets[bucket_index] = PrefixBucket{
            class_offsets[bucket_index], uint32_t(__popc(occupied)),
            physical.prefix, 0};
    }
}

__global__ void scatter_direct_weight_classes(
    const CanonicalDeviceMask* __restrict__ canonical_masks,
    const uint8_t* __restrict__ canonical_weight_ordinals,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    const uint32_t* __restrict__ dense_bucket_map,
    const DirectBucketAux* __restrict__ bucket_aux,
    uint32_t* __restrict__ candidate_positions,
    PrefixSuffix* __restrict__ output, uint32_t* __restrict__ failure) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        size_t source = description.source_offset + index;
        uint64_t mask = canonical_masks[source];
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        uint32_t ordinal = canonical_weight_ordinals[source];
        if (ordinal >= description.weight_count) {
            atomicExch(failure, 1U);
            continue;
        }
        uint32_t bucket = dense_bucket_map[dense_base + prefix];
        uint32_t destination = atomicAdd(
            &candidate_positions[
                bucket_aux[bucket].candidate_offset + ordinal],
            1U);
        output[destination] = suffix;
    }
}

#if GRID_ROWS == 7
__global__ void scatter_direct_weight_classes_packed(
    const uint64_t* __restrict__ canonical,
    const DirectWeightBuildDesc* __restrict__ descriptions,
    const uint32_t* __restrict__ dense_bucket_map,
    const DirectBucketAux* __restrict__ bucket_aux,
    uint32_t* __restrict__ candidate_positions,
    PrefixSuffix* __restrict__ output, uint32_t* __restrict__ failure) {
    const DirectWeightBuildDesc description = descriptions[blockIdx.x];
    __shared__ uint64_t pair_images[2 * PAIRS];
    initialise_shared_pair_permutation(description.row_map, pair_images);
    size_t dense_base = size_t(blockIdx.x) * PREFIX_BUCKET_COUNT;
    for (uint32_t index = threadIdx.x; index < description.count;
         index += blockDim.x) {
        uint64_t packed = canonical[description.source_offset + index];
        uint64_t mask = packed & PACKED_CANONICAL_MASK;
        mask = transform_pair_mask_shared(mask, pair_images);
        uint16_t prefix;
        PrefixSuffix suffix;
        split_pair_mask(mask, prefix, suffix);
        uint32_t ordinal = uint32_t(packed >> PACKED_CANONICAL_WEIGHT_SHIFT);
        if (ordinal >= description.weight_count) {
            atomicExch(failure, 1U);
            continue;
        }
        uint32_t bucket = dense_bucket_map[dense_base + prefix];
        uint32_t destination = atomicAdd(
            &candidate_positions[
                bucket_aux[bucket].candidate_offset + ordinal],
            1U);
        output[destination] = suffix;
    }
}
#endif

template <typename PrefixHistogram, typename WeightHistogram,
          typename ScatterEntries>
static DeviceWeightClassLayout
build_direct_weight_class_layout_from_descriptions(
    std::vector<DirectWeightBuildDesc> descriptions, size_t pair_count,
    uint64_t total_entries, const uint32_t* class_weights,
    const uint8_t* class_orbit_sizes,
    DirectWeightClassWorkspace& workspace, double start,
    PrefixHistogram&& histogram_prefixes,
    WeightHistogram&& histogram_weights, ScatterEntries&& scatter_entries) {
    if (!class_weights || descriptions.size() != pair_count * 2 ||
        total_entries > uint64_t(UINT32_MAX) + 1) {
        throw std::runtime_error("invalid direct grouped build plan");
    }
    DeviceWeightClassLayout result;
    result.pairs.resize(pair_count);
    result.entry_count = size_t(total_entries);
    if (!result.entry_count) {
        result.build_seconds = seconds_now() - start;
        result.total_seconds = result.build_seconds;
        return result;
    }
    if (descriptions.size() > SIZE_MAX / PREFIX_BUCKET_COUNT ||
        descriptions.size() * size_t(PREFIX_BUCKET_COUNT) >
            SIZE_MAX / sizeof(uint32_t)) {
        throw std::overflow_error("direct dense prefix map is too large");
    }
    size_t dense_count = descriptions.size() * size_t(PREFIX_BUCKET_COUNT);
    workspace.dense.reserve(dense_count);
    workspace.occupied_counts.reserve(descriptions.size());
    workspace.failure.reserve(1);
    workspace.descriptions.reserve(descriptions.size());
    uint32_t* dense = workspace.dense.get();
    uint32_t* occupied_counts = workspace.occupied_counts.get();
    uint32_t* failure = workspace.failure.get();
    DirectWeightBuildDesc* device_descriptions = workspace.descriptions.get();
    CUDA_CHECK(cudaMemset(dense, 0, dense_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(failure, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(device_descriptions, descriptions.data(),
                          descriptions.size() * sizeof(DirectWeightBuildDesc),
                          cudaMemcpyHostToDevice));
    histogram_prefixes(device_descriptions, dense);
    CUDA_CHECK(cudaGetLastError());
    count_direct_occupied_prefixes<<<unsigned(descriptions.size()), THREADS>>>(
        dense, occupied_counts);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> host_occupied(descriptions.size());
    CUDA_CHECK(cudaMemcpy(host_occupied.data(), occupied_counts,
                          host_occupied.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    uint64_t total_buckets = 0;
    uint64_t total_candidates = 0;
    for (size_t logical = 0; logical < descriptions.size(); logical++) {
        uint32_t occupied = host_occupied[logical];
        if (total_buckets + occupied > UINT32_MAX) {
            throw std::overflow_error("direct grouped bucket count overflow");
        }
        uint64_t candidates =
            uint64_t(occupied) * descriptions[logical].weight_count;
        if (total_candidates + candidates > UINT32_MAX) {
            throw std::overflow_error(
                "direct grouped candidate scratch count overflow");
        }
        descriptions[logical].bucket_base = uint32_t(total_buckets);
        descriptions[logical].bucket_count = occupied;
        descriptions[logical].candidate_base = uint32_t(total_candidates);
        PrefixDistribution& distribution = logical & 1
            ? result.pairs[logical / 2].complement
            : result.pairs[logical / 2].selected;
        distribution.direct_offset = descriptions[logical].destination_offset;
        distribution.entry_count = descriptions[logical].count;
        distribution.bucket_offset = uint32_t(total_buckets);
        distribution.bucket_count = occupied;
        total_buckets += occupied;
        total_candidates += candidates;
    }
    result.bucket_count = size_t(total_buckets);
    CUDA_CHECK(cudaMemcpy(device_descriptions, descriptions.data(),
                          descriptions.size() * sizeof(DirectWeightBuildDesc),
                          cudaMemcpyHostToDevice));
    workspace.bucket_aux.reserve(result.bucket_count);
    DirectBucketAux* bucket_aux = workspace.bucket_aux.get();
    result.buckets.reserve(result.bucket_count);
    build_direct_prefix_metadata<<<unsigned(descriptions.size()), 32>>>(
        device_descriptions, dense, result.buckets.get(), bucket_aux, failure);
    CUDA_CHECK(cudaGetLastError());

    size_t candidate_count = size_t(total_candidates);
    result.candidate_slots = candidate_count;
    result.fixed_candidate_slots =
        result.bucket_count * size_t(WEIGHT_CLASS_HASH_SLOTS);
    workspace.candidates.reserve(std::max<size_t>(1, candidate_count));
    uint32_t* candidates = workspace.candidates.get();
    CUDA_CHECK(cudaMemset(candidates, 0,
                          candidate_count * sizeof(uint32_t)));
    histogram_weights(device_descriptions, dense, bucket_aux, candidates,
                      failure);
    CUDA_CHECK(cudaGetLastError());

    workspace.class_offsets.reserve(result.bucket_count + 1);
    workspace.maximum.reserve(1);
    uint32_t* class_offsets = workspace.class_offsets.get();
    uint32_t* maximum = workspace.maximum.get();
    CUDA_CHECK(cudaMemset(class_offsets + result.bucket_count, 0,
                          sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(maximum, 0, sizeof(uint32_t)));
    unsigned metadata_blocks = unsigned(std::min<size_t>(
        65535, (result.bucket_count + THREADS - 1) / THREADS));
    count_direct_classes<<<metadata_blocks, THREADS>>>(
        candidates, bucket_aux, result.bucket_count, class_offsets, maximum);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<uint32_t> offsets_ptr(class_offsets);
    thrust::exclusive_scan(offsets_ptr,
                           offsets_ptr + result.bucket_count + 1,
                           offsets_ptr);
    uint32_t class_count = 0;
    uint32_t maximum_classes = 0;
    uint32_t failed = 0;
    CUDA_CHECK(cudaMemcpy(&class_count,
                          class_offsets + result.bucket_count,
                          sizeof(class_count), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&maximum_classes, maximum,
                          sizeof(maximum_classes), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&failed, failure, sizeof(failed),
                          cudaMemcpyDeviceToHost));
    if (failed) {
        throw std::runtime_error("direct grouped histogram mismatch");
    }
    result.class_count = class_count;
    result.maximum_classes = maximum_classes;
    result.suffixes.reserve(result.entry_count);
    result.classes.reserve(std::max<size_t>(1, result.class_count));
    constexpr unsigned warps_per_block = THREADS / 32;
    unsigned bucket_blocks = unsigned(
        (result.bucket_count + warps_per_block - 1) / warps_per_block);
    build_direct_weight_classes<<<bucket_blocks, THREADS>>>(
        result.buckets.get(), bucket_aux, result.bucket_count, candidates,
        class_weights, class_orbit_sizes, class_offsets, result.classes.get());
    CUDA_CHECK(cudaGetLastError());
    scatter_entries(device_descriptions, dense, bucket_aux, candidates,
                    result.suffixes.get(), failure);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&failed, failure, sizeof(failed),
                          cudaMemcpyDeviceToHost));
    if (failed) throw std::runtime_error("direct grouped scatter mismatch");
    result.build_seconds = seconds_now() - start;
    result.total_seconds = result.build_seconds;
    return result;
}

#endif
