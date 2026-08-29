#pragma once

#include <cstddef>
#include <cstdint>

namespace six_by_six_cache {

constexpr uint32_t FORMAT_VERSION = 1;
constexpr uint32_t ROWS = 6;
constexpr uint32_t COLUMNS = 6;
constexpr uint32_t CLASS_SLOTS = 32;
constexpr uint64_t CANONICAL_COUNT = 251610;
constexpr uint64_t ENTRY_COUNT = UINT64_C(3469067567);
constexpr uint64_t MULTISET_COUNT = UINT64_C(119877472);

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t rows;
    uint32_t columns;
    uint32_t class_slots;
    uint64_t canonical_count;
    uint64_t entry_count;
    uint64_t multiset_count;
    uint64_t keys_offset;
    uint64_t descriptors_offset;
    uint64_t masks_offset;
    uint64_t ordinals_offset;
    uint64_t class_weights_offset;
    uint64_t class_orbits_offset;
    uint64_t class_counts_offset;
    uint64_t references_offset;
    uint64_t file_bytes;
    uint8_t reserved[8];
};
static_assert(sizeof(Header) == 128,
              "six-by-six cache header must have a stable ABI");

struct Descriptor {
    uint64_t offset;
    uint32_t count;
    uint32_t reserved;
};
static_assert(sizeof(Descriptor) == 16,
              "six-by-six cache descriptors must have a stable ABI");

inline uint64_t align_up(uint64_t value, uint64_t alignment = 4096) {
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace six_by_six_cache
