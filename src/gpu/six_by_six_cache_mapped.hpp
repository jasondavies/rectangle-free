#pragma once

#include "six_by_six_cache_artifact.hpp"

#include <algorithm>
#include <array>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class MappedSixBySixCache {
  public:
    explicit MappedSixBySixCache(const std::string& path) {
        descriptor_ = open(path.c_str(), O_RDONLY);
        struct stat status{};
        if (descriptor_ < 0 || fstat(descriptor_, &status) ||
            status.st_size < (off_t)sizeof(six_by_six_cache::Header)) {
            throw std::runtime_error("cannot open six-by-six cache artifact");
        }
        bytes_ = size_t(status.st_size);
        mapping_ = static_cast<const uint8_t*>(mmap(
            nullptr, bytes_, PROT_READ, MAP_SHARED, descriptor_, 0));
        if (mapping_ == MAP_FAILED)
            throw std::runtime_error("cannot map six-by-six cache artifact");
        header_ = reinterpret_cast<const six_by_six_cache::Header*>(mapping_);
        validate();
        keys = pointer<uint64_t>(header_->keys_offset,
                                 header_->canonical_count);
        descriptors = pointer<six_by_six_cache::Descriptor>(
            header_->descriptors_offset, header_->canonical_count);
        masks = pointer<uint32_t>(header_->masks_offset,
                                  header_->entry_count);
        ordinals = pointer<uint8_t>(header_->ordinals_offset,
                                    header_->entry_count);
        class_weights = pointer<uint32_t>(
            header_->class_weights_offset,
            header_->canonical_count * header_->class_slots);
        class_orbits = pointer<uint8_t>(
            header_->class_orbits_offset,
            header_->canonical_count * header_->class_slots);
        class_counts = pointer<uint8_t>(header_->class_counts_offset,
                                        header_->canonical_count);
        references = pointer<uint32_t>(header_->references_offset,
                                       header_->multiset_count);
    }

    ~MappedSixBySixCache() {
        if (mapping_ && mapping_ != MAP_FAILED) munmap(
            const_cast<uint8_t*>(mapping_), bytes_);
        if (descriptor_ >= 0) close(descriptor_);
    }

    MappedSixBySixCache(const MappedSixBySixCache&) = delete;
    MappedSixBySixCache& operator=(const MappedSixBySixCache&) = delete;

    const six_by_six_cache::Header& header() const { return *header_; }

    CanonicalFactory factory() const {
        CanonicalFactory result{};
        result.columns = 6;
        result.canonical_keys.assign(
            keys, keys + six_by_six_cache::CANONICAL_COUNT);
        result.descriptors.resize(six_by_six_cache::CANONICAL_COUNT);
        for (size_t index = 0; index < result.descriptors.size(); ++index) {
            result.descriptors[index] = CanonicalDescriptor{
                descriptors[index].offset, descriptors[index].count};
        }
        return result;
    }

    std::array<CanonicalRef, 2> resolve(PrefixKey raw) const {
        std::array<uint8_t, 6> columns = column_vectors(raw);
        const uint32_t selected = references[multiset_rank(columns)];
        for (uint8_t& value : columns) value ^= 63U;
        const uint32_t complement = references[multiset_rank(columns)];
        return {unpack_reference(selected), unpack_reference(complement)};
    }

    const uint64_t* keys = nullptr;
    const six_by_six_cache::Descriptor* descriptors = nullptr;
    const uint32_t* masks = nullptr;
    const uint8_t* ordinals = nullptr;
    const uint32_t* class_weights = nullptr;
    const uint8_t* class_orbits = nullptr;
    const uint8_t* class_counts = nullptr;
    const uint32_t* references = nullptr;

  private:
    template <typename T>
    const T* pointer(uint64_t offset, uint64_t count) const {
        if (offset > bytes_ || count > (bytes_ - offset) / sizeof(T))
            throw std::runtime_error("six-by-six cache section is truncated");
        return reinterpret_cast<const T*>(mapping_ + offset);
    }

    void validate() const {
        if (std::memcmp(header_->magic, "R6C6Q01", 8) ||
            header_->version != six_by_six_cache::FORMAT_VERSION ||
            header_->rows != six_by_six_cache::ROWS ||
            header_->columns != six_by_six_cache::COLUMNS ||
            header_->class_slots != six_by_six_cache::CLASS_SLOTS ||
            header_->canonical_count != six_by_six_cache::CANONICAL_COUNT ||
            header_->entry_count != six_by_six_cache::ENTRY_COUNT ||
            header_->multiset_count != six_by_six_cache::MULTISET_COUNT ||
            header_->file_bytes != bytes_) {
            throw std::runtime_error("invalid six-by-six cache artifact");
        }
    }

    static const std::array<uint32_t, 720>& row_maps() {
        static const std::array<uint32_t, 720> result = [] {
            std::array<uint32_t, 720> maps{};
            std::array<uint8_t, 6> permutation{0, 1, 2, 3, 4, 5};
            size_t index = 0;
            do {
                uint32_t packed = 0;
                for (unsigned row = 0; row < 6; ++row)
                    packed |= uint32_t(permutation[row]) << (4 * row);
                maps[index++] = packed;
            } while (std::next_permutation(permutation.begin(),
                                           permutation.end()));
            if (index != maps.size())
                throw std::logic_error("bad six-row permutation table");
            return maps;
        }();
        return result;
    }

    static std::array<uint8_t, 6> column_vectors(PrefixKey raw) {
        std::array<uint8_t, 6> rows{};
        for (int row = 5; row >= 0; --row) {
            rows[size_t(row)] = uint8_t(raw & 63U);
            raw >>= 6;
        }
        std::array<uint8_t, 6> result{};
        for (unsigned column = 0; column < 6; ++column)
            for (unsigned row = 0; row < 6; ++row)
                result[column] |= uint8_t((rows[row] >> column) & 1U) << row;
        return result;
    }

    static uint32_t multiset_rank(std::array<uint8_t, 6> values) {
        static const std::array<std::array<uint64_t, 7>, 70> choose = [] {
            std::array<std::array<uint64_t, 7>, 70> table{};
            table[0][0] = 1;
            for (unsigned n = 1; n < table.size(); ++n) {
                table[n][0] = 1;
                for (unsigned k = 1; k <= 6; ++k)
                    table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            }
            return table;
        }();
        std::sort(values.begin(), values.end());
        uint64_t rank = 0;
        for (unsigned index = 0; index < values.size(); ++index)
            rank += choose[unsigned(values[index]) + index][index + 1];
        if (rank >= six_by_six_cache::MULTISET_COUNT)
            throw std::logic_error("six-column multiset rank overflow");
        return uint32_t(rank);
    }

    static CanonicalRef unpack_reference(uint32_t packed) {
        const uint32_t distribution = packed >> 10;
        const uint32_t permutation = packed & 1023U;
        if (distribution >= six_by_six_cache::CANONICAL_COUNT ||
            permutation >= 720) {
            throw std::runtime_error("invalid six-by-six canonical reference");
        }
        return CanonicalRef{distribution, row_maps()[permutation]};
    }

    int descriptor_ = -1;
    size_t bytes_ = 0;
    const uint8_t* mapping_ = nullptr;
    const six_by_six_cache::Header* header_ = nullptr;
};
