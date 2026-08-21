#ifndef RECTANGLE_FREE_SHA256_HPP
#define RECTANGLE_FREE_SHA256_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

class Sha256 {
  public:
    Sha256()
        : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const void* source, size_t length) {
        const auto* bytes = static_cast<const uint8_t*>(source);
        total_bytes_ += length;
        while (length) {
            size_t take = std::min(length, block_.size() - buffered_);
            std::memcpy(block_.data() + buffered_, bytes, take);
            buffered_ += take;
            bytes += take;
            length -= take;
            if (buffered_ == block_.size()) {
                compress(block_.data());
                buffered_ = 0;
            }
        }
    }

    void update(const std::string& value) { update(value.data(), value.size()); }

    std::string finish_hex() {
        const uint64_t bit_count = total_bytes_ * 8;
        block_[buffered_++] = 0x80;
        if (buffered_ > 56) {
            std::fill(block_.begin() + buffered_, block_.end(), uint8_t(0));
            compress(block_.data());
            buffered_ = 0;
        }
        std::fill(block_.begin() + buffered_, block_.begin() + 56, uint8_t(0));
        for (unsigned index = 0; index < 8; index++) {
            block_[63 - index] = uint8_t(bit_count >> (8 * index));
        }
        compress(block_.data());
        std::ostringstream result;
        result << std::hex << std::setfill('0');
        for (uint32_t word : state_) result << std::setw(8) << word;
        return result.str();
    }

  private:
    static uint32_t rotate(uint32_t value, unsigned shift) {
        return (value >> shift) | (value << (32 - shift));
    }

    void compress(const uint8_t* input) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        uint32_t words[64];
        for (unsigned index = 0; index < 16; index++) {
            words[index] = uint32_t(input[4 * index]) << 24 |
                           uint32_t(input[4 * index + 1]) << 16 |
                           uint32_t(input[4 * index + 2]) << 8 |
                           uint32_t(input[4 * index + 3]);
        }
        for (unsigned index = 16; index < 64; index++) {
            uint32_t x = words[index - 15];
            uint32_t y = words[index - 2];
            uint32_t s0 = rotate(x, 7) ^ rotate(x, 18) ^ (x >> 3);
            uint32_t s1 = rotate(y, 17) ^ rotate(y, 19) ^ (y >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned index = 0; index < 64; index++) {
            uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            uint32_t choose = (e & f) ^ (~e & g);
            uint32_t t1 = h + s1 + choose + constants[index] + words[index];
            uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> block_{};
    size_t buffered_ = 0;
    uint64_t total_bytes_ = 0;
};

static std::string sha256_string(const std::string& value) {
    Sha256 hash;
    hash.update(value);
    return hash.finish_hex();
}

static std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash " + path);
    Sha256 hash;
    std::array<char, 1 << 20> buffer;
    while (input) {
        input.read(buffer.data(), buffer.size());
        std::streamsize count = input.gcount();
        if (count > 0) hash.update(buffer.data(), size_t(count));
    }
    if (!input.eof()) throw std::runtime_error("failed hashing " + path);
    return hash.finish_hex();
}

#endif
