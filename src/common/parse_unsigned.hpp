#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rectangle {
// No signs, whitespace, partial parses or saturating overflow. Shared by
// manifest and CLI readers; zero/end-of-file semantics remain caller-owned.
inline uint64_t parse_u64(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("empty unsigned integer");
    uint64_t value = 0;
    for (unsigned char ch : text) {
        if (ch < '0' || ch > '9' ||
            value > (std::numeric_limits<uint64_t>::max() - (ch - '0')) / 10)
            throw std::invalid_argument("invalid unsigned integer: " + text);
        value = value * 10 + (ch - '0');
    }
    return value;
}
inline uint32_t parse_u32(const std::string& text) {
    uint64_t value = parse_u64(text);
    if (value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("unsigned integer exceeds 32 bits: " + text);
    return static_cast<uint32_t>(value);
}
} // namespace rectangle
