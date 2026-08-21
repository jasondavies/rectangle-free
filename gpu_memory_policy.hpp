#ifndef RECTANGLE_FREE_GPU_MEMORY_POLICY_HPP
#define RECTANGLE_FREE_GPU_MEMORY_POLICY_HPP

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_memory_policy {

constexpr size_t gib(size_t value) {
    return value << 30;
}

inline size_t environment_mib(const char* name, size_t fallback_bytes) {
    const char* text = std::getenv(name);
    if (!text || !*text) return fallback_bytes;
    errno = 0;
    char* end = nullptr;
    unsigned long long mib = std::strtoull(text, &end, 10);
    if (errno || end == text || *end ||
        mib > std::numeric_limits<size_t>::max() / (size_t(1) << 20)) {
        throw std::runtime_error(std::string("invalid ") + name +
                                 " MiB value");
    }
    return size_t(mib) << 20;
}

inline size_t reserve_bytes(size_t fallback_bytes) {
    return environment_mib("RECT_GPU_MEMORY_RESERVE_MIB", fallback_bytes);
}

// A resident canonical cache is useful only if it leaves enough room for the
// recurring right layout, persistent left layouts, builder scratch and result
// buffers.  This deliberately rejects a cache that merely fits: starving the
// right layout created many more batches and was slower in production.
inline bool prefer_resident_cache(size_t free_bytes, size_t cache_bytes,
                                  size_t reserve,
                                  size_t recurring_headroom) {
    return cache_bytes <= free_bytes &&
           recurring_headroom <= free_bytes - cache_bytes &&
           reserve <= free_bytes - cache_bytes - recurring_headroom;
}

}  // namespace gpu_memory_policy

#endif
