#ifndef RECTANGLE_FREE_SHA256_C_H
#define RECTANGLE_FREE_SHA256_C_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    uint32_t state[8];
    uint8_t block[64];
    size_t buffered;
    uint64_t bytes;
} RectSha256;
void rect_sha256_init(RectSha256* h);
void rect_sha256_update(RectSha256* h, const void* data, size_t length);
void rect_sha256_finish(RectSha256* h, char hex[65]);
#endif
