#include "../../research/probes/partition_append_block.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    for (uint32_t x = 0; x < 65536; ++x) {
        uint32_t expected = 0;
        for (unsigned r = 0; r < 4; ++r)
            for (unsigned c = 0; c < 4; ++c)
                expected |= ((x >> (4*r+c)) & 1u) << (4*c+r);
        assert(partition_transpose4(x) == expected);
        assert(partition_transpose4(expected) == x);
    }
    puts("65536 overlap-block transposes exact=OK");
    return 0;
}
