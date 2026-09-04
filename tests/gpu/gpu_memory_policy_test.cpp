#include "gpu_memory_policy.hpp"

#include <cassert>
#include <iostream>

using Memory = gpu_memory_policy::BatchMemory<3>;

int main() {
    Memory metadata_heavy{{80, 10, 20}, 5};
    Memory entry_heavy{{10, 80, 5}, 20};
    // Each fresh batch fits 150 bytes; their retained component maxima do not.
    assert(metadata_heavy.bytes() == 115);
    assert(entry_heavy.bytes() == 115);
    auto retained = entry_heavy.retaining(metadata_heavy);
    assert(retained.bytes() == 200);
    assert(retained.persistent[2] == 20); // join/result high water also persists
    assert(entry_heavy.retaining(Memory{}).bytes() == 115); // boundary reset

    Memory output_heavy{{1, 1, 1}, 1000};
    assert(entry_heavy.retaining(output_heavy).bytes() == 115);
    // Old right-layout outputs must not persist into the next batch.
    Memory small{{1, 2, 3}, 4};
    auto after_small = small.retaining(retained);
    assert(after_small.bytes() == 184);
    assert(small.retaining(after_small).bytes() == 184);

    Memory group_sum = metadata_heavy;
    group_sum += entry_heavy;
    assert(group_sum.bytes() == 230);
    assert(group_sum.persistent[0] == 90 && group_sum.transient == 25);
    assert(Memory{}.bytes() == 0);

    bool overflow = false;
    try { (void)Memory{{UINT64_MAX, 1, 0}, 0}.bytes(); }
    catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
    overflow = false;
    try {
        Memory huge{{UINT64_MAX, 0, 0}, 0};
        huge += Memory{{1, 0, 0}, 0};
    } catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
    std::cout << "GPU memory policy: OK\n";
}
