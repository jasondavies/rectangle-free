// Optional external-counter test. Compile with the patched SharpSAT-TD src
// include path and its clhash.c object; no third-party source is vendored.
#include <cassert>
#include <vector>
#include "hasher.h"
#include "component_types/component.h"
#define private public
#include "component_types/cacheable_component.h"
#undef private

struct TestNumber { unsigned long InternalSize() const { return 0; } };

int main() {
    std::mt19937_64 gen(501);
    Hasher hasher(gen);
    Component a,b;
    for (auto p : {&a,&b}) {
        p->addVar(1);p->addVar(2);p->closeVariableData();
    }
    a.addCl(3);a.closeClauseData();
    b.addCl(4);b.closeClauseData();
    CacheableComponent<TestNumber> ca(a,hasher),same(a,hasher),cb(b,hasher);
    assert(ca.equals(same));
    cb.clhashkey_=ca.clhashkey_; // Force a full 128-bit collision.
    assert(!ca.equals(cb));
    assert(!cb.equals(ca));
    assert(ca.SizeInBytes()>=sizeof(ca)+a.RawData().size()*sizeof(unsigned));
    assert(ca.sys_overhead_SizeInBytes()>=ca.SizeInBytes());
    a.clear(); // Entries own their key, not a reference to a reused Component.
    assert(ca.equals(same));
    cb.exact_key_=ca.exact_key_;
    cb.exact_key_.push_back(0);
    assert(!ca.equals(cb));
}
