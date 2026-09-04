// Geometry-specific catalog; execution and arithmetic are shared.
#include "six_by_twenty_nine_optimized_catalog.hpp"
#include "hafnian_residual_engine.cuh"

struct Campaign29 {
    using Query=six_by_twenty_nine_optimized::Query;
    using Catalog=six_by_twenty_nine_optimized::Catalog;
    static constexpr unsigned WIDTH=29;
    static constexpr const char* FORMAT="six-by-twenty-nine-hafnian-v2";
    static constexpr const char* CONTROL_FORMAT="six-by-twenty-nine-hafnian-v1";
    static Catalog build_catalog() { return six_by_twenty_nine_optimized::build_catalog(); }
};

int main(int argc,char** argv) {
    return hafnian_residual::Engine<Campaign29>{}.main(argc,argv);
}
