// Geometry-specific catalog; execution and arithmetic are shared.
#include "six_by_twenty_eight_catalog.hpp"
#include "hafnian_residual_engine.cuh"

struct Campaign28 {
    using Query=six_by_twenty_eight::Query;
    using Catalog=six_by_twenty_eight::Catalog;
    static constexpr unsigned WIDTH=28;
    static constexpr const char* FORMAT="six-by-twenty-eight-hafnian-v2";
    static constexpr const char* CONTROL_FORMAT="six-by-twenty-eight-hafnian-v1";
    static Catalog build_catalog() { return six_by_twenty_eight::build_catalog(); }
};

int main(int argc,char** argv) {
    return hafnian_residual::Engine<Campaign28>{}.main(argc,argv);
}
