// Endpoint Laplace minor; execution and arithmetic are shared with 6x29/6x28.
#include "six_by_thirty_optimized_catalog.hpp"
#include "hafnian_residual_engine.cuh"

struct Campaign30 {
    using Query=six_by_thirty_optimized::Query;
    using Catalog=six_by_thirty_optimized::Catalog;
    static constexpr unsigned WIDTH=30;
    static constexpr const char* FORMAT="six-by-thirty-hafnian-v2";
    static constexpr const char* CONTROL_FORMAT="six-by-thirty-edge-minor-control-v1";
    static Catalog build_catalog() { return six_by_thirty_optimized::build_catalog(); }
};

int main(int argc,char** argv) {
    return hafnian_residual::Engine<Campaign30>{}.main(argc,argv);
}
