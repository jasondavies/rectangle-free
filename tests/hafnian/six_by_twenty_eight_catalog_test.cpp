#include <array>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>

#include "../../src/hafnian/six_by_twenty_eight_catalog.hpp"

int main() try {
    auto catalog=six_by_twenty_eight::build_catalog();
    const std::array<uint32_t,4> primes={
        2147483647U,2147483629U,2147483587U,2147483579U};
    unsigned __int128 modulus=1;
    std::array<unsigned,5> prime_histogram{};
    std::map<std::pair<unsigned,unsigned>,std::array<uint64_t,2>> sectors;
    for(const auto& query:catalog.queries) {
        unsigned needed=0;
        unsigned __int128 target=static_cast<unsigned __int128>(1)
            <<query.matching_bound_power;
        unsigned __int128 current=1;
        while(current<=target) {
            if(needed==primes.size())
                throw std::runtime_error("four primes do not certify query");
            current*=primes[needed++];
        }
        ++prime_histogram[needed];
        auto& sector=sectors[{query.excess,query.defect_count}];
        ++sector[0];
        sector[1]+=query.defect_coefficient;
    }
    for(uint32_t prime:primes)modulus*=prime;
    std::printf(
        "HAFNIAN_6X28_CATALOG queries=%zu digest=%s "
        "prime_histogram_2=%u prime_histogram_3=%u prime_histogram_4=%u "
        "modulus_bits=%u sectors=%zu exact=OK\n",
        catalog.queries.size(),catalog.digest.c_str(),prime_histogram[2],
        prime_histogram[3],prime_histogram[4],
        128U-unsigned(__builtin_clzll(uint64_t(modulus>>64))),sectors.size());
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"error: %s\n",error.what());
    return 2;
}
