#include "../../src/hafnian/hafnian_term_reference.hpp"
#include "../../src/hafnian/six_by_twenty_nine_catalog.hpp"
#include <iostream>

using Query=six_by_twenty_nine::Query;
uint64_t brute(const Query& q,uint64_t mask){
    if(!mask)return 1;
    unsigned first=__builtin_ctzll(mask);mask&=mask-1;uint64_t sum=0;
    for(uint64_t scan=mask;scan;scan&=scan-1){unsigned j=__builtin_ctzll(scan);
        if(q.adjacency[first*q.vertices+j])sum+=brute(q,mask^(UINT64_C(1)<<j));}
    return sum;
}
int main(){
    uint64_t random=486;unsigned checks=0;
    for(unsigned n:{2u,4u,6u,8u,10u,12u})for(unsigned trial=0;trial<12;++trial){
        Query q;q.vertices=n;q.adjacency.resize(n*n);
        for(unsigned i=0;i<n;++i)for(unsigned j=0;j<i;++j){
            random=random*UINT64_C(6364136223846793005)+1;
            q.adjacency[i*n+j]=q.adjacency[j*n+i]=unsigned((random>>32)%3!=0);}
        auto want=brute(q,(UINT64_C(1)<<n)-1);
        for(uint32_t prime:{2147483647u,2147483629u,2147483587u,2147483579u}){
            hafnian_reference::Mod m{prime};uint32_t sum=0;unsigned domain=1u<<(n/2-1);
            for(unsigned i=0;i<domain;++i)sum=m.add(sum,hafnian_reference::term(q,i,prime));
            if(m.mul(sum,m.inverse(domain))!=want)throw std::runtime_error("reference/brute mismatch");
            ++checks;
        }
    }
    using Wide=unsigned __int128;
    std::vector<Wide> neighbours(66);
    for(unsigned i=0;i<66;++i)neighbours[i]=Wide(1)<<(i^1);
    std::vector<std::pair<unsigned,unsigned>> matching;
    if(!six_by_twenty_nine::find_perfect_matching(neighbours,(Wide(1)<<66)-1,matching)||matching.size()!=33)
        throw std::runtime_error("wide matching mismatch");
    six_by_twenty_nine::Geometry geometry;
    Query q;q.unmatched=6;six_by_twenty_nine::build_query_graph(geometry,q,27,true);
    if(q.vertices!=66||q.order.size()!=66||q.adjacency.size()!=4356)
        throw std::runtime_error("order66 preparation mismatch");
    std::cout<<"TAIL_REFERENCE brute_checks="<<checks<<" primes=4 wide_matching=OK order66=OK\n";
}
