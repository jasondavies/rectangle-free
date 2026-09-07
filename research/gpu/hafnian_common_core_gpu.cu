// Isolated exact partial-core prototype. One CTA per common-core sign.
// Compile with -DCORE_HOST_EMULATION using g++ for the same cooperative
// arithmetic and barriers on OpenMP threads. This is not GPU emulation of
// scheduling/occupancy and supplies no GPU timing evidence.
#define COMMON_CORE_NO_MAIN
#include "../probes/hafnian_common_core_bench.cpp"
#include "../../src/hafnian/hafnian_common_core.cuh"
#include "../../src/hafnian/hafnian_common_core_runner.cuh"
namespace core_gpu {
template<uint32_t P>
std::vector<uint32_t> execute(const Input& in,uint64_t begin,unsigned count,unsigned threads,double& elapsed) {
    std::vector<uint32_t> out(size_t(count)*in.queries);
#ifdef CORE_HOST_EMULATION
    std::vector<uint32_t> scratch(in.words);
    auto started=Clock::now();
    for(unsigned i=0;i<count;++i){uint64_t index=begin+i;
        #pragma omp parallel num_threads(threads)
        term<P>(in,index^(index>>1),scratch.data(),out.data()+size_t(i)*in.queries,
            unsigned(omp_get_thread_num()),unsigned(omp_get_num_threads()));
    }
    elapsed=common_bench::seconds(started,Clock::now());
#else
    Input* device_in=nullptr;uint32_t* device_out=nullptr;
    uint64_t* device_phases=nullptr;
#if CORE_PROFILE
    checked(cudaMalloc(&device_phases,size_t(count)*6*sizeof(uint64_t)));
#endif
    checked(cudaMalloc(&device_in,sizeof(Input)));checked(cudaMalloc(&device_out,out.size()*sizeof(uint32_t)));
    checked(cudaMemcpy(device_in,&in,sizeof(Input),cudaMemcpyHostToDevice));
    checked(cudaFuncSetAttribute(kernel<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,in.words*sizeof(uint32_t)));
    int active=0;checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<P>,threads,in.words*sizeof(uint32_t)));
    cudaEvent_t start,stop;checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
    checked(cudaEventRecord(start));
    kernel<P><<<count,threads,in.words*sizeof(uint32_t)>>>(device_in,begin,device_out,device_phases);
    checked(cudaGetLastError());checked(cudaEventRecord(stop));checked(cudaEventSynchronize(stop));
    float ms=0;checked(cudaEventElapsedTime(&ms,start,stop));elapsed=ms/1000.0;
    checked(cudaMemcpy(out.data(),device_out,out.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
#if CORE_PROFILE
    std::vector<uint64_t> phases(size_t(count)*6);
    checked(cudaMemcpy(phases.data(),device_phases,phases.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    uint64_t cycles[5]{};
    for(unsigned i=0;i<count;++i)for(unsigned j=0;j<5;++j)cycles[j]+=phases[i*6+j+1]-phases[i*6+j];
    std::printf("CORE_CUDA_PHASE cycles_init_hess=%llu cycles_poly=%llu cycles_moments=%llu cycles_boundary=%llu cycles_output=%llu scope=summed_CTA_elapsed_cycles_instrumented\n",
        (unsigned long long)cycles[0],(unsigned long long)cycles[1],(unsigned long long)cycles[2],
        (unsigned long long)cycles[3],(unsigned long long)cycles[4]);
    checked(cudaFree(device_phases));
#endif
    std::printf("CORE_CUDA_LAUNCH prime=%u core=%u pool=%u queries=%u states=%u shared_bytes=%zu threads=%u active_ctas_per_sm=%d signs=%u kernel_s=%.6f\n",
        P,in.c,in.q,in.queries,in.states,in.words*sizeof(uint32_t),threads,active,count,elapsed);
    checked(cudaEventDestroy(start));checked(cudaEventDestroy(stop));checked(cudaFree(device_out));checked(cudaFree(device_in));
#endif
    return out;
}


template<uint32_t P>
void check_reduced(const Input& in,uint64_t begin,unsigned count,unsigned chunk,
        unsigned threads,const std::vector<uint32_t>& raw) {
    static ReducedRunner<P> runner;
    auto started=Clock::now();double elapsed=0;
    auto totals=runner.run(in,begin,count,chunk,threads,elapsed);
    double wall=common_bench::seconds(started,Clock::now());
    uint64_t checksum=0;
    for(unsigned j=0;j<in.queries;++j){uint32_t expected=0;
        for(unsigned i=0;i<count;++i)expected=Mod{P}.add(expected,raw[size_t(i)*in.queries+j]);
        if(totals[j]!=expected)throw std::runtime_error("device/host modular sum mismatch");
        checksum=checksum*UINT64_C(6364136223846793005)+totals[j];
    }
    std::printf("CORE_REDUCED_AB prime=%u core=%u pool=%u queries=%u signs=%u chunk=%u chunks=%llu raw_download_bytes=%llu reduced_download_bytes=%zu device_s=%.9f wall_s=%.9f checksum=%llu all_signs=OK scope=prepared_group_including_reduction\n",
        P,in.c,in.q,in.queries,count,chunk,(unsigned long long)((uint64_t(count)+chunk-1)/chunk),
        (unsigned long long)(uint64_t(count)*in.queries*4),totals.size()*4,elapsed,wall,(unsigned long long)checksum);
}

void check_samples(const common_bench::Problem& p,uint32_t prime,uint64_t begin,unsigned count,
                   unsigned threads,bool compare_all,std::vector<uint32_t>& out,double& elapsed,unsigned reduced_chunk=0) {
    auto started=Clock::now();
    auto in=pack(p,prime);
    double prepare_seconds=common_bench::seconds(started,Clock::now());
    switch(prime){
        case 2147483647U:out=execute<2147483647U>(in,begin,count,threads,elapsed);break;
        case 2147483629U:out=execute<2147483629U>(in,begin,count,threads,elapsed);break;
        case 2147483587U:out=execute<2147483587U>(in,begin,count,threads,elapsed);break;
        case 2147483579U:out=execute<2147483579U>(in,begin,count,threads,elapsed);break;
        default:throw std::runtime_error("uncertified field");
    }
    std::printf("CORE_CUDA_EXECUTION prime=%u queries=%zu signs=%u solve_wall_s=%.6f kernel_s=%.6f prepare_s=%.9f\n",
        prime,p.masks.size(),count,common_bench::seconds(started,Clock::now()),elapsed,prepare_seconds);
    common_bench::Workspace w(p,Mod{prime});
    unsigned checks=compare_all?count:std::min(count,128u);
    for(unsigned j=0;j<checks;++j){unsigned i=compare_all?j:unsigned(uint64_t(j)*(count-1)/std::max(1u,checks-1));
        uint64_t index=begin+i;
        auto expected=common_bench::shared_term(p,w,index^(index>>1),Mod{prime});
        for(unsigned k=0;k<expected.size();++k)if(expected[k]!=out[size_t(i)*expected.size()+k])
            throw std::runtime_error("cooperative/CPU mismatch at sign "+std::to_string(index)+" child "+std::to_string(k));
    }
    std::printf("CORE_CUDA_PARITY prime=%u core=%u pool=%u queries=%zu compared_signs=%u exact=OK\n",prime,p.core,p.q,p.masks.size(),checks);
    if(reduced_chunk)switch(prime){
        case 2147483647U:check_reduced<2147483647U>(in,begin,count,reduced_chunk,threads,out);break;
        case 2147483629U:check_reduced<2147483629U>(in,begin,count,reduced_chunk,threads,out);break;
        case 2147483587U:check_reduced<2147483587U>(in,begin,count,reduced_chunk,threads,out);break;
        case 2147483579U:check_reduced<2147483579U>(in,begin,count,reduced_chunk,threads,out);break;
    }
}

void self_test(unsigned threads,unsigned reduced_chunk=0) {
    std::mt19937_64 random(477);unsigned checks=0;
    for(unsigned bad_mask:{3u<<12,1u,4095u}){
        common_bench::Problem invalid{Matrix(13),0,13,{bad_mask}};
        bool rejected=false;try{prepare(invalid);}catch(const std::exception&){rejected=true;}
        if(!rejected)throw std::runtime_error("unsupported boundary mask accepted");
    }
    for(uint32_t prime:{2147483647u,2147483629u,2147483587u,2147483579u}) {
        // Test the pseudo-Mersenne reduction independently, including limits.
        for(unsigned i=0;i<10000;++i){uint32_t a=random()%prime,b=random()%prime;
            if(i<4){a=prime-1-i;b=prime-1;}
#ifdef CORE_HOST_EMULATION
            auto got=prime==2147483647u?Field<2147483647u>::mul(a,b):prime==2147483629u?Field<2147483629u>::mul(a,b):
                prime==2147483587u?Field<2147483587u>::mul(a,b):Field<2147483579u>::mul(a,b);
            if(got!=Mod{prime}.mul(a,b))throw std::runtime_error("field product mismatch");
            if(a){
                auto inv=prime==2147483647u?Field<2147483647u>::inverse(a):prime==2147483629u?Field<2147483629u>::inverse(a):
                    prime==2147483587u?Field<2147483587u>::inverse(a):Field<2147483579u>::inverse(a);
                if(Mod{prime}.mul(a,inv)!=1)throw std::runtime_error("inverse chain mismatch");
            }
            uint64_t pending=4*uint64_t(a)*b;
            auto reduced=prime==2147483647u?Field<2147483647u>::reduce(pending):prime==2147483629u?Field<2147483629u>::reduce(pending):
                prime==2147483587u?Field<2147483587u>::reduce(pending):Field<2147483579u>::reduce(pending);
            if(reduced!=pending%prime)throw std::runtime_error("four-product reduction mismatch");
            pending=48*uint64_t(a);
            reduced=prime==2147483647u?Field<2147483647u>::reduce(pending):prime==2147483629u?Field<2147483629u>::reduce(pending):
                prime==2147483587u?Field<2147483587u>::reduce(pending):Field<2147483579u>::reduce(pending);
            if(reduced!=pending%prime)throw std::runtime_error("neighbour sum reduction mismatch");
#endif
        }
        for(unsigned c:{0u,2u,6u,10u})for(unsigned q:{5u,7u})for(unsigned mode:{0u,1u}) {
            common_bench::Problem p{Matrix(c+q),c,q,{}};
            for(unsigned mask=0;mask<(1u<<q);++mask)if(unsigned(__builtin_popcount(mask))==q-3)p.masks.push_back(mask);
            for(unsigned i=0;i<c+q;++i)for(unsigned j=0;j<i;++j)
                p.adjacency.at(i,j)=p.adjacency.at(j,i)=mode?unsigned(random()%2):0;
            unsigned count=c?1u<<(c/2-1):1;double seconds=0;std::vector<uint32_t> out;
            check_samples(p,prime,0,count,threads,true,out,seconds,reduced_chunk);checks+=count*p.masks.size();
            for(unsigned j=0;j<p.masks.size();++j){uint32_t sum=0;
                for(unsigned i=0;i<count;++i)sum=Mod{prime}.add(sum,out[size_t(i)*p.masks.size()+j]);
                auto single=common_bench::single_problem(p,p.masks[j]);
                auto want=common_bench::brute(single.adjacency,(UINT64_C(1)<<single.adjacency.n)-1,Mod{prime});
                if(Mod{prime}.mul(sum,Mod{prime}.inverse(count))!=want)throw std::runtime_error("complete cooperative/brute mismatch");
            }
        }
    }
    std::printf("CORE_CUDA_SELF_TEST child_sign_checks=%u primes=4 complete_brute=OK exact=OK\n",checks);
}
} // namespace core_gpu

int main(int argc,char** argv)try {
    std::string path;unsigned count=4096,threads=128,cap=11,limit=0,slack=3,order=0;
    unsigned reduced_chunk=0;
    uint32_t prime=2147483647;bool test=false,complete=false,sweep=false;
#ifdef CORE_HOST_EMULATION
    threads=4;count=8;
#endif
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--self-test")test=true;
        else if(a=="--reduced-ab"&&i+1<argc){
            auto value=number(argv[++i]);if(!value||value>(1u<<20))throw std::runtime_error("invalid reduction chunk");
            reduced_chunk=unsigned(value);
        }
        else if(a=="--sweep")sweep=true;
        else if(a=="--complete6x28"){complete=true;slack=2;}
        else if(a=="--groups"&&i+1<argc)path=argv[++i];
        else if(a=="--count"&&i+1<argc)count=unsigned(number(argv[++i]));
        else if(a=="--threads"&&i+1<argc)threads=unsigned(number(argv[++i]));
        else if(a=="--cap"&&i+1<argc)cap=unsigned(number(argv[++i]));
        else if(a=="--order"&&i+1<argc)order=unsigned(number(argv[++i]));
        else if(a=="--query-limit"&&i+1<argc)limit=unsigned(number(argv[++i]));
        else if(a=="--prime"&&i+1<argc)prime=uint32_t(number(argv[++i]));
        else throw std::runtime_error("usage: --self-test | --groups LOG [--count N --cap 7|9|11 --order N --query-limit N --threads N --prime P --complete6x28 --reduced-ab CHUNK]");
    }
    std::printf("CORE_CUDA_CONFIG hess=%d boundary=%d scratch=%d profile=%d warp_poly=%d sparse_moments=%d boundary_order=%d max_pool=%d threads=%u inverse_chain=%d live_moments=%d sync_clear=%d\n",
        CORE_OPT_HESS,CORE_OPT_BOUNDARY,CORE_OPT_SCRATCH,CORE_PROFILE,
        CORE_OPT_WARP_POLY,CORE_OPT_SPARSE_MOMENTS,CORE_BOUNDARY_ORDER,CORE_MAX_POOL,threads,
        CORE_OPT_INVERSE_CHAIN,CORE_OPT_LIVE_MOMENTS,CORE_OPT_SYNC_CLEAR);
    const std::array<uint32_t,4> primes{2147483647,2147483629,2147483587,2147483579};
    auto prime_it=std::find(primes.begin(),primes.end(),prime);
    if(prime_it==primes.end())throw std::runtime_error("uncertified prime");
    unsigned prime_index=unsigned(prime_it-primes.begin());
#ifdef CORE_HOST_EMULATION
    if(!threads||threads>16||count>128||complete)throw std::runtime_error("host gate requires <=16 threads, <=128 signs and no full real group");
#else
    if(threads!=64&&threads!=128&&threads!=256)throw std::runtime_error("CUDA block must have 64/128/256 threads");
    cudaDeviceProp props{};core_gpu::checked(cudaGetDeviceProperties(&props,0));
    std::printf("CORE_CUDA_DEVICE name=%s sm=%d.%d\n",props.name,props.major,props.minor);
#endif
    if(reduced_chunk>(1u<<20))throw std::runtime_error("reduction chunk exceeds bounded gate");
    if(test)core_gpu::self_test(threads,reduced_chunk);
    if(path.empty())return test?0:2;
    auto groups=common_bench::read_groups(path);
    if(sweep){
        if(complete||!count||count>(1u<<20))throw std::runtime_error("invalid sweep range");
        six_by_twenty_nine::Geometry geometry;unsigned cases=0;
        for(const auto& in:groups){
            if(in.prime_index!=UINT32_MAX&&in.prime_index!=prime_index)continue;
            unsigned n=60+2*slack-2*in.e-2*in.d;if(order&&n!=order)continue;
            auto p=common_bench::make_problem(geometry,in,slack);
            uint64_t domain=p.core?UINT64_C(1)<<(p.core/2-1):1;
            unsigned steps=unsigned(std::min(uint64_t(count),domain));
            uint64_t begin=six_by_common_core::hash(in.root)%(domain-steps+1);
            std::vector<uint32_t> out;double elapsed=0;
            core_gpu::check_samples(p,prime,begin,steps,threads,false,out,elapsed,reduced_chunk);
            std::printf("CORE_CUDA_SWEEP order=%u core=%u pool=%u active_queries=%zu prime_index=%u root=%llu signs=%u kernel_s=%.9f exact=OK\n",
                n,p.core,p.q,p.masks.size(),prime_index,(unsigned long long)in.root,steps,elapsed);
            std::fflush(stdout);++cases;
        }
        if(!cases)throw std::runtime_error("no matching sweep samples");return 0;
    }
    std::sort(groups.begin(),groups.end(),[](const auto& a,const auto& b){return six_by_common_core::hash(a.root)<six_by_common_core::hash(b.root);});
    auto it=std::find_if(groups.begin(),groups.end(),[&](const auto& g){return g.cap==cap&&(!order||60+2*slack-2*g.e-2*g.d==order);});
    if(it==groups.end())throw std::runtime_error("no group at requested cap");
    if(limit&&it->members.size()>limit)it->members.resize(limit);
    six_by_twenty_nine::Geometry geometry;
    auto p=common_bench::make_problem(geometry,*it,slack);
    std::printf("CORE_CUDA_GROUP e=%u d=%u root=%llu parent=%llu boundary=%llu core=%u queries=%zu\n",
        it->e,it->d,(unsigned long long)it->root,(unsigned long long)it->parent,
        (unsigned long long)it->boundary,p.core,p.masks.size());
    uint64_t domain=p.core?UINT64_C(1)<<(p.core/2-1):1;
    if(complete)count=unsigned(domain);
    if(!count||count>domain||count>(1u<<20))throw std::runtime_error("sign range exceeds bounded gate");
    uint64_t begin=complete?0:six_by_common_core::hash(it->root)%(domain-count+1);
    double elapsed=0;std::vector<uint32_t> out;
    core_gpu::check_samples(p,prime,begin,count,threads,false,out,elapsed,reduced_chunk);
    if(complete){auto catalog=six_by_twenty_eight::build_catalog();
        for(unsigned j=0;j<p.masks.size();++j){uint32_t sum=0;
            for(unsigned i=0;i<count;++i)sum=Mod{prime}.add(sum,out[size_t(i)*p.masks.size()+j]);
            auto query=std::find_if(catalog.queries.begin(),catalog.queries.end(),[&](const auto& v){return v.occupied==it->members[j].first&&v.excess==it->e&&v.defect_count==it->d;});
            if(query==catalog.queries.end())throw std::runtime_error("query absent from production catalog");
            std::printf("CORE_COMPLETE query_id=%u occupied=%llu prime=%u augmented_hafnian=%u query_digest=%s\n",query->id,
                (unsigned long long)query->occupied,prime,Mod{prime}.mul(sum,Mod{prime}.inverse(count)),query->digest.c_str());
        }
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"error: %s\n",e.what());return 1;}
