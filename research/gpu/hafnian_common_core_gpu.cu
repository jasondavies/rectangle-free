// Isolated exact partial-core prototype. One CTA per common-core sign.
// Compile with -DCORE_HOST_EMULATION using g++ for the same cooperative
// arithmetic and barriers on OpenMP threads. This is not GPU emulation of
// scheduling/occupancy and supplies no GPU timing evidence.
#define COMMON_CORE_NO_MAIN
#include "../probes/hafnian_common_core_bench.cpp"
#ifndef CORE_HOST_EMULATION
#include <cuda_runtime.h>
#define CORE_DEVICE __device__
#define CORE_SYNC() __syncthreads()
#else
#define CORE_DEVICE
#define CORE_SYNC() _Pragma("omp barrier")
#endif

namespace core_gpu {
constexpr unsigned MAX_C=48, MAX_Q=11, MAX_MASK=1u<<MAX_Q;
struct Input {
    unsigned c,q,n,degree,stride,states,queries,words;
    unsigned starts[7]{}; // even boundary subset sizes 0,2,...,10
    unsigned masks[MAX_MASK]{},slot[MAX_MASK]{},answers[256]{};
    uint32_t inverse[49]{};
    uint8_t adjacency[64*64]{};
};

Input pack(const common_bench::Problem& p,uint32_t prime) {
    if(p.core>MAX_C||p.q>MAX_Q||p.adjacency.n>64||p.masks.size()>256)
        throw std::runtime_error("CUDA gate dimensions exceeded");
    Input in{};in.c=p.core;in.q=p.q;in.n=p.adjacency.n;
    in.degree=p.core/2;in.stride=in.degree+1;in.queries=p.masks.size();
    common_bench::Workspace w(p,Mod{prime});
    auto plan=w.plan;
    std::sort(plan.begin(),plan.end(),[](unsigned a,unsigned b){
        return std::make_pair(__builtin_popcount(a),a)<std::make_pair(__builtin_popcount(b),b);});
    in.states=plan.size();
    for(unsigned i=0;i<plan.size();++i){in.masks[i]=plan[i];in.slot[plan[i]]=i;}
    for(unsigned level=0;level<=6;++level) {
        unsigned start=0;while(start<plan.size()&&unsigned(__builtin_popcount(plan[start]))<2*level)++start;
        in.starts[level]=start;
    }
    for(unsigned j=0;j<p.masks.size();++j)in.answers[j]=in.slot[p.masks[j]];
    for(unsigned j=1;j<=p.core;++j)in.inverse[j]=Mod{prime}.inverse(j);
    for(unsigned i=0;i<in.n;++i)for(unsigned j=0;j<in.n;++j)
        in.adjacency[i*in.n+j]=p.adjacency.at(i,j);
    in.words=in.c*in.c+(in.c+1)*in.stride+in.stride+in.c+2+
        in.q*in.q*in.stride+2*in.c*in.q+in.states*in.stride;
    return in;
}

template<uint32_t P> struct Field {
    CORE_DEVICE static uint32_t add(uint32_t a,uint32_t b){uint32_t s=a+b;return s>=P?s-P:s;}
    CORE_DEVICE static uint32_t neg(uint32_t a){return a?P-a:0;}
    CORE_DEVICE static uint32_t sub(uint32_t a,uint32_t b){return a>=b?a-b:P-(b-a);}
    CORE_DEVICE static uint32_t mul(uint32_t a,uint32_t b){
        constexpr uint64_t mask=UINT64_C(2147483647),delta=UINT64_C(2147483648)-P;
        uint64_t t=uint64_t(a)*b;
        // Certified fields are 2^31-d, d<=69. Two folds leave <2p.
        t=(t&mask)+(t>>31)*delta;t=(t&mask)+(t>>31)*delta;
        return uint32_t(t>=P?t-P:t);
    }
    CORE_DEVICE static uint32_t inverse(uint32_t a){
        uint32_t out=1;for(uint32_t e=P-2;e;e>>=1){if(e&1)out=mul(out,a);a=mul(a,a);}return out;
    }
};

template<uint32_t P>
CORE_DEVICE void term(const Input& in,uint64_t signs,uint32_t* scratch,
                      uint32_t* output,unsigned tid,unsigned nt) {
    using F=Field<P>;
    unsigned c=in.c,q=in.q,m=in.degree,s=in.stride,n=in.n;
    uint32_t* h=scratch;
    uint32_t* poly=h+c*c;
    uint32_t* f=poly+(c+1)*s;
    uint32_t* factors=f+s;
    uint32_t* pivot=factors+c;
    uint32_t* k=pivot+2;
    uint32_t* power=k+q*q*s;
    uint32_t* next=power+c*q;
    uint32_t* memo=next+c*q;
    for(unsigned i=tid;i<in.words;i+=nt)scratch[i]=0;
    CORE_SYNC();
    for(unsigned ij=tid;ij<c*c;ij+=nt){unsigned i=ij/c,j=ij%c;
        bool positive=i/2==0||(signs&(UINT64_C(1)<<(i/2-1)));
        h[ij]=in.adjacency[(i^1)*n+j]?(positive?1:P-1):0;
    }
    CORE_SYNC();
    for(unsigned col=0;col+2<c;++col) {
        if(tid==0){unsigned r=col+1;while(r<c&&!h[r*c+col])++r;*pivot=r;}
        CORE_SYNC();
        unsigned r=*pivot;
        // All threads must copy the pivot before a zero-column skip lets
        // thread zero publish the next column's pivot.
        CORE_SYNC();
        if(r==c)continue; // identical branch across CTA
        for(unsigned j=tid;j<c;j+=nt){uint32_t t=h[r*c+j];h[r*c+j]=h[(col+1)*c+j];h[(col+1)*c+j]=t;}
        CORE_SYNC();
        for(unsigned i=tid;i<c;i+=nt){uint32_t t=h[i*c+r];h[i*c+r]=h[i*c+col+1];h[i*c+col+1]=t;}
        CORE_SYNC();
        if(tid==0)pivot[1]=F::inverse(h[(col+1)*c+col]);
        CORE_SYNC();
        for(unsigned i=col+2+tid;i<c;i+=nt)factors[i]=F::mul(h[i*c+col],pivot[1]);
        CORE_SYNC();
        // Commuting row eliminations followed by their joint inverse-column
        // update. The pivot row and every factor are stable during each pass.
        for(unsigned ij=tid;ij<c*c;ij+=nt){unsigned i=ij/c,j=ij%c;
            if(i>col+1&&j>=col)h[ij]=F::sub(h[ij],F::mul(factors[i],h[(col+1)*c+j]));}
        CORE_SYNC();
        for(unsigned i=tid;i<c;i+=nt){uint32_t value=h[i*c+col+1];
            for(unsigned j=col+2;j<c;++j)value=F::add(value,F::mul(factors[j],h[i*c+j]));
            h[i*c+col+1]=value;}
        CORE_SYNC();
    }
    if(tid==0)poly[0]=1;
    CORE_SYNC();
    for(unsigned size=1;size<=c;++size) {
        for(unsigned d=tid;d<=m&&d<=size;d+=nt) {
            uint32_t value=d<size?poly[(size-1)*s+d]:0;
            if(d)value=F::sub(value,F::mul(h[(size-1)*c+size-1],poly[(size-1)*s+d-1]));
            uint32_t product=1;
            for(unsigned dist=1;dist<size&&dist+1<=d;++dist){
                product=F::mul(product,h[(size-dist)*c+size-dist-1]);
                value=F::sub(value,F::mul(F::mul(product,h[(size-dist-1)*c+size-1]),poly[(size-dist-1)*s+d-dist-1]));}
            poly[size*s+d]=value;
        }
        CORE_SYNC();
    }
    if(tid==0){f[0]=1;for(unsigned d=1;d<=m;++d){uint32_t value=0;
        for(unsigned j=1;j<=d;++j)value=F::add(value,F::mul(2*d-j,F::mul(poly[c*s+j],f[d-j])));
        f[d]=F::neg(F::mul(value,in.inverse[2*d]));}}
    for(unsigned ij=tid;ij<q*q;ij+=nt)k[ij*s]=in.adjacency[(c+ij/q)*n+c+ij%q];
    for(unsigned vj=tid;vj<c*q;vj+=nt){unsigned v=vj/q,j=vj%q;
        bool positive=v/2==0||(signs&(UINT64_C(1)<<(v/2-1)));
        power[vj]=in.adjacency[(v^1)*n+c+j]?(positive?1:P-1):0;}
    CORE_SYNC();
    for(unsigned d=1;d<=m;++d) {
        for(unsigned ij=tid;ij<q*q;ij+=nt){unsigned i=ij/q,j=ij%q;uint32_t value=0;
            for(unsigned v=0;v<c;++v)if(in.adjacency[(c+i)*n+v])value=F::add(value,power[v*q+j]);
            k[ij*s+d]=value;}
        if(d<m){
            for(unsigned vj=tid;vj<c*q;vj+=nt){unsigned v=vj/q,j=vj%q;uint32_t value=0;
                for(unsigned u=0;u<c;++u)if(in.adjacency[(v^1)*n+u])value=F::add(value,power[u*q+j]);
                bool positive=v/2==0||(signs&(UINT64_C(1)<<(v/2-1)));
                next[vj]=positive?value:F::neg(value);}
            CORE_SYNC();
            uint32_t* temporary=power;power=next;next=temporary;
        }
        CORE_SYNC();
    }
    if(tid==0)memo[in.slot[0]*s]=1;
    CORE_SYNC();
    for(unsigned level=1;level<=5;++level) {
        for(unsigned idx=in.starts[level]*s+tid;idx<in.starts[level+1]*s;idx+=nt){
            unsigned slot=idx/s,d=idx%s,mask=in.masks[slot];
            unsigned first=0;while(!(mask&(1u<<first)))++first;
            unsigned rest=mask^(1u<<first);uint32_t value=0;
            for(unsigned j=first+1;j<q;++j)if(rest&(1u<<j)) {
                const uint32_t* child=memo+in.slot[rest^(1u<<j)]*s;
                const uint32_t* edge=k+(first*q+j)*s;
                for(unsigned a=0;a<=d;++a)value=F::add(value,F::mul(edge[a],child[d-a]));
            }
            memo[idx]=value;
        }
        CORE_SYNC();
    }
    unsigned ones=0;for(uint64_t b=signs;b;b&=b-1)++ones;
    bool negative=m&&((m-1-ones)&1);
    for(unsigned j=tid;j<in.queries;j+=nt){uint32_t value=0;
        for(unsigned d=0;d<=m;++d)value=F::add(value,F::mul(f[d],memo[in.answers[j]*s+m-d]));
        output[j]=negative?F::neg(value):value;}
    CORE_SYNC();
}
} // namespace core_gpu

namespace core_gpu {
#ifndef CORE_HOST_EMULATION
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<uint32_t P> __global__ void kernel(const Input* in,uint64_t begin,uint32_t* output) {
    extern __shared__ uint32_t scratch[];
    uint64_t index=begin+blockIdx.x;
    term<P>(*in,index^(index>>1),scratch,output+size_t(blockIdx.x)*in->queries,threadIdx.x,blockDim.x);
}
#endif

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
    checked(cudaMalloc(&device_in,sizeof(Input)));checked(cudaMalloc(&device_out,out.size()*sizeof(uint32_t)));
    checked(cudaMemcpy(device_in,&in,sizeof(Input),cudaMemcpyHostToDevice));
    checked(cudaFuncSetAttribute(kernel<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,in.words*sizeof(uint32_t)));
    int active=0;checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<P>,threads,in.words*sizeof(uint32_t)));
    cudaEvent_t start,stop;checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
    checked(cudaEventRecord(start));
    kernel<P><<<count,threads,in.words*sizeof(uint32_t)>>>(device_in,begin,device_out);
    checked(cudaGetLastError());checked(cudaEventRecord(stop));checked(cudaEventSynchronize(stop));
    float ms=0;checked(cudaEventElapsedTime(&ms,start,stop));elapsed=ms/1000.0;
    checked(cudaMemcpy(out.data(),device_out,out.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    std::printf("CORE_CUDA_LAUNCH prime=%u core=%u pool=%u queries=%u states=%u shared_bytes=%zu threads=%u active_ctas_per_sm=%d signs=%u kernel_s=%.6f\n",
        P,in.c,in.q,in.queries,in.states,in.words*sizeof(uint32_t),threads,active,count,elapsed);
    checked(cudaEventDestroy(start));checked(cudaEventDestroy(stop));checked(cudaFree(device_out));checked(cudaFree(device_in));
#endif
    return out;
}

void check_samples(const common_bench::Problem& p,uint32_t prime,uint64_t begin,unsigned count,
                   unsigned threads,bool compare_all,std::vector<uint32_t>& out,double& elapsed) {
    auto started=Clock::now();
    auto in=pack(p,prime);
    switch(prime){
        case 2147483647U:out=execute<2147483647U>(in,begin,count,threads,elapsed);break;
        case 2147483629U:out=execute<2147483629U>(in,begin,count,threads,elapsed);break;
        case 2147483587U:out=execute<2147483587U>(in,begin,count,threads,elapsed);break;
        case 2147483579U:out=execute<2147483579U>(in,begin,count,threads,elapsed);break;
        default:throw std::runtime_error("uncertified field");
    }
    std::printf("CORE_CUDA_EXECUTION prime=%u queries=%zu signs=%u solve_wall_s=%.6f kernel_s=%.6f\n",
        prime,p.masks.size(),count,common_bench::seconds(started,Clock::now()),elapsed);
    common_bench::Workspace w(p,Mod{prime});
    unsigned checks=compare_all?count:std::min(count,128u);
    for(unsigned j=0;j<checks;++j){unsigned i=compare_all?j:unsigned(uint64_t(j)*(count-1)/std::max(1u,checks-1));
        uint64_t index=begin+i;
        auto expected=common_bench::shared_term(p,w,index^(index>>1),Mod{prime});
        for(unsigned k=0;k<expected.size();++k)if(expected[k]!=out[size_t(i)*expected.size()+k])
            throw std::runtime_error("cooperative/CPU mismatch at sign "+std::to_string(index)+" child "+std::to_string(k));
    }
    std::printf("CORE_CUDA_PARITY prime=%u core=%u pool=%u queries=%zu compared_signs=%u exact=OK\n",prime,p.core,p.q,p.masks.size(),checks);
}

void self_test(unsigned threads) {
    std::mt19937_64 random(477);unsigned checks=0;
    for(uint32_t prime:{2147483647u,2147483629u,2147483587u,2147483579u}) {
        // Test the pseudo-Mersenne reduction independently, including limits.
        for(unsigned i=0;i<10000;++i){uint32_t a=random()%prime,b=random()%prime;
            if(i<4){a=prime-1-i;b=prime-1;}
#ifdef CORE_HOST_EMULATION
            auto got=prime==2147483647u?Field<2147483647u>::mul(a,b):prime==2147483629u?Field<2147483629u>::mul(a,b):
                prime==2147483587u?Field<2147483587u>::mul(a,b):Field<2147483579u>::mul(a,b);
            if(got!=Mod{prime}.mul(a,b))throw std::runtime_error("field product mismatch");
#endif
        }
        for(unsigned c:{0u,2u,6u,10u})for(unsigned q:{5u,7u})for(unsigned mode:{0u,1u}) {
            common_bench::Problem p{Matrix(c+q),c,q,{}};
            for(unsigned mask=0;mask<(1u<<q);++mask)if(unsigned(__builtin_popcount(mask))==q-3)p.masks.push_back(mask);
            for(unsigned i=0;i<c+q;++i)for(unsigned j=0;j<i;++j)
                p.adjacency.at(i,j)=p.adjacency.at(j,i)=mode?unsigned(random()%2):0;
            unsigned count=c?1u<<(c/2-1):1;double seconds=0;std::vector<uint32_t> out;
            check_samples(p,prime,0,count,threads,true,out,seconds);checks+=count*p.masks.size();
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
    uint32_t prime=2147483647;bool test=false,complete=false,sweep=false;
#ifdef CORE_HOST_EMULATION
    threads=4;count=8;
#endif
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--self-test")test=true;
        else if(a=="--sweep")sweep=true;
        else if(a=="--complete6x28"){complete=true;slack=2;}
        else if(a=="--groups"&&i+1<argc)path=argv[++i];
        else if(a=="--count"&&i+1<argc)count=unsigned(number(argv[++i]));
        else if(a=="--threads"&&i+1<argc)threads=unsigned(number(argv[++i]));
        else if(a=="--cap"&&i+1<argc)cap=unsigned(number(argv[++i]));
        else if(a=="--order"&&i+1<argc)order=unsigned(number(argv[++i]));
        else if(a=="--query-limit"&&i+1<argc)limit=unsigned(number(argv[++i]));
        else if(a=="--prime"&&i+1<argc)prime=uint32_t(number(argv[++i]));
        else throw std::runtime_error("usage: --self-test | --groups LOG [--count N --cap 7|9|11 --order N --query-limit N --threads N --prime P --complete6x28]");
    }
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
    if(test)core_gpu::self_test(threads);
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
            core_gpu::check_samples(p,prime,begin,steps,threads,false,out,elapsed);
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
    core_gpu::check_samples(p,prime,begin,count,threads,false,out,elapsed);
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
