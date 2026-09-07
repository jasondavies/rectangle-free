#pragma once
#include "hafnian_common_core.cuh"
namespace core_gpu {
// Persistent, bounded sign evaluation and exact modular reduction.
#ifndef CORE_HOST_EMULATION
template<class T> struct CoreDeviceArray {
    T* data=nullptr;size_t capacity=0;
    CoreDeviceArray()=default;
    CoreDeviceArray(const CoreDeviceArray&)=delete;
    CoreDeviceArray& operator=(const CoreDeviceArray&)=delete;
    ~CoreDeviceArray(){if(data)cudaFree(data);}
    void reserve(size_t n){
        if(n<=capacity)return;
        T* next=nullptr;checked(cudaMalloc(&next,n*sizeof(T)));
        if(data){auto status=cudaFree(data);if(status!=cudaSuccess){cudaFree(next);checked(status);}}
        data=next;capacity=n;
    }
};

// A two-stage modular reduction. First spread each child's strided sign
// stream across multiple CTAs; then one writer per child accumulates its
// chunk total. Every addition is reduced, including the cross-chunk sum.
template<uint32_t P> __global__ void reduce_signs(const uint32_t* values,
        uint32_t* partial,unsigned count,unsigned queries) {
    using F=Field<P>;
    __shared__ uint32_t warps[8];
    uint32_t sum=0;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        sum=F::add(sum,values[size_t(i)*queries+blockIdx.y]);
    for(unsigned offset=16;offset;offset>>=1)
        sum=F::add(sum,__shfl_down_sync(0xffffffff,sum,offset));
    if(!(threadIdx.x&31))warps[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32){
        sum=threadIdx.x<8?warps[threadIdx.x]:0;
        for(unsigned offset=16;offset;offset>>=1)
            sum=F::add(sum,__shfl_down_sync(0xffffffff,sum,offset));
        if(!threadIdx.x)partial[size_t(blockIdx.x)*queries+blockIdx.y]=sum;
    }
}
template<uint32_t P> __global__ void accumulate_parts(const uint32_t* partial,
        uint32_t* totals,unsigned parts,unsigned queries) {
    unsigned j=threadIdx.x;
    if(j<queries){uint32_t sum=totals[j];
        for(unsigned i=0;i<parts;++i)sum=Field<P>::add(sum,partial[size_t(i)*queries+j]);
        totals[j]=sum;
    }
}
#endif

// One dispatcher thread owns a runner. CPU producers may prepare independent
// Input objects concurrently, but must not submit through the same runner.
template<uint32_t P> class ReducedRunner {
#ifndef CORE_HOST_EMULATION
    CoreDeviceArray<Input> input;
    CoreDeviceArray<uint32_t> values,partial,totals;
    cudaStream_t stream=nullptr;
    cudaEvent_t start=nullptr,stop=nullptr;
#endif
public:
    ReducedRunner(){
#ifndef CORE_HOST_EMULATION
        try{checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
        }catch(...){release();throw;}
#endif
    }
    ReducedRunner(const ReducedRunner&)=delete;
    ReducedRunner& operator=(const ReducedRunner&)=delete;
    ~ReducedRunner(){
#ifndef CORE_HOST_EMULATION
        release();
#endif
    }
#ifndef CORE_HOST_EMULATION
    void release(){
        if(stream)cudaStreamSynchronize(stream);
        if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);
        if(stream)cudaStreamDestroy(stream);
        start=stop=nullptr;stream=nullptr;
    }
#endif
    // Input already prepared once for this active-child set and field.
    // count may exceed one chunk; allocations persist across groups.
    std::vector<uint32_t> run(const Input& in,uint64_t begin,uint64_t count,
            unsigned chunk,unsigned threads,double& device_seconds) {
        uint64_t domain=in.c?UINT64_C(1)<<(in.c/2-1):1;
        if(!count||!chunk||chunk>(1u<<20)||begin>domain||count>domain-begin||
           !in.queries||in.queries>256)throw std::runtime_error("invalid reduced sign range");
        std::vector<uint32_t> out(in.queries);
#ifdef CORE_HOST_EMULATION
        auto started=std::chrono::steady_clock::now();
        std::vector<uint32_t> scratch(in.words),raw(in.queries);
        for(uint64_t index=begin;index<begin+count;++index){
            #pragma omp parallel num_threads(threads)
            term<P>(in,index^(index>>1),scratch.data(),raw.data(),
                unsigned(omp_get_thread_num()),unsigned(omp_get_num_threads()));
            for(unsigned j=0;j<in.queries;++j)out[j]=Field<P>::add(out[j],raw[j]);
        }
        device_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
#else
        if(CORE_PROFILE)throw std::runtime_error("reduced runner requires uninstrumented kernel");
        if(threads!=64&&threads!=128&&threads!=256)throw std::runtime_error("invalid CUDA block size");
        unsigned capacity=unsigned(std::min(uint64_t(chunk),count));
        input.reserve(1);values.reserve(size_t(capacity)*in.queries);
        partial.reserve(size_t(32)*in.queries);totals.reserve(in.queries);
        checked(cudaMemcpyAsync(input.data,&in,sizeof(Input),cudaMemcpyHostToDevice,stream));
        checked(cudaMemsetAsync(totals.data,0,in.queries*sizeof(uint32_t),stream));
        checked(cudaFuncSetAttribute(kernel<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,in.words*sizeof(uint32_t)));
        checked(cudaEventRecord(start,stream));
        for(uint64_t offset=0;offset<count;offset+=chunk){
            unsigned steps=unsigned(std::min(uint64_t(chunk),count-offset));
            kernel<P><<<steps,threads,in.words*sizeof(uint32_t),stream>>>(input.data,begin+offset,values.data,nullptr);
            checked(cudaGetLastError());
            unsigned parts=std::min(32u,(steps+255)/256);
            reduce_signs<P><<<dim3(parts,in.queries),256,0,stream>>>(values.data,partial.data,steps,in.queries);
            checked(cudaGetLastError());
            accumulate_parts<P><<<1,256,0,stream>>>(partial.data,totals.data,parts,in.queries);
            checked(cudaGetLastError());
        }
        checked(cudaEventRecord(stop,stream));checked(cudaEventSynchronize(stop));
        float ms=0;checked(cudaEventElapsedTime(&ms,start,stop));device_seconds=ms/1000.0;
        checked(cudaMemcpyAsync(out.data(),totals.data,in.queries*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
#endif
        return out;
    }
};

} // namespace core_gpu
