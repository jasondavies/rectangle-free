#ifndef RECTANGLE_FREE_GPU_CUDA_UTILS_CUH
#define RECTANGLE_FREE_GPU_CUDA_UTILS_CUH

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

// Geometry-independent ownership primitives for the production CUDA drivers.
template <class T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { reserve(count); }
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void reserve(size_t count) {
        if (count <= capacity_) return;
        if (data_) CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
        capacity_ = 0;
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        capacity_ = count;
    }

    void reset() {
        if (data_) CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
        capacity_ = 0;
    }

    T* get() const { return data_; }
    size_t capacity() const { return capacity_; }
    explicit operator bool() const { return data_ != nullptr; }
    operator T*() const { return data_; }

  private:
    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(capacity_, other.capacity_);
    }

    T* data_ = nullptr;
    size_t capacity_ = 0;
};

class CudaEvent {
  public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    operator cudaEvent_t() const { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

class CudaStream {
  public:
    CudaStream() = default;
    explicit CudaStream(unsigned flags) { create(flags); }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            reset();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    void create(unsigned flags) {
        reset();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    void reset() {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    cudaStream_t get() const { return stream_; }
    explicit operator bool() const { return stream_ != nullptr; }
    operator cudaStream_t() const { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

template <class T>
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    ~PinnedBuffer() {
        if (data_) cudaFreeHost(data_);
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept { swap(other); }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void reserve(size_t count) {
        if (count <= capacity_) return;
        reset();
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_),
                                 count * sizeof(T),
                                 cudaHostAllocDefault));
        capacity_ = count;
    }
    void reset() {
        if (data_) cudaFreeHost(data_);
        data_ = nullptr;
        capacity_ = 0;
    }
    T* get() const { return data_; }
    size_t capacity() const { return capacity_; }
    explicit operator bool() const { return data_ != nullptr; }

  private:
    void swap(PinnedBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(capacity_, other.capacity_);
    }

    T* data_ = nullptr;
    size_t capacity_ = 0;
};

template <class T>
static T* upload_vector(const std::vector<T>& host) {
    if (host.empty()) return nullptr;
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

template <class T>
static DeviceBuffer<T> upload_buffer(const std::vector<T>& host) {
    DeviceBuffer<T> device;
    device.reserve(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(device.get(), host.data(),
                              host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    return device;
}

#endif
