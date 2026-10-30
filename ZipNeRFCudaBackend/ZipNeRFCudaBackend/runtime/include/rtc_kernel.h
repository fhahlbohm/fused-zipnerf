#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace zipnerf::runtime {

    // directory for compiled kernels (cubin or ptx), keyed by a hash of the source, options, and architecture
    void set_cache_dir(const std::string& cache_dir);

    // a kernel compiled at runtime with NVRTC; the source must declare it as extern "C" __global__
    class RtcKernel {
    public:
        RtcKernel(const std::string& name, const std::string& source);
        ~RtcKernel();
        RtcKernel(const RtcKernel&) = delete;
        RtcKernel& operator=(const RtcKernel&) = delete;

        template <typename... Args>
        void launch(const uint32_t n_blocks, const uint32_t block_size, const uint32_t shmem_bytes, const cudaStream_t stream, Args... args) {
            void* argument_pointers[] = {static_cast<void*>(&args)...};
            launch_impl(n_blocks, block_size, shmem_bytes, stream, argument_pointers);
        }

    private:
        void launch_impl(uint32_t n_blocks, uint32_t block_size, uint32_t shmem_bytes, cudaStream_t stream, void** arguments);

        CUmodule m_module = nullptr;
        CUfunction m_function = nullptr;
        uint32_t m_max_shmem_bytes = 0;  // dynamic shared memory limit set on the function (static + dynamic may exceed 48 KiB)
    };

}
