#include "rtc_kernel.h"

#include <nvrtc.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace zipnerf::runtime {

    static std::string cache_dir;

    void set_cache_dir(const std::string& dir) {
        cache_dir = dir;
    }

    static void check_driver(const CUresult result, const char* call) {
        if (result != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(result, &message);
            throw std::runtime_error(std::string("ZipNeRF: ") + call + " failed: " + (message != nullptr ? message : "unknown error"));
        }
    }

    static void check_nvrtc(const nvrtcResult result, const char* call) {
        if (result != NVRTC_SUCCESS) throw std::runtime_error(std::string("ZipNeRF: ") + call + " failed: " + nvrtcGetErrorString(result));
    }

    // compiles to a cubin for the current device, or to ptx for the newest supported virtual architecture if nvrtc is
    // older than the device (the driver then compiles the ptx); binaries are cached on disk and written atomically
    static std::string compile(const std::string& name, const std::string& source, const bool read_cache) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        const int device_arch = properties.major * 10 + properties.minor;
        int n_archs;
        check_nvrtc(nvrtcGetNumSupportedArchs(&n_archs), "nvrtcGetNumSupportedArchs");
        std::vector<int> archs(n_archs);
        check_nvrtc(nvrtcGetSupportedArchs(archs.data()), "nvrtcGetSupportedArchs");
        const bool is_cubin = std::find(archs.begin(), archs.end(), device_arch) != archs.end();
        const int arch = is_cubin ? device_arch : *std::max_element(archs.begin(), archs.end());
        const std::string arch_option = (is_cubin ? "--gpu-architecture=sm_" : "--gpu-architecture=compute_") + std::to_string(arch);
        const char* options[] = {arch_option.c_str(), "-std=c++17", "--use_fast_math"};
        int nvrtc_major, nvrtc_minor;
        check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");
        const std::string options_key = arch_option + " -std=c++17 --use_fast_math nvrtc " + std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor);

        const size_t hash = std::hash<std::string>{}(source + options_key);
        std::stringstream cache_name;
        cache_name << name << "." << std::hex << hash << (is_cubin ? ".cubin" : ".ptx");
        const std::filesystem::path cache_path = std::filesystem::path(cache_dir) / cache_name.str();
        if (read_cache && !cache_dir.empty() && std::filesystem::exists(cache_path)) {
            std::ifstream file(cache_path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }

        nvrtcProgram program;
        check_nvrtc(nvrtcCreateProgram(&program, source.c_str(), (name + ".cu").c_str(), 0, nullptr, nullptr), "nvrtcCreateProgram");
        const nvrtcResult result = nvrtcCompileProgram(program, 3, options);
        if (result != NVRTC_SUCCESS) {
            size_t log_size;
            nvrtcGetProgramLogSize(program, &log_size);
            std::string log(log_size, '\0');
            nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program);
            throw std::runtime_error("ZipNeRF: compiling " + name + " failed:\n" + log);
        }
        size_t size;
        std::string binary;
        if (is_cubin) {
            check_nvrtc(nvrtcGetCUBINSize(program, &size), "nvrtcGetCUBINSize");
            binary.resize(size);
            check_nvrtc(nvrtcGetCUBIN(program, binary.data()), "nvrtcGetCUBIN");
        }
        else {
            check_nvrtc(nvrtcGetPTXSize(program, &size), "nvrtcGetPTXSize");
            binary.resize(size);
            check_nvrtc(nvrtcGetPTX(program, binary.data()), "nvrtcGetPTX");
        }
        nvrtcDestroyProgram(&program);
        if (!cache_dir.empty()) {
            // concurrent processes may compile the same kernel, so each writes a temporary file and renames it
            std::error_code error;
            std::filesystem::create_directories(cache_dir, error);
            if (!error) {
                const std::filesystem::path temp_path = cache_path.string() + ".tmp" + std::to_string(std::random_device{}());
                {
                    std::ofstream file(temp_path, std::ios::binary);
                    file.write(binary.data(), static_cast<std::streamsize>(binary.size()));
                }
                std::filesystem::rename(temp_path, cache_path, error);
                if (error) std::filesystem::remove(temp_path, error);
            }
        }
        return binary;
    }

    RtcKernel::RtcKernel(const std::string& name, const std::string& source) {
        // the runtime api initializes the primary context that the driver api calls below use
        cudaFree(nullptr);
        if (cuModuleLoadData(&m_module, compile(name, source, true).data()) != CUDA_SUCCESS) {
            // an invalid cache entry (e.g., from a process killed while writing) is replaced by a fresh compilation
            check_driver(cuModuleLoadData(&m_module, compile(name, source, false).data()), "cuModuleLoadData");
        }
        check_driver(cuModuleGetFunction(&m_function, m_module, name.c_str()), "cuModuleGetFunction");
    }

    RtcKernel::~RtcKernel() {
        if (m_module != nullptr) cuModuleUnload(m_module);
    }

    void RtcKernel::launch_impl(const uint32_t n_blocks, const uint32_t block_size, const uint32_t shmem_bytes, const cudaStream_t stream, void** arguments) {
        if (n_blocks == 0) return;
        if (shmem_bytes > m_max_shmem_bytes) {
            check_driver(cuFuncSetAttribute(m_function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(shmem_bytes)), "cuFuncSetAttribute");
            m_max_shmem_bytes = shmem_bytes;
        }
        check_driver(cuLaunchKernel(m_function, n_blocks, 1, 1, block_size, 1, 1, shmem_bytes, stream, arguments, nullptr), "cuLaunchKernel");
    }

}
