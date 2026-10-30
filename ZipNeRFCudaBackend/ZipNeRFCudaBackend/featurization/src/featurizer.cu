#include "featurizer.h"
#include "random.cuh"
#include "zipnerf_config.h"

#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace zipnerf::featurization {

    static std::string rtc_kernel_dir;

    void set_rtc_dirs(const std::string& cache_dir, const std::string& kernel_dir) {
        runtime::set_cache_dir(cache_dir);
        rtc_kernel_dir = kernel_dir;
    }

    static std::string read_rtc_kernel_file(const std::string& filename) {
        std::ifstream file(rtc_kernel_dir + "/" + filename);
        if (!file) throw std::runtime_error("ZipNeRF: could not read rtc source: " + filename);
        std::stringstream stream;
        stream << file.rdbuf();
        return stream.str();
    }

    static void check_input(const torch::Tensor& tensor, const char* name, const torch::ScalarType dtype) {
        if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != dtype) {
            throw std::runtime_error(std::string("ZipNeRF: ") + name + " must be a contiguous cuda tensor of the expected type");
        }
    }

    // row-strided 2D tensors (e.g., column slices of a larger buffer)
    static void check_rows(const torch::Tensor& tensor, const char* name, const torch::ScalarType dtype, const int64_t n_rows, const int64_t n_columns) {
        if (!tensor.is_cuda() || tensor.scalar_type() != dtype || tensor.dim() != 2 || tensor.size(0) != n_rows || tensor.size(1) < n_columns || tensor.stride(1) != 1) {
            throw std::runtime_error(std::string("ZipNeRF: ") + name + " has an invalid layout");
        }
    }

    template <typename T>
    static T* optional_ptr(const torch::Tensor& tensor) {
        return tensor.numel() > 0 ? reinterpret_cast<T*>(tensor.data_ptr()) : nullptr;
    }

    Featurizer::Featurizer(const int level, const std::vector<int64_t>& level_sizes, const std::vector<int64_t>& level_entries, const int n_features, const int mlp_width) :
        m_level(level), m_n_levels(static_cast<int>(level_sizes.size())), m_n_features(n_features), m_mlp_width(mlp_width), m_level_sizes(level_sizes), m_level_entries(level_entries)
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&m_n_sms, cudaDevAttrMultiProcessorCount, device);
        m_level_offsets.push_back(0);
        for (const int64_t entries : level_entries) m_level_offsets.push_back(m_level_offsets.back() + entries);
        if (m_level_offsets.back() * n_features > static_cast<int64_t>(UINT32_MAX)) throw std::runtime_error("ZipNeRF: grid table too large for 32-bit indexing");
    }

    runtime::RtcKernel& Featurizer::kernel(std::unique_ptr<runtime::RtcKernel>& kernel, const std::string& name) {
        if (kernel != nullptr) return *kernel;
        std::stringstream preamble;
        preamble << "typedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\n";
        preamble << "#define JIT_N_LEVELS " << m_n_levels << "u\n";
        preamble << "#define JIT_N_FEATURES " << m_n_features << "u\n";
        preamble << "#define JIT_MLP_WIDTH " << m_mlp_width << "u\n";
        preamble << "#define JIT_PRECONDITION_SCALING " << std::showpoint << std::setprecision(17) << zipnerf::config::precondition_scaling << "f\n";
        preamble << "#define JIT_BLOCK_SIZE " << config::block_size << "u\n";
        preamble << "#define JIT_PATTERN_STREAM " << random::pattern + m_level << "ull\n";
        // the per-level grid layout is emitted as switches that fold to constants in the unrolled level loops
        const auto emit_switch = [&](const char* function, const char* type, const auto& values) {
            preamble << "__device__ inline " << type << " " << function << "(const uint32_t level) {\n    switch (level) {\n";
            for (int level = 0; level < m_n_levels; level++) preamble << "        case " << level << ": return " << values(level) << ";\n";
            preamble << "        default: __trap();\n    }\n    return 0;\n}\n";
        };
        preamble << "namespace zipnerf { namespace config {\n";
        emit_switch("level_size", "uint32_t", [&](int level) { return std::to_string(m_level_sizes[level]) + "u"; });
        emit_switch("level_entries", "uint32_t", [&](int level) { return std::to_string(m_level_entries[level]) + "u"; });
        emit_switch("level_offset", "uint32_t", [&](int level) { return std::to_string(m_level_offsets[level]) + "u"; });
        emit_switch("level_hashed", "bool", [&](int level) { return std::string(m_level_sizes[level] * m_level_sizes[level] * m_level_sizes[level] > m_level_entries[level] ? "true" : "false"); });
        preamble << "} }\n";
        // the sources are inlined (instead of included) so that the cache key changes whenever any of the files change
        const std::string source = preamble.str() + read_rtc_kernel_file("../../common/include/random.cuh") + "\n" + read_rtc_kernel_file("featurize_common.cuh") + "\n" + read_rtc_kernel_file(name + ".cu");
        kernel = std::make_unique<runtime::RtcKernel>(name, source);
        return *kernel;
    }

    static uint32_t n_blocks(const int64_t n) { return static_cast<uint32_t>((n + config::block_size - 1) / config::block_size); }

    void Featurizer::forward(
        const torch::Tensor& origins,
        const torch::Tensor& directions,
        const torch::Tensor& radii,
        const torch::Tensor& tdist,
        const int64_t pattern_seed,
        const bool interleave,
        const torch::Tensor& table,
        const torch::Tensor& scale_magnitudes,
        const torch::Tensor& w0,
        const torch::Tensor& b0,
        const torch::Tensor& w1,
        const torch::Tensor& b1,
        torch::Tensor& raw_density,
        torch::Tensor& features_float,
        torch::Tensor& features_half,
        const torch::Tensor& encoded_directions,
        torch::Tensor& nerf_inputs)
    {
        check_input(origins, "origins", torch::kFloat);
        check_input(directions, "directions", torch::kFloat);
        check_input(radii, "radii", torch::kFloat);
        check_input(tdist, "tdist", torch::kFloat);
        check_input(table, "table", torch::kFloat);
        check_input(scale_magnitudes, "scale_magnitudes", torch::kFloat);
        const int64_t n_intervals = tdist.size(1) - 1;
        const int64_t n_samples = tdist.size(0) * n_intervals;
        if (raw_density.numel() > 0) {
            check_input(raw_density, "raw_density", torch::kFloat);
            if (raw_density.numel() != n_samples) throw std::runtime_error("ZipNeRF: invalid raw_density");
            check_input(w0, "w0", torch::kFloat);
            if (m_mlp_width == 0 || w0.numel() != m_mlp_width * n_output_features()) throw std::runtime_error("ZipNeRF: invalid proposal mlp weights");
            check_input(b0, "b0", torch::kFloat);
            check_input(w1, "w1", torch::kFloat);
            check_input(b1, "b1", torch::kFloat);
        }
        if (features_float.numel() > 0) check_input(features_float, "features_float", torch::kFloat);
        if (features_half.numel() > 0) {
            check_input(features_half, "features_half", torch::kHalf);
            if (features_half.size(0) != n_samples || features_half.size(1) < n_output_features()) throw std::runtime_error("ZipNeRF: invalid features_half");
        }
        if (nerf_inputs.numel() > 0) {
            check_input(nerf_inputs, "nerf_inputs", torch::kHalf);
            check_input(encoded_directions, "encoded_directions", torch::kHalf);
            if (nerf_inputs.size(0) != n_samples || nerf_inputs.size(1) < encoded_directions.size(1)) throw std::runtime_error("ZipNeRF: invalid nerf_inputs");
        }
        if (n_samples == 0) return;
        kernel(m_forward_kernel, "featurize_forward").launch(
            n_blocks(n_samples), config::block_size, 0, c10::cuda::getCurrentCUDAStream(),
            reinterpret_cast<const float3*>(origins.data_ptr<float>()),
            reinterpret_cast<const float3*>(directions.data_ptr<float>()),
            radii.data_ptr<float>(),
            tdist.data_ptr<float>(),
            static_cast<long long>(pattern_seed),
            interleave,
            table.data_ptr<float>(),
            scale_magnitudes.data_ptr<float>(),
            optional_ptr<const float>(raw_density.numel() > 0 ? w0 : raw_density),
            optional_ptr<const float>(raw_density.numel() > 0 ? b0 : raw_density),
            optional_ptr<const float>(raw_density.numel() > 0 ? w1 : raw_density),
            optional_ptr<const float>(raw_density.numel() > 0 ? b1 : raw_density),
            optional_ptr<float>(raw_density),
            optional_ptr<float>(features_float),
            optional_ptr<uint16_t>(features_half),
            features_half.numel() > 0 ? static_cast<int>(features_half.size(1)) : 0,
            optional_ptr<const uint16_t>(nerf_inputs.numel() > 0 ? encoded_directions : nerf_inputs),
            optional_ptr<uint16_t>(nerf_inputs),
            nerf_inputs.numel() > 0 ? static_cast<int>(nerf_inputs.size(1)) : 0,
            nerf_inputs.numel() > 0 ? static_cast<int>(encoded_directions.size(1)) : 0,
            static_cast<uint32_t>(n_samples),
            static_cast<uint32_t>(n_intervals));
    }

    void Featurizer::backward(
        const torch::Tensor& origins,
        const torch::Tensor& directions,
        const torch::Tensor& radii,
        const torch::Tensor& tdist,
        const int64_t pattern_seed,
        const torch::Tensor& grad_features,
        torch::Tensor& grad_table)
    {
        check_input(grad_table, "grad_table", torch::kFloat);
        const int64_t n_intervals = tdist.size(1) - 1;
        const int64_t n_samples = tdist.size(0) * n_intervals;
        check_rows(grad_features, "grad_features", torch::kHalf, n_samples, n_output_features());
        if (n_samples == 0) return;
        kernel(m_backward_kernel, "featurize_backward").launch(
            n_blocks(n_samples), config::block_size, 0, c10::cuda::getCurrentCUDAStream(),
            reinterpret_cast<const float3*>(origins.data_ptr<float>()),
            reinterpret_cast<const float3*>(directions.data_ptr<float>()),
            radii.data_ptr<float>(),
            tdist.data_ptr<float>(),
            static_cast<long long>(pattern_seed),
            reinterpret_cast<const uint16_t*>(grad_features.data_ptr()),
            static_cast<int>(grad_features.stride(0)),
            static_cast<float>(1.0 / zipnerf::config::nerf_mlp_grad_scale),
            grad_table.data_ptr<float>(),
            static_cast<uint32_t>(n_samples),
            static_cast<uint32_t>(n_intervals));
    }

    void Featurizer::backward_mlp(
        const torch::Tensor& origins,
        const torch::Tensor& directions,
        const torch::Tensor& radii,
        const torch::Tensor& tdist,
        const int64_t pattern_seed,
        const torch::Tensor& features,
        const torch::Tensor& grad_raw_density,
        const torch::Tensor& w0,
        const torch::Tensor& b0,
        const torch::Tensor& w1,
        torch::Tensor& grad_table,
        torch::Tensor& grad_w0,
        torch::Tensor& grad_b0,
        torch::Tensor& grad_w1,
        torch::Tensor& grad_b1)
    {
        const int64_t n_intervals = tdist.size(1) - 1;
        const int64_t n_samples = tdist.size(0) * n_intervals;
        check_input(features, "features", torch::kFloat);
        check_input(grad_raw_density, "grad_raw_density", torch::kFloat);
        if (features.numel() != n_samples * n_output_features() || grad_raw_density.numel() != n_samples) throw std::runtime_error("ZipNeRF: invalid backward_mlp inputs");
        check_input(w0, "w0", torch::kFloat);
        check_input(b0, "b0", torch::kFloat);
        check_input(w1, "w1", torch::kFloat);
        check_input(grad_table, "grad_table", torch::kFloat);
        check_input(grad_w0, "grad_w0", torch::kFloat);
        check_input(grad_b0, "grad_b0", torch::kFloat);
        check_input(grad_w1, "grad_w1", torch::kFloat);
        check_input(grad_b1, "grad_b1", torch::kFloat);
        if (m_mlp_width == 0 || w0.numel() != m_mlp_width * n_output_features() || grad_w0.numel() != w0.numel()) throw std::runtime_error("ZipNeRF: invalid proposal mlp weights");
        if (n_samples == 0) return;
        const uint32_t blocks = std::min(n_blocks(n_samples), static_cast<uint32_t>(m_n_sms * config::backward_mlp_blocks_per_sm));
        // pre-activations, features, and raw density gradients of one block step
        const uint32_t shmem_bytes = config::block_size * (m_mlp_width + n_output_features() + 1) * sizeof(float);
        kernel(m_backward_mlp_kernel, "featurize_backward_mlp").launch(
            blocks, config::block_size, shmem_bytes, c10::cuda::getCurrentCUDAStream(),
            reinterpret_cast<const float3*>(origins.data_ptr<float>()),
            reinterpret_cast<const float3*>(directions.data_ptr<float>()),
            radii.data_ptr<float>(),
            tdist.data_ptr<float>(),
            static_cast<long long>(pattern_seed),
            features.data_ptr<float>(),
            grad_raw_density.data_ptr<float>(),
            w0.data_ptr<float>(),
            b0.data_ptr<float>(),
            w1.data_ptr<float>(),
            grad_table.data_ptr<float>(),
            grad_w0.data_ptr<float>(),
            grad_b0.data_ptr<float>(),
            grad_w1.data_ptr<float>(),
            grad_b1.data_ptr<float>(),
            static_cast<uint32_t>(n_samples),
            static_cast<uint32_t>(n_intervals));
    }

}
