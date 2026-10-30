#pragma once

#include "rtc_kernel.h"

#include <torch/extension.h>

#include <memory>
#include <string>
#include <vector>

namespace zipnerf::featurization {

    namespace config {
        inline constexpr int block_size = 128;
        inline constexpr int backward_mlp_blocks_per_sm = 8;
    }

    void set_rtc_dirs(const std::string& cache_dir, const std::string& kernel_dir);

    // multisampled hash grid featurization of one sampling level with an optional fused fp32 density mlp (in -> mlp_width
    // -> 1, none for mlp_width 0), compiled at runtime with the grid layout as constants
    class Featurizer {
    public:
        // level selects the random stream of the multisampling patterns
        Featurizer(int level, const std::vector<int64_t>& level_sizes, const std::vector<int64_t>& level_entries, int n_features, int mlp_width);

        int n_output_features() const { return m_n_levels * (m_n_features + 1); }

        // featurizes all intervals of tdist (random patterns for pattern_seed >= 0) and writes the given outputs (empty
        // tensors are skipped): raw_density (requires the mlp weights), features_float, features_half (zero-padded), and
        // the encoded directions and bias lane in the last columns of nerf_inputs; interleave maps one interval of 32 rays to a warp
        void forward(
            const torch::Tensor& origins,
            const torch::Tensor& directions,
            const torch::Tensor& radii,
            const torch::Tensor& tdist,
            int64_t pattern_seed,
            bool interleave,
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
            torch::Tensor& nerf_inputs);

        // accumulates the grid gradient from fp16 feature gradients (scaled by the NeRF MLP gradient scale)
        void backward(
            const torch::Tensor& origins,
            const torch::Tensor& directions,
            const torch::Tensor& radii,
            const torch::Tensor& tdist,
            int64_t pattern_seed,
            const torch::Tensor& grad_features,
            torch::Tensor& grad_table);

        // accumulates the grid and mlp parameter gradients from the stored fp32 features and the raw density gradient
        void backward_mlp(
            const torch::Tensor& origins,
            const torch::Tensor& directions,
            const torch::Tensor& radii,
            const torch::Tensor& tdist,
            int64_t pattern_seed,
            const torch::Tensor& features,
            const torch::Tensor& grad_raw_density,
            const torch::Tensor& w0,
            const torch::Tensor& b0,
            const torch::Tensor& w1,
            torch::Tensor& grad_table,
            torch::Tensor& grad_w0,
            torch::Tensor& grad_b0,
            torch::Tensor& grad_w1,
            torch::Tensor& grad_b1);

    private:
        runtime::RtcKernel& kernel(std::unique_ptr<runtime::RtcKernel>& kernel, const std::string& name);

        int m_level;
        int m_n_levels;
        int m_n_features;
        int m_mlp_width;
        int m_n_sms;
        std::vector<int64_t> m_level_sizes, m_level_entries, m_level_offsets;
        std::unique_ptr<runtime::RtcKernel> m_forward_kernel, m_backward_kernel, m_backward_mlp_kernel;
    };

}
