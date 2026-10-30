#pragma once

#include <torch/extension.h>

#include <string>
#include <tuple>
#include <vector>

namespace zipnerf::mlp {

    // fp16 weights of the GEMM chain, packed into one buffer
    enum Weight { w_density, b_density, w_combined, w_skip, w_1, b_1, w_2, b_2, w_rgb, n_weights };

    // tensors of the fp32 NeRF MLP parameter buffer in layout order
    enum Parameter {
        density_0_weight, density_0_bias, density_1_weight, density_1_bias, bottleneck_weight, bottleneck_bias, view_0_weight,
        view_0_bias, view_1_weight, view_1_bias, view_2_weight, view_2_bias, rgb_weight, rgb_bias, n_parameters
    };

    // layer widths and element offsets of the packed weights, the fp32 weight gradients (same layout, biases unused), the
    // fp16 bias gradients, and the parameter buffer tensors; the last entries are the sizes
    struct Layout {
        int64_t n_features, density_width, bottleneck_width, view_width, n_direction_features;
        int64_t n_inputs, n_view_inputs, n_rgb_inputs;
        int64_t weights[n_weights + 1];
        int64_t bias_grads[n_weights + 1];
        int64_t params[n_parameters + 1];
    };

    // name, shape, and fan-in (zero for biases) of each tensor of the fp32 parameter buffer in layout order
    std::vector<std::tuple<std::string, std::vector<int64_t>, int64_t>> nerf_mlp_parameters(
        int64_t n_features, int64_t density_width, int64_t bottleneck_width, int64_t view_width, int64_t direction_degree);

    // NeRF MLP of the final level as a chain of fp16 cuBLASLt GEMMs with fused epilogues (row-major activations):
    //   h = relu(features @ w_density^T + b_density)                     written into the first columns of inputs
    //   v0 = relu(inputs @ w_combined^T)                                 inputs = [h | encoded direction | 1]
    //   outputs = inputs @ w_skip^T                                      8 columns: raw density, rgb logit skip terms
    //   v1 = relu(v0 @ w_1^T + b_1), v2 = relu(v1 @ w_2^T + b_2)
    //   outputs += v2 @ w_rgb^T                                          completes the rgb logits (columns 1-3)
    // the bottleneck only feeds linear maps (the first view layer and, via the skip connection, the rgb layer), so it is
    // folded into w_combined and w_skip; the biases of the layers reading inputs are their last column (bias lane)
    class NerfMlp {
    public:
        // n_features is the unpadded feature count, the widths must be multiples of 8
        NerfMlp(int64_t n_features, int64_t density_width, int64_t bottleneck_width, int64_t view_width, int64_t direction_degree);

        int64_t n_features() const { return m_layout.n_features; }
        int64_t n_inputs() const { return m_layout.n_inputs; }

        // folds the fp32 parameters into the packed fp16 weights
        torch::Tensor pack(const torch::Tensor& params) const;

        // returns the fp16 outputs and the context of the backward (relu masks are only stored when training)
        std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(
            const torch::Tensor& features,
            torch::Tensor& inputs,
            const torch::Tensor& packed,
            bool training) const;

        // backward from the (scaled) fp16 output gradient: accumulates the (equally scaled) gradients of the fp32
        // parameters into grad_params and returns the fp16 feature gradient
        torch::Tensor backward(
            const torch::Tensor& grad_outputs,
            const torch::Tensor& features,
            const torch::Tensor& packed,
            const std::vector<torch::Tensor>& context,
            const torch::Tensor& params,
            torch::Tensor& grad_params) const;

    private:
        Layout m_layout;
    };

}
