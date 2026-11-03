#pragma once

#include <torch/extension.h>

#include <vector>

namespace zipnerf::featurization {

    // cell centers of a contracted-space density grid, mapped back to world space; reference supplies the device
    torch::Tensor proposal_grid_points_wrapper(const torch::Tensor& reference, int resolution, int64_t start, int count);

    // deterministic six-point proposal sampling from an FP32 density mip pyramid instead of the proposal network
    void sample_proposal_grid_wrapper(
        const std::vector<torch::Tensor>& levels,
        const torch::Tensor& origins,
        const torch::Tensor& directions,
        const torch::Tensor& radii,
        const torch::Tensor& tdist,
        torch::Tensor& raw_density);

}
