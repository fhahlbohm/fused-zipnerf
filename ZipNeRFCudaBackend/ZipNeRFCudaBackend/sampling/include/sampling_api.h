#pragma once

#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace zipnerf::sampling {

    // training: rays, colors, encoded directions, and first-level intervals of random pixels of all training views
    std::vector<torch::Tensor> generate_rays_wrapper(
        int64_t seed,
        int n_rays,
        const torch::Tensor& view_offsets,
        const torch::Tensor& view_cameras,
        const torch::Tensor& camera_offsets,
        const torch::Tensor& local_directions,
        const torch::Tensor& c2w,
        const torch::Tensor& camera_radii,
        const torch::Tensor& pool_rgb,
        int direction_degree,
        int n_samples,
        const std::vector<double>& ray_warp);

    // inference: rays, encoded directions, and first-level intervals (centered samples) of the given pixels of a view
    std::vector<torch::Tensor> view_rays_wrapper(
        const torch::Tensor& pixel_indices,
        const torch::Tensor& local_directions,
        const torch::Tensor& c2w,
        double radius,
        int direction_degree,
        int n_samples,
        const std::vector<double>& ray_warp);

    // encoded directions and first-level intervals of given rays (random jitter for seed >= 0, centered samples otherwise)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> initialize_rays_wrapper(
        int64_t seed,
        const torch::Tensor& directions,
        int direction_degree,
        int n_samples,
        const std::vector<double>& ray_warp);

    // the uniform random numbers of a stream as used by the training kernels (for the reference implementation)
    torch::Tensor sample_uniforms_wrapper(int64_t seed, int64_t stream, int64_t n);

    // weights of a proposal level and the next level's intervals (random jitter for seed >= 0, centered samples otherwise)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> composite_and_resample_wrapper(
        const torch::Tensor& raw_density,
        const torch::Tensor& sdist,
        const torch::Tensor& tdist,
        const torch::Tensor& directions,
        double anneal,
        int n_samples,
        int64_t seed,
        int next_level,
        const std::vector<double>& ray_warp);

    // inference: color, opacity, and expected or median depth from the NeRF MLP outputs, written to the rays' pixels
    // (empty buffers are skipped)
    void composite_final_wrapper(
        const torch::Tensor& nerf_outputs,
        const torch::Tensor& tdist,
        const torch::Tensor& directions,
        const torch::Tensor& background,
        const torch::Tensor& pixel_indices,
        torch::Tensor& rgb,
        torch::Tensor& alpha,
        torch::Tensor& depth,
        const bool use_median_depth);

    // training losses and their gradients w.r.t. the raw outputs of all levels: returns the NeRF MLP output gradient
    // (scaled by the NeRF MLP gradient scale), the proposal raw density gradients, and the loss sums if requested
    std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> train_final_wrapper(
        const torch::Tensor& nerf_outputs,
        const torch::Tensor& sdist,
        const torch::Tensor& tdist,
        const std::vector<torch::Tensor>& proposal_raw_densities,
        const std::vector<torch::Tensor>& proposal_sdists,
        const std::vector<torch::Tensor>& proposal_tdists,
        const std::vector<torch::Tensor>& proposal_weights,
        const torch::Tensor& directions,
        int64_t seed,
        const torch::Tensor& rgb_target,
        double charbonnier_padding,
        const std::vector<double>& interlevel_weights,
        const std::vector<double>& interlevel_blurs,
        double distortion_weight,
        double distortion_power,
        double distortion_premult,
        bool compute_loss_sums);

}
