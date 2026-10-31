#pragma once

#include <cstdint>

namespace zipnerf::config {

    // hash grid initialization range and feature preconditioning of the official implementation
    inline constexpr double hash_init_range = 1e-4;
    inline constexpr double precondition_scaling = 10.0;
    // static scale of the NeRF grid gradients (about 1e-15 to 1e-2): keeps the largest far from fp16 overflow and full
    // fp16 precision down to about 6e-11 (smaller ones are dominated by the grid regularizer)
    inline constexpr double nerf_grid_grad_scale = 1048576.0;
    // NeRF grid levels with fewer entries accumulate their gradients in fp32, as their entries receive too many
    // (mostly canceling) contributions per step for an fp16 sum
    inline constexpr int64_t nerf_grid_min_half_grad_entries = 1 << 20;

    // width of the encoded view direction (identity, sines, and cosines) and the bias lane, padded to a multiple of 8
    inline constexpr int64_t direction_block_width(const int64_t degree) { return (3 + 6 * degree + 1 + 7) / 8 * 8; }

}
