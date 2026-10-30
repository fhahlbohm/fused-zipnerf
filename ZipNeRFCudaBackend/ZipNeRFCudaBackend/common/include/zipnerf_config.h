#pragma once

#include <cstdint>

namespace zipnerf::config {

    // hash grid initialization range and feature preconditioning of the official implementation
    inline constexpr double hash_init_range = 1e-4;
    inline constexpr double precondition_scaling = 10.0;
    // static fp16 loss scale of the NeRF MLP gradients
    inline constexpr double nerf_mlp_grad_scale = 1024.0;

    // width of the encoded view direction (identity, sines, and cosines) and the bias lane, padded to a multiple of 8
    inline constexpr int64_t direction_block_width(const int64_t degree) { return (3 + 6 * degree + 1 + 7) / 8 * 8; }

}
