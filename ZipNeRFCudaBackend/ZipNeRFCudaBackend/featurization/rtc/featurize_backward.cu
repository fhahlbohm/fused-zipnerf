// backward featurization of the NeRF level: one thread per interval scatters the feature gradient into the grid gradient

extern "C" __global__ void featurize_backward(
    const float3* __restrict__ origins,
    const float3* __restrict__ directions,
    const float* __restrict__ radii,
    const float* __restrict__ tdist,
    const long long pattern_seed,
    const uint16_t* __restrict__ grad_features,
    const int grad_features_stride,
    const float inv_grad_scale,
    float* __restrict__ grad_table,
    const uint32_t n_samples,
    const uint32_t n_intervals)
{
    using namespace zipnerf;
    const uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) return;
    float grad[n_output_features];
    const uint16_t* row = grad_features + static_cast<uint64_t>(sample_idx) * grad_features_stride;
    #pragma unroll
    for (uint32_t i = 0; i < n_output_features; i++) grad[i] = feature_scale * inv_grad_scale * half_to_float(row[i]);
    const RayInterval interval = setup_interval(origins, directions, radii, tdist, pattern_seed, sample_idx, n_intervals);
    scatter_feature_gradients(interval, grad, grad_table);
}
