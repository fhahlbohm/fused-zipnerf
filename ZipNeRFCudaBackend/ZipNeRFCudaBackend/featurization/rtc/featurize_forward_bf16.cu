// Packed BF16 final-grid outputs; the table and accumulation stay FP32.
namespace zipnerf {
    __device__ inline void featurize_packed(
        const RayInterval& interval, const float* __restrict__ table,
        const float* __restrict__ scale_magnitudes, uint32_t* features)
    {
        float3 x[n_multisamples];
        float scale[n_multisamples];
        #pragma unroll
        for (uint32_t j = 0; j < n_multisamples; j++) multisample(interval, j, x[j], scale[j]);
        #pragma unroll
        for (uint32_t level = 0; level < JIT_N_LEVELS; level++) {
            float weighting[n_multisamples];
            float scale_feature = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < n_multisamples; j++) {
                weighting[j] = level_weighting(level, scale[j]);
                scale_feature += 2.0f * weighting[j] - 1.0f;
            }
            GatherFeatures gather = {table, {0.0f}};
            visit_level_corners(level, x, weighting, gather);
            #pragma unroll
            for (uint32_t f = 0; f < n_level_features; f++) {
                const uint32_t i = level * n_level_features + f;
                const float value = f < JIT_N_FEATURES ? feature_scale * gather.values[f % JIT_N_FEATURES]
                    : feature_scale * scale_feature * scale_magnitudes[level];
                const uint32_t half = float_to_bf16(value);
                if (i % 2 == 0) features[i / 2] = half;
                else features[i / 2] |= half << 16;
            }
        }
    }
}

extern "C" __global__ void featurize_forward_bf16(
    const float3* __restrict__ origins,
    const float3* __restrict__ directions,
    const float* __restrict__ radii,
    const float* __restrict__ tdist,
    const long long pattern_seed,
    const bool interleave,
    const float* __restrict__ table,
    const float* __restrict__ scale_magnitudes,
    const float* __restrict__ w0,
    const float* __restrict__ b0,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    float* __restrict__ raw_density,
    float* __restrict__ features_float,
    uint16_t* __restrict__ features_bf16,
    const int features_bf16_width,
    const uint16_t* __restrict__ encoded_directions,
    uint16_t* __restrict__ nerf_inputs,
    const int nerf_inputs_width,
    const int encoded_width,
    const uint32_t n_samples,
    const uint32_t n_intervals)
{
    using namespace zipnerf;
    const uint32_t thread_sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_sample >= n_samples) return;
    uint32_t sample_idx = thread_sample;
    if (interleave) {
        const uint32_t group = thread_sample / (32u * n_intervals), offset = thread_sample % (32u * n_intervals);
        const uint32_t group_rays = min(32u, n_samples / n_intervals - 32u * group);
        sample_idx = (32u * group + offset % group_rays) * n_intervals + offset / group_rays;
    }
    const RayInterval interval = setup_interval(origins, directions, radii, tdist, pattern_seed, sample_idx, n_intervals);
    constexpr uint32_t n_packed = (n_output_features + 1) / 2;
    constexpr uint32_t n_vectors = (n_output_features + 7) / 8;
    uint32_t features[n_packed];
    featurize_packed(interval, table, scale_magnitudes, features);
    uint4* out = reinterpret_cast<uint4*>(features_bf16 + static_cast<uint64_t>(sample_idx) * features_bf16_width);
    #pragma unroll
    for (uint32_t i = 0; i < n_vectors; i++) {
        out[i] = make_uint4(
            4 * i < n_packed ? features[4 * i] : 0u,
            4 * i + 1 < n_packed ? features[4 * i + 1] : 0u,
            4 * i + 2 < n_packed ? features[4 * i + 2] : 0u,
            4 * i + 3 < n_packed ? features[4 * i + 3] : 0u);
    }
    for (uint32_t i = n_vectors; i < static_cast<uint32_t>(features_bf16_width / 8); i++) out[i] = make_uint4(0, 0, 0, 0);
    if (nerf_inputs != nullptr) {
        const uint4* encoded = reinterpret_cast<const uint4*>(encoded_directions + static_cast<uint64_t>(sample_idx / n_intervals) * encoded_width);
        uint16_t* inputs = nerf_inputs + static_cast<uint64_t>(sample_idx + 1) * nerf_inputs_width - encoded_width;
        uint4* encoded_out = reinterpret_cast<uint4*>(inputs);
        for (int i = 0; i < encoded_width / 8; i++) encoded_out[i] = encoded[i];
        inputs[encoded_width - 1] = 0x3f80;
    }
}
