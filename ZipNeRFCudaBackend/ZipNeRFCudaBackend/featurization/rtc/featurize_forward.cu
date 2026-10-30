// forward featurization: one thread per interval computes the multisampled grid features and the given outputs
// (raw density of the fused mlp, fp32 features, fp16 features, and the last columns of the NeRF MLP inputs); with
// interleave, a warp processes the same interval of 32 consecutive rays, so the grid lookups of coherent rays mostly
// hit the same cache sectors (this kernel is bound by L2 requests)

extern "C" __global__ void featurize_forward(
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
    uint16_t* __restrict__ features_half,
    const int features_half_width,
    const uint16_t* __restrict__ encoded_directions,
    uint16_t* __restrict__ nerf_inputs,
    const int nerf_inputs_width,
    const int encoded_width,
    const uint32_t n_samples,
    const uint32_t n_intervals)
{
    using namespace zipnerf;
#if JIT_MLP_WIDTH > 0
    __shared__ float shared_w0[JIT_MLP_WIDTH * n_output_features];
    __shared__ float shared_b0[JIT_MLP_WIDTH];
    __shared__ float shared_w1[JIT_MLP_WIDTH];
    const bool with_mlp = raw_density != nullptr;
    if (with_mlp) {
        for (uint32_t i = threadIdx.x; i < JIT_MLP_WIDTH * n_output_features; i += blockDim.x) shared_w0[i] = w0[i];
        for (uint32_t i = threadIdx.x; i < JIT_MLP_WIDTH; i += blockDim.x) {
            shared_b0[i] = b0[i];
            shared_w1[i] = w1[i];
        }
        __syncthreads();
    }
#endif
    const uint32_t thread_sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_sample >= n_samples) return;
    uint32_t sample_idx = thread_sample;
    if (interleave) {
        const uint32_t group = thread_sample / (32u * n_intervals), offset = thread_sample % (32u * n_intervals);
        const uint32_t group_rays = min(32u, n_samples / n_intervals - 32u * group);
        sample_idx = (32u * group + offset % group_rays) * n_intervals + offset / group_rays;
    }

    const RayInterval interval = setup_interval(origins, directions, radii, tdist, pattern_seed, sample_idx, n_intervals);
    float features[n_output_features];
    featurize(interval, table, scale_magnitudes, features);

#if JIT_MLP_WIDTH > 0
    if (with_mlp) {
        float raw = b1[0];
        for (uint32_t j = 0; j < JIT_MLP_WIDTH; j++) {
            float pre = shared_b0[j];
            #pragma unroll
            for (uint32_t i = 0; i < n_output_features; i++) pre += shared_w0[j * n_output_features + i] * features[i];
            raw += shared_w1[j] * fmaxf(pre, 0.0f);
        }
        raw_density[sample_idx] = raw;
    }
#endif
    if (features_float != nullptr) {
        #pragma unroll
        for (uint32_t i = 0; i < n_output_features; i++) features_float[static_cast<uint64_t>(sample_idx) * n_output_features + i] = features[i];
    }
    if (features_half != nullptr) {
        uint16_t* out = features_half + static_cast<uint64_t>(sample_idx) * features_half_width;
        #pragma unroll
        for (uint32_t i = 0; i < n_output_features; i++) out[i] = float_to_half(features[i]);
        for (uint32_t i = n_output_features; i < static_cast<uint32_t>(features_half_width); i++) out[i] = 0;
    }
    if (nerf_inputs != nullptr) {
        // the first columns are the hidden features (written by the first GEMM), the last one is the bias lane
        const uint16_t* encoded = encoded_directions + static_cast<uint64_t>(sample_idx / n_intervals) * encoded_width;
        uint16_t* out = nerf_inputs + static_cast<uint64_t>(sample_idx + 1) * nerf_inputs_width - encoded_width;
        for (int i = 0; i < encoded_width - 1; i++) out[i] = encoded[i];
        out[encoded_width - 1] = 0x3c00;  // fp16 1.0
    }
}
