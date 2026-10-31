// backward of a proposal level: one thread per interval recomputes the fused mlp from the stored features and scatters
// its input gradient into the grid gradient; the parameter gradients are reduced per block in shared memory, accumulated
// in registers over the block-strided loop, and added to the global gradients once per block
//
// with dpre_j = [pre_j > 0] g w1_j: grad_w0[j][i] = w1_j sum [pre_j > 0] g f_i, grad_b0[j] = w1_j sum [pre_j > 0] g,
// grad_w1[j] = sum g relu(pre_j), grad_b1 = sum g

constexpr uint32_t n_w0 = JIT_MLP_WIDTH * zipnerf::n_output_features;
constexpr uint32_t n_mlp_params = n_w0 + 2 * JIT_MLP_WIDTH + 1;
constexpr uint32_t block_size = JIT_BLOCK_SIZE;
constexpr uint32_t n_outputs_per_thread = (n_mlp_params + block_size - 1) / block_size;

extern "C" __global__ void __launch_bounds__(JIT_BLOCK_SIZE) featurize_backward_mlp(
    const float3* __restrict__ origins,
    const float3* __restrict__ directions,
    const float* __restrict__ radii,
    const float* __restrict__ tdist,
    const long long pattern_seed,
    const float* __restrict__ features,
    const float* __restrict__ grad_raw_density,
    const float* __restrict__ w0,
    const float* __restrict__ b0,
    const float* __restrict__ w1,
    float* __restrict__ grad_table,
    float* __restrict__ grad_w0,
    float* __restrict__ grad_b0,
    float* __restrict__ grad_w1,
    float* __restrict__ grad_b1,
    const uint32_t n_samples,
    const uint32_t n_intervals)
{
    using namespace zipnerf;
    __shared__ float shared_w0[n_w0];
    __shared__ float shared_b0[JIT_MLP_WIDTH];
    __shared__ float shared_w1[JIT_MLP_WIDTH];
    // per sample of the current block step: pre-activations, features, and raw density gradient
    extern __shared__ float shared_samples[];
    float* shared_pre = shared_samples;
    float* shared_features = shared_pre + block_size * JIT_MLP_WIDTH;
    float* shared_grad = shared_features + block_size * n_output_features;
    for (uint32_t i = threadIdx.x; i < n_w0; i += block_size) shared_w0[i] = w0[i];
    for (uint32_t i = threadIdx.x; i < JIT_MLP_WIDTH; i += block_size) {
        shared_b0[i] = b0[i];
        shared_w1[i] = w1[i];
    }
    float accumulators[n_outputs_per_thread];
    #pragma unroll
    for (uint32_t q = 0; q < n_outputs_per_thread; q++) accumulators[q] = 0.0f;
    __syncthreads();

    // the loop bound is uniform per block, so all threads reach the barriers
    for (uint32_t base = blockIdx.x * block_size; base < n_samples; base += gridDim.x * block_size) {
        const uint32_t sample_idx = base + threadIdx.x;
        const bool active = sample_idx < n_samples;
        float f[n_output_features];
        #pragma unroll
        for (uint32_t i = 0; i < n_output_features; i++) f[i] = active ? features[static_cast<uint64_t>(sample_idx) * n_output_features + i] : 0.0f;
        const float g = active ? grad_raw_density[sample_idx] : 0.0f;
        float grad_f[n_output_features];
        #pragma unroll
        for (uint32_t i = 0; i < n_output_features; i++) grad_f[i] = 0.0f;
        for (uint32_t j = 0; j < JIT_MLP_WIDTH; j++) {
            float pre = shared_b0[j];
            #pragma unroll
            for (uint32_t i = 0; i < n_output_features; i++) pre += shared_w0[j * n_output_features + i] * f[i];
            shared_pre[threadIdx.x * JIT_MLP_WIDTH + j] = pre;
            if (pre > 0.0f) {
                const float grad_pre = g * shared_w1[j];
                #pragma unroll
                for (uint32_t i = 0; i < n_output_features; i++) grad_f[i] += grad_pre * shared_w0[j * n_output_features + i];
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < n_output_features; i++) shared_features[threadIdx.x * n_output_features + i] = f[i];
        shared_grad[threadIdx.x] = g;
        if (active && g != 0.0f) {
            #pragma unroll
            for (uint32_t i = 0; i < n_output_features; i++) grad_f[i] *= feature_scale;
            const RayInterval interval = setup_interval(origins, directions, radii, tdist, pattern_seed, sample_idx, n_intervals);
            scatter_feature_gradients<JIT_N_LEVELS>(interval, grad_f, grad_table, nullptr);
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t q = 0; q < n_outputs_per_thread; q++) {
            const uint32_t output = threadIdx.x + q * block_size;
            if (output >= n_mlp_params) break;
            float sum = 0.0f;
            if (output < n_w0) {
                const uint32_t j = output / n_output_features, i = output % n_output_features;
                for (uint32_t s = 0; s < block_size; s++) {
                    if (shared_pre[s * JIT_MLP_WIDTH + j] > 0.0f) sum += shared_grad[s] * shared_features[s * n_output_features + i];
                }
            }
            else if (output < n_w0 + JIT_MLP_WIDTH) {
                const uint32_t j = output - n_w0;
                for (uint32_t s = 0; s < block_size; s++) {
                    if (shared_pre[s * JIT_MLP_WIDTH + j] > 0.0f) sum += shared_grad[s];
                }
            }
            else if (output < n_w0 + 2 * JIT_MLP_WIDTH) {
                const uint32_t j = output - n_w0 - JIT_MLP_WIDTH;
                for (uint32_t s = 0; s < block_size; s++) sum += shared_grad[s] * fmaxf(shared_pre[s * JIT_MLP_WIDTH + j], 0.0f);
            }
            else {
                for (uint32_t s = 0; s < block_size; s++) sum += shared_grad[s];
            }
            accumulators[q] += sum;
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t q = 0; q < n_outputs_per_thread; q++) {
        const uint32_t output = threadIdx.x + q * block_size;
        if (output >= n_mlp_params || accumulators[q] == 0.0f) continue;
        if (output < n_w0) atomicAdd(grad_w0 + output, shared_w1[output / n_output_features] * accumulators[q]);
        else if (output < n_w0 + JIT_MLP_WIDTH) atomicAdd(grad_b0 + output - n_w0, shared_w1[output - n_w0] * accumulators[q]);
        else if (output < n_w0 + 2 * JIT_MLP_WIDTH) atomicAdd(grad_w1 + output - n_w0 - JIT_MLP_WIDTH, accumulators[q]);
        else atomicAdd(grad_b1, accumulators[q]);
    }
}
