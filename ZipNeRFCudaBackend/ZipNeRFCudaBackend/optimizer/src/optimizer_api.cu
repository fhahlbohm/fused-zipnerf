#include "optimizer_api.h"

#include <c10/cuda/CUDAStream.h>

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace zipnerf::optimizer {

    constexpr int block_size = 256;
    constexpr int blocks_per_sm = 8;
    constexpr int max_buffers = 8;
    constexpr int max_levels = 32;
    constexpr int max_ranges = max_levels + max_buffers;

    struct Buffer {
        float4* param;
        float4* grad;
        float4* exp_avg;
        float4* exp_avg_sq;
        int64_t start;  // first group in the concatenated groups of all buffers
        float inv_grad_scale;
    };

    // consecutive groups of one buffer that are a regularized level (level >= 0) or unregularized (level -1)
    struct Range {
        int64_t end;
        int buffer;
        int level;
    };

    struct StepParams {
        Buffer buffers[max_buffers];
        Range ranges[max_ranges];
        float level_reg_weights[max_levels];
        double level_inv_sizes[max_levels];
        int n_ranges, n_levels;
        int64_t n_groups;
        float lr, beta1, beta2, eps, bias_correction1, bias_correction2;
        double max_value_sq;
    };

    __device__ inline float adam_update(float& p, float& g, float& m, float& v, const float inv_grad_scale, const float reg_weight, const StepParams& params) {
        g = g * inv_grad_scale + reg_weight * p;
        // jnp.nan_to_num semantics
        g = isnan(g) ? 0.0f : fminf(fmaxf(g, -FLT_MAX), FLT_MAX);
        m = params.beta1 * m + (1.0f - params.beta1) * g;
        v = params.beta2 * v + (1.0f - params.beta2) * g * g;
        p -= params.lr * (m / params.bias_correction1) / (sqrtf(v / params.bias_correction2) + params.eps);
        g = 0.0f;
        return p * p;
    }

    // grid-stride loop over groups of four parameters of all buffers; per-level sums of squares are accumulated in
    // registers (a thread's range only increases) and the last block reduces them into the magnitudes and the regularizer
    __global__ void adam_step_kernel(
        const StepParams params,
        double* __restrict__ level_sums,
        unsigned int* __restrict__ counter,
        float* __restrict__ magnitudes,
        float* __restrict__ regularization)
    {
        __shared__ float block_sums[max_levels];
        __shared__ bool last_block;
        for (int level = threadIdx.x; level < params.n_levels; level += blockDim.x) block_sums[level] = 0.0f;
        __syncthreads();
        int range_idx = 0;
        float range_sum = 0.0f;
        for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < params.n_groups; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
            if (idx >= params.ranges[range_idx].end) {
                if (params.ranges[range_idx].level >= 0) atomicAdd(&block_sums[params.ranges[range_idx].level], range_sum);
                range_sum = 0.0f;
                while (idx >= params.ranges[range_idx].end) range_idx++;
            }
            const Range& range = params.ranges[range_idx];
            const Buffer& buffer = params.buffers[range.buffer];
            const float reg_weight = range.level >= 0 ? params.level_reg_weights[range.level] : 0.0f;
            const int64_t i = idx - buffer.start;
            float4 p = buffer.param[i], g = buffer.grad[i], m = buffer.exp_avg[i], v = buffer.exp_avg_sq[i];
            range_sum += adam_update(p.x, g.x, m.x, v.x, buffer.inv_grad_scale, reg_weight, params);
            range_sum += adam_update(p.y, g.y, m.y, v.y, buffer.inv_grad_scale, reg_weight, params);
            range_sum += adam_update(p.z, g.z, m.z, v.z, buffer.inv_grad_scale, reg_weight, params);
            range_sum += adam_update(p.w, g.w, m.w, v.w, buffer.inv_grad_scale, reg_weight, params);
            buffer.param[i] = p;
            buffer.grad[i] = g;
            buffer.exp_avg[i] = m;
            buffer.exp_avg_sq[i] = v;
        }
        if (range_idx < params.n_ranges && params.ranges[range_idx].level >= 0) atomicAdd(&block_sums[params.ranges[range_idx].level], range_sum);
        __syncthreads();
        for (int level = threadIdx.x; level < params.n_levels; level += blockDim.x) {
            if (block_sums[level] != 0.0f) atomicAdd(&level_sums[level], static_cast<double>(block_sums[level]));
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last_block = atomicAdd(counter, 1u) == gridDim.x - 1;
        __syncthreads();
        if (!last_block || threadIdx.x >= 32) return;
        double value = 0.0;
        for (int level = threadIdx.x; level < params.n_levels; level += 32) {
            const double sum = __ldcg(&level_sums[level]);
            magnitudes[level] = static_cast<float>(sqrt(params.max_value_sq + sum * params.level_inv_sizes[level]));
            value += 0.5 * params.level_reg_weights[level] * sum;
            level_sums[level] = 0.0;
        }
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(0xffffffffu, value, offset);
        if (threadIdx.x == 0) {
            *regularization = static_cast<float>(value);
            *counter = 0;
        }
    }

    static void check_buffer(const torch::Tensor& tensor, const char* name) {
        if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != torch::kFloat || tensor.numel() % 4 != 0) {
            throw std::runtime_error(std::string("ZipNeRF: ") + name + " must be a contiguous float cuda tensor with a multiple of four elements");
        }
    }

    Adam::Adam(
        const std::vector<torch::Tensor>& params,
        const std::vector<torch::Tensor>& grads,
        const std::vector<double>& grad_scales,
        const std::vector<std::vector<int64_t>>& level_offsets,
        const double reg_weight,
        const double max_value,
        const double beta1,
        const double beta2,
        const double eps,
        torch::Tensor magnitudes) :
        m_params(params), m_grads(grads), m_grad_scales(grad_scales), m_level_offsets(level_offsets),
        m_reg_weight(reg_weight), m_max_value(max_value),
        m_beta1(static_cast<float>(beta1)), m_beta2(static_cast<float>(beta2)), m_eps(static_cast<float>(eps)), m_magnitudes(magnitudes)
    {
        const size_t n_buffers = params.size();
        if (n_buffers == 0 || n_buffers > max_buffers || grads.size() != n_buffers || grad_scales.size() != n_buffers || level_offsets.size() != n_buffers) {
            throw std::runtime_error("ZipNeRF: invalid optimizer buffers");
        }
        for (size_t buffer = 0; buffer < n_buffers; buffer++) {
            check_buffer(params[buffer], "param");
            check_buffer(grads[buffer], "grad");
            if (grads[buffer].numel() != params[buffer].numel()) throw std::runtime_error("ZipNeRF: optimizer gradient size mismatch");
            const std::vector<int64_t>& offsets = level_offsets[buffer];
            if (!offsets.empty()) {
                if (offsets.front() != 0 || offsets.back() != params[buffer].numel()) throw std::runtime_error("ZipNeRF: level offsets must cover the buffer");
                for (size_t level = 0; level + 1 < offsets.size(); level++) {
                    if (offsets[level] % 4 != 0 || offsets[level + 1] <= offsets[level]) throw std::runtime_error("ZipNeRF: invalid level offsets");
                }
                m_n_levels += static_cast<int>(offsets.size()) - 1;
            }
            m_exp_avgs.push_back(torch::zeros_like(params[buffer]));
            m_exp_avg_sqs.push_back(torch::zeros_like(params[buffer]));
        }
        if (m_n_levels > max_levels || static_cast<size_t>(m_n_levels) + n_buffers > max_ranges) throw std::runtime_error("ZipNeRF: too many optimizer levels");
        if (!magnitudes.is_cuda() || !magnitudes.is_contiguous() || magnitudes.scalar_type() != torch::kFloat || magnitudes.numel() != m_n_levels) {
            throw std::runtime_error("ZipNeRF: magnitudes must be a contiguous float cuda tensor with one entry per level");
        }
        const torch::TensorOptions options = params.front().options();
        m_regularization = torch::zeros({}, options);
        m_level_sums = torch::zeros({std::max(m_n_levels, 1)}, options.dtype(torch::kDouble));
        m_counter = torch::zeros({1}, options.dtype(torch::kInt));
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&m_n_sms, cudaDevAttrMultiProcessorCount, device);
    }

    void Adam::step(const double lr) {
        m_step++;
        StepParams params{};
        int64_t n_groups = 0;
        int level = 0;
        for (size_t buffer = 0; buffer < m_params.size(); buffer++) {
            params.buffers[buffer] = {
                reinterpret_cast<float4*>(m_params[buffer].data_ptr<float>()), reinterpret_cast<float4*>(m_grads[buffer].data_ptr<float>()),
                reinterpret_cast<float4*>(m_exp_avgs[buffer].data_ptr<float>()), reinterpret_cast<float4*>(m_exp_avg_sqs[buffer].data_ptr<float>()),
                n_groups, static_cast<float>(1.0 / m_grad_scales[buffer])};
            const std::vector<int64_t>& offsets = m_level_offsets[buffer];
            if (offsets.empty()) {
                params.ranges[params.n_ranges++] = {n_groups + m_params[buffer].numel() / 4, static_cast<int>(buffer), -1};
            }
            for (size_t i = 0; i + 1 < offsets.size(); i++, level++) {
                const int64_t size = offsets[i + 1] - offsets[i];
                params.level_reg_weights[level] = static_cast<float>(m_reg_weight / static_cast<double>(size));
                params.level_inv_sizes[level] = 1.0 / static_cast<double>(size);
                params.ranges[params.n_ranges++] = {n_groups + offsets[i + 1] / 4, static_cast<int>(buffer), level};
            }
            n_groups += m_params[buffer].numel() / 4;
        }
        params.n_levels = m_n_levels;
        params.n_groups = n_groups;
        params.lr = static_cast<float>(lr);
        params.beta1 = m_beta1;
        params.beta2 = m_beta2;
        params.eps = m_eps;
        params.bias_correction1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_beta1), static_cast<double>(m_step)));
        params.bias_correction2 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_beta2), static_cast<double>(m_step)));
        params.max_value_sq = m_max_value * m_max_value;
        const int64_t n_blocks = std::min<int64_t>((n_groups + block_size - 1) / block_size, static_cast<int64_t>(m_n_sms) * blocks_per_sm);
        adam_step_kernel<<<static_cast<unsigned int>(n_blocks), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            params, m_level_sums.data_ptr<double>(), reinterpret_cast<unsigned int*>(m_counter.data_ptr<int>()),
            m_magnitudes.data_ptr<float>(), m_regularization.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

}
