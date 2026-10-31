#include "sampling_api.h"
#include "ray_ops.cuh"
#include "random.cuh"
#include "zipnerf_config.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace zipnerf::sampling {

    constexpr int block_size = 128;
    constexpr float rgb_padding = 0.001f;
    constexpr int n_nerf_outputs = 8;  // raw density, rgb logits, zero padding

    struct ProposalLevels {
        int n_levels;
        int n_intervals[max_proposal_levels];
        const float* raw_density[max_proposal_levels];
        const float* sdist[max_proposal_levels];
        const float* tdist[max_proposal_levels];
        const float* weights[max_proposal_levels];
        float* raw_density_grad[max_proposal_levels];
        float interlevel_weight[max_proposal_levels];
        float interlevel_blur[max_proposal_levels];
    };

    __device__ inline float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(0xffffffffu, value, offset);
        return value;
    }

    __device__ inline float3 load_float3(const float* data, const int64_t idx) {
        return make_float3(data[3 * idx], data[3 * idx + 1], data[3 * idx + 2]);
    }

    __device__ inline float norm3(const float3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }

    // view direction encoding (identity, sin(2^k x) for degrees k and axes, then the cosines, zero padding) and the
    // intervals of the first level, which samples a single unit interval (jitter < 0: centered samples)
    __device__ inline void initialize_ray(
        const float3 d, const float jitter, const int direction_degree, const int n_samples, const RayWarp& warp,
        __nv_bfloat16* encoded, float* sdist, float* tdist)
    {
        const float inv_norm = 1.0f / norm3(d);
        const float v[3] = {d.x * inv_norm, d.y * inv_norm, d.z * inv_norm};
        for (int i = 0; i < config::direction_block_width(direction_degree); i++) encoded[i] = __float2bfloat16(0.0f);
        for (int c = 0; c < 3; c++) {
            encoded[c] = __float2bfloat16(v[c]);
            for (int k = 0; k < direction_degree; k++) {
                float sin_value, cos_value;
                sincosf(scalbnf(v[c], k), &sin_value, &cos_value);
                encoded[3 + 3 * k + c] = __float2bfloat16(sin_value);
                encoded[3 + 3 * (direction_degree + k) + c] = __float2bfloat16(cos_value);
            }
        }
        const float t_init[2] = {0.0f, 1.0f};
        const float w_init[1] = {1.0f};
        sample_intervals(t_init, w_init, 1, 1.0f, n_samples, jitter, sdist);
        for (int i = 0; i <= n_samples; i++) tdist[i] = s_to_t(sdist[i], warp);
    }

    // world-space ray direction of a local camera direction (c2w: row-major 3x4)
    __device__ inline float3 world_direction(const float* m, const float3 local) {
        return make_float3(m[0] * local.x + m[1] * local.y + m[2] * local.z, m[4] * local.x + m[5] * local.y + m[6] * local.z, m[8] * local.x + m[9] * local.y + m[10] * local.z);
    }

    // training: rays and colors of random pixels of all training views, initialized for the first level
    __global__ void generate_rays_kernel(
        const uint64_t seed,
        const int64_t n_pixels,
        const int64_t* __restrict__ view_offsets,
        const int n_views,
        const int* __restrict__ view_cameras,
        const int64_t* __restrict__ camera_offsets,
        const float* __restrict__ local_directions,
        const float* __restrict__ c2w,
        const float* __restrict__ camera_radii,
        const uint8_t* __restrict__ pool_rgb,
        float* __restrict__ origins,
        float* __restrict__ directions,
        float* __restrict__ radii,
        float* __restrict__ rgb,
        __nv_bfloat16* __restrict__ encoded_directions,
        float* __restrict__ sdist,
        float* __restrict__ tdist,
        const int direction_degree,
        const int n_samples,
        const RayWarp warp,
        const int n_rays)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_rays) return;
        const int64_t pixel_idx = static_cast<int64_t>(random::uniform_int(seed, random::pixel, ray_idx, n_pixels));
        // last view whose offset is <= pixel_idx
        int lo = 0, hi = n_views;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (view_offsets[mid] <= pixel_idx) lo = mid; else hi = mid;
        }
        const int camera = view_cameras[lo];
        const float3 local = load_float3(local_directions, camera_offsets[camera] + pixel_idx - view_offsets[lo]);
        const float* m = c2w + 12 * lo;
        const float3 d = world_direction(m, local);
        origins[3 * ray_idx] = m[3];
        origins[3 * ray_idx + 1] = m[7];
        origins[3 * ray_idx + 2] = m[11];
        directions[3 * ray_idx] = d.x;
        directions[3 * ray_idx + 1] = d.y;
        directions[3 * ray_idx + 2] = d.z;
        radii[ray_idx] = camera_radii[camera];
        for (int c = 0; c < 3; c++) rgb[3 * ray_idx + c] = pool_rgb[3 * pixel_idx + c] / 255.0f;
        initialize_ray(d, random::uniform(seed, random::jitter, ray_idx), direction_degree, n_samples, warp,
            encoded_directions + ray_idx * config::direction_block_width(direction_degree), sdist + ray_idx * (n_samples + 1), tdist + ray_idx * (n_samples + 1));
    }

    // inference: rays of the given pixels of a view (centered samples)
    __global__ void view_rays_kernel(
        const int64_t* __restrict__ pixel_indices,
        const float* __restrict__ local_directions,
        const float* __restrict__ c2w,
        const float radius,
        float* __restrict__ origins,
        float* __restrict__ directions,
        float* __restrict__ radii,
        __nv_bfloat16* __restrict__ encoded_directions,
        float* __restrict__ sdist,
        float* __restrict__ tdist,
        const int direction_degree,
        const int n_samples,
        const RayWarp warp,
        const int n_rays)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_rays) return;
        const float3 d = world_direction(c2w, load_float3(local_directions, pixel_indices[ray_idx]));
        origins[3 * ray_idx] = c2w[3];
        origins[3 * ray_idx + 1] = c2w[7];
        origins[3 * ray_idx + 2] = c2w[11];
        directions[3 * ray_idx] = d.x;
        directions[3 * ray_idx + 1] = d.y;
        directions[3 * ray_idx + 2] = d.z;
        radii[ray_idx] = radius;
        initialize_ray(d, -1.0f, direction_degree, n_samples, warp,
            encoded_directions + ray_idx * config::direction_block_width(direction_degree), sdist + ray_idx * (n_samples + 1), tdist + ray_idx * (n_samples + 1));
    }

    // initialization of given rays, with random jitter for seed >= 0 (as generate_rays) and centered samples otherwise
    __global__ void initialize_rays_kernel(
        const int64_t seed,
        const float* __restrict__ directions,
        __nv_bfloat16* __restrict__ encoded_directions,
        float* __restrict__ sdist,
        float* __restrict__ tdist,
        const int direction_degree,
        const int n_samples,
        const RayWarp warp,
        const int n_rays)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_rays) return;
        const float jitter = seed >= 0 ? random::uniform(seed, random::jitter, ray_idx) : -1.0f;
        initialize_ray(load_float3(directions, ray_idx), jitter, direction_degree, n_samples, warp,
            encoded_directions + ray_idx * config::direction_block_width(direction_degree), sdist + ray_idx * (n_samples + 1), tdist + ray_idx * (n_samples + 1));
    }

    // uniform random numbers of the training kernels, materialized for the reference implementation
    __global__ void sample_uniforms_kernel(const uint64_t seed, float* __restrict__ values, const uint64_t stream, const int64_t n) {
        const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n) values[idx] = random::uniform(seed, stream, idx);
    }

    __global__ void composite_and_resample_kernel(
        const float* __restrict__ raw_density,
        const float* __restrict__ sdist,
        const float* __restrict__ tdist,
        const float* __restrict__ directions,
        float* __restrict__ weights,
        float* __restrict__ sdist_next,
        float* __restrict__ tdist_next,
        const int n_intervals,
        const float anneal,
        const int n_samples,
        const int64_t seed,
        const int next_level,
        const RayWarp warp,
        const int n_rays)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_rays) return;
        const float* t = tdist + ray_idx * (n_intervals + 1);
        float density[max_intervals];
        for (int i = 0; i < n_intervals; i++) density[i] = activate_density(raw_density[ray_idx * n_intervals + i]);
        float* w = weights + ray_idx * n_intervals;
        composite(density, t, n_intervals, norm3(load_float3(directions, ray_idx)), w, nullptr);
        float* s_next = sdist_next + ray_idx * (n_samples + 1);
        const float jitter = seed >= 0 ? random::uniform(seed, random::jitter + next_level, ray_idx) : -1.0f;
        sample_intervals(sdist + ray_idx * (n_intervals + 1), w, n_intervals, anneal, n_samples, jitter, s_next);
        for (int i = 0; i <= n_samples; i++) tdist_next[ray_idx * (n_samples + 1) + i] = s_to_t(s_next[i], warp);
    }

    // densities, colors (and their sigmoids), optical depths, and weights of the final level from the NeRF MLP outputs
    __device__ inline void evaluate_final(
        const __nv_bfloat16* nerf_outputs, const float* t,
        const int n, const float direction_norm, const int ray_idx,
        float* density, float* x, float* w, float3* colors, float3* sigmoids)
    {
        for (int i = 0; i < n; i++) {
            const __nv_bfloat16* raw = nerf_outputs + n_nerf_outputs * (static_cast<int64_t>(ray_idx) * n + i);
            const float r0 = __bfloat162float(raw[0]), r1 = __bfloat162float(raw[1]), r2 = __bfloat162float(raw[2]), r3 = __bfloat162float(raw[3]);
            // a sample with non-finite outputs is empty, so that it cannot spread NaNs to the loss and all gradients
            const bool valid = isfinite(r0) && isfinite(r1) && isfinite(r2) && isfinite(r3);
            density[i] = valid ? activate_density(r0) : 0.0f;
            const float3 s = valid ? make_float3(1.0f / (1.0f + expf(-r1)), 1.0f / (1.0f + expf(-r2)), 1.0f / (1.0f + expf(-r3))) : make_float3(0.5f, 0.5f, 0.5f);
            sigmoids[i] = s;
            colors[i] = make_float3(s.x * (1.0f + 2.0f * rgb_padding) - rgb_padding, s.y * (1.0f + 2.0f * rgb_padding) - rgb_padding, s.z * (1.0f + 2.0f * rgb_padding) - rgb_padding);
        }
        composite(density, t, n, direction_norm, w, x);
    }

    // backpropagates gradients of the weights of n intervals to the raw densities (activation with bias -1)
    __device__ inline void weights_backward_to_raw(float* grad, const float* x, const float* w, const float* density, const float* raw, const float* t, const int n, const float direction_norm) {
        composite_backward(grad, x, w, n);
        for (int i = 0; i < n; i++) {
            const bool clipped = raw[i] - 1.0f > max_density_exponent;
            grad[i] = clipped ? 0.0f : grad[i] * (t[i + 1] - t[i]) * direction_norm * density[i];
        }
    }

    __global__ void train_final_kernel(
        const __nv_bfloat16* __restrict__ nerf_outputs,
        const float* __restrict__ sdist,
        const float* __restrict__ tdist,
        const ProposalLevels proposals,
        const float* __restrict__ directions,
        const uint64_t seed,
        const float* __restrict__ rgb_target,
        __nv_bfloat16* __restrict__ nerf_outputs_grad,
        float* __restrict__ loss_sums,
        const int n_intervals,
        const float charbonnier_padding,
        const float distortion_weight,
        const float distortion_power,
        const float distortion_premult,
        const int n_rays)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        // inactive threads take part in the warp reductions with zero contributions
        const bool active = ray_idx < n_rays;
        const int n = active ? n_intervals : 0;
        const float inv_n_rays = 1.0f / n_rays;
        float loss_data = 0.0f, loss_mse = 0.0f, loss_interlevel = 0.0f, loss_distortion = 0.0f;

        const float* t = tdist + static_cast<int64_t>(ray_idx) * (n_intervals + 1);
        const float* s = sdist + static_cast<int64_t>(ray_idx) * (n_intervals + 1);
        const float direction_norm = active ? norm3(load_float3(directions, ray_idx)) : 0.0f;
        float density[max_intervals], x[max_intervals], w[max_intervals], grad_w[max_intervals];
        float3 colors[max_intervals], sigmoids[max_intervals];
        evaluate_final(nerf_outputs, t, n, direction_norm, ray_idx, density, x, w, colors, sigmoids);

        // charbonnier data loss of the composited color
        float3 rgb = make_float3(0.0f, 0.0f, 0.0f);
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            rgb.x += w[i] * colors[i].x;
            rgb.y += w[i] * colors[i].y;
            rgb.z += w[i] * colors[i].z;
            acc += w[i];
        }
        float3 grad_rgb = make_float3(0.0f, 0.0f, 0.0f);
        float3 bg = make_float3(0.0f, 0.0f, 0.0f);
        if (active) {
            bg = make_float3(random::uniform(seed, random::background, 3 * ray_idx), random::uniform(seed, random::background, 3 * ray_idx + 1), random::uniform(seed, random::background, 3 * ray_idx + 2));
            const float background_weight = fmaxf(1.0f - acc, 0.0f);
            const float3 target = load_float3(rgb_target, ray_idx);
            const float r[3] = {rgb.x + background_weight * bg.x - target.x, rgb.y + background_weight * bg.y - target.y, rgb.z + background_weight * bg.z - target.z};
            float g[3];
            for (int c = 0; c < 3; c++) {
                const float q = sqrtf(r[c] * r[c] + charbonnier_padding * charbonnier_padding);
                loss_data += q * inv_n_rays / 3.0f;
                loss_mse += r[c] * r[c] * inv_n_rays / 3.0f;
                g[c] = r[c] / q * inv_n_rays / 3.0f;
            }
            grad_rgb = make_float3(g[0], g[1], g[2]);
        }
        const float grad_background = 1.0f - acc >= 0.0f ? grad_rgb.x * bg.x + grad_rgb.y * bg.y + grad_rgb.z * bg.z : 0.0f;
        for (int i = 0; i < n; i++) {
            grad_w[i] = grad_rgb.x * colors[i].x + grad_rgb.y * colors[i].y + grad_rgb.z * colors[i].z - grad_background;
        }

        // distortion loss on the curved metric distances
        if (n > 0) {
            float u[max_intervals], ds[max_intervals];
            float previous = power_ladder(t[0], distortion_power, distortion_premult);
            float w_total = 0.0f, wu_total = 0.0f;
            for (int i = 0; i < n; i++) {
                const float next = power_ladder(t[i + 1], distortion_power, distortion_premult);
                u[i] = 0.5f * (previous + next);
                ds[i] = next - previous;
                previous = next;
                w_total += w[i];
                wu_total += w[i] * u[i];
            }
            const float scale = distortion_weight * inv_n_rays;
            float w_before = 0.0f, wu_before = 0.0f, loss = 0.0f;
            for (int i = 0; i < n; i++) {
                const float w_after = w_total - w_before - w[i], wu_after = wu_total - wu_before - w[i] * u[i];
                const float inter = u[i] * w_before - wu_before + wu_after - u[i] * w_after;  // sum_j w_j |u_i - u_j|
                loss += w[i] * (u[i] * w_before - wu_before) * 2.0f + w[i] * w[i] * ds[i] / 3.0f;
                grad_w[i] += scale * (2.0f * inter + 2.0f / 3.0f * w[i] * ds[i]);
                w_before += w[i];
                wu_before += w[i] * u[i];
            }
            loss_distortion += scale * loss;
        }

        // gradients of the final level's raw outputs (raw density and rgb logits, zero padding)
        float raw[max_intervals];
        for (int i = 0; i < n; i++) raw[i] = __bfloat162float(nerf_outputs[n_nerf_outputs * (static_cast<int64_t>(ray_idx) * n_intervals + i)]);
        weights_backward_to_raw(grad_w, x, w, density, raw, t, n, direction_norm);
        for (int i = 0; i < n; i++) {
            __nv_bfloat16* grad = nerf_outputs_grad + n_nerf_outputs * (static_cast<int64_t>(ray_idx) * n_intervals + i);
            const float3 sg = sigmoids[i];
            const float color_scale = w[i] * (1.0f + 2.0f * rgb_padding);
            grad[0] = __float2bfloat16(grad_w[i]);
            grad[1] = __float2bfloat16(color_scale * grad_rgb.x * sg.x * (1.0f - sg.x));
            grad[2] = __float2bfloat16(color_scale * grad_rgb.y * sg.y * (1.0f - sg.y));
            grad[3] = __float2bfloat16(color_scale * grad_rgb.z * sg.z * (1.0f - sg.z));
            for (int c = 4; c < n_nerf_outputs; c++) grad[c] = __float2bfloat16(0.0f);
        }

        // interlevel losses of the proposal levels against the blurred final weights
        for (int level = 0; level < proposals.n_levels; level++) {
            const int n_p = active ? proposals.n_intervals[level] : 0;
            const int64_t offset = static_cast<int64_t>(ray_idx) * proposals.n_intervals[level];
            const float* s_p = proposals.sdist[level] + offset + ray_idx;
            const float* t_p = proposals.tdist[level] + offset + ray_idx;
            const float* w_p = proposals.weights[level] + offset;
            float grad_p[max_intervals], raw_p[max_intervals], density_p[max_intervals], x_p[max_intervals];
            if (n_p > 0) blur_and_resample(s_p, n_p + 1, s, w, n, proposals.interlevel_blur[level], grad_p);
            const float scale = proposals.interlevel_weight[level] * inv_n_rays / max(n_p, 1);
            for (int i = 0; i < n_p; i++) {
                const float r = fmaxf(grad_p[i] - w_p[i], 0.0f);
                const float denominator = w_p[i] + eps;
                loss_interlevel += scale * r * r / denominator;
                grad_p[i] = scale * (-2.0f * r / denominator - r * r / (denominator * denominator));
                raw_p[i] = proposals.raw_density[level][offset + i];
                density_p[i] = activate_density(raw_p[i]);
                x_p[i] = density_p[i] * (t_p[i + 1] - t_p[i]) * direction_norm;
            }
            weights_backward_to_raw(grad_p, x_p, w_p, density_p, raw_p, t_p, n_p, direction_norm);
            for (int i = 0; i < n_p; i++) proposals.raw_density_grad[level][offset + i] = grad_p[i];
        }

        loss_data = warp_sum(loss_data);
        loss_mse = warp_sum(loss_mse);
        loss_interlevel = warp_sum(loss_interlevel);
        loss_distortion = warp_sum(loss_distortion);
        if (loss_sums != nullptr && threadIdx.x % 32 == 0) {
            atomicAdd(loss_sums, loss_data);
            atomicAdd(loss_sums + 1, loss_mse);
            atomicAdd(loss_sums + 2, loss_interlevel);
            atomicAdd(loss_sums + 3, loss_distortion);
        }
    }

    // inference: color, opacity, and depth from the NeRF MLP outputs of all intervals, written to the rays' pixels
    __global__ void composite_final_kernel(
        const __nv_bfloat16* __restrict__ nerf_outputs,
        const float* __restrict__ tdist,
        const float* __restrict__ directions,
        const float* __restrict__ background,
        const int64_t* __restrict__ pixel_indices,
        float* __restrict__ rgb,
        float* __restrict__ alpha,
        float* __restrict__ depth,
        const int n_intervals,
        const int n_rays,
        const bool use_median_depth)
    {
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_rays) return;
        const float* t = tdist + static_cast<int64_t>(ray_idx) * (n_intervals + 1);
        float density[max_intervals], x[max_intervals], w[max_intervals];
        float3 colors[max_intervals], sigmoids[max_intervals];
        evaluate_final(nerf_outputs, t, n_intervals, norm3(load_float3(directions, ray_idx)), ray_idx, density, x, w, colors, sigmoids);
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float acc = 0.0f;
        for (int i = 0; i < n_intervals; i++) {
            color.x += w[i] * colors[i].x;
            color.y += w[i] * colors[i].y;
            color.z += w[i] * colors[i].z;
            acc += w[i];
        }
        const float background_weight = fmaxf(1.0f - acc, 0.0f);
        const int64_t pixel = pixel_indices[ray_idx];
        rgb[3 * pixel] = color.x + background_weight * background[0];
        rgb[3 * pixel + 1] = color.y + background_weight * background[1];
        rgb[3 * pixel + 2] = color.z + background_weight * background[2];
        if (alpha != nullptr) alpha[pixel] = acc;
        if (depth != nullptr) {
            if (use_median_depth) {
                // start of the interval where the normalized weights reach one half, as interpolating within it is unbounded
                // at silhouettes, whose background is covered by intervals that extend to the far plane
                const float target = 0.5f * acc;
                float cw = 0.0f, median = t[n_intervals];
                for (int i = 0; i < n_intervals; i++) {
                    cw += w[i];
                    if (w[i] > 0.0f && cw >= target) {
                        median = t[i];
                        break;
                    }
                }
                depth[pixel] = median;
            }
            else {
                // expected log distance as in the official code, so that small weights far out do not dominate
                float weighted_log_t = 0.0f;
                for (int i = 0; i < n_intervals; i++) weighted_log_t += w[i] * logf(0.5f * (t[i] + t[i + 1]));
                depth[pixel] = fminf(fmaxf(expf(weighted_log_t / fmaxf(acc, eps)), t[0]), t[n_intervals]);
            }
        }
    }

    static void check_input(const torch::Tensor& tensor, const char* name, const torch::ScalarType dtype) {
        if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != dtype) {
            throw std::runtime_error(std::string("ZipNeRF: ") + name + " must be a contiguous cuda tensor of the expected type");
        }
    }

    static RayWarp make_ray_warp(const std::vector<double>& values) {
        if (values.size() != 7) throw std::runtime_error("ZipNeRF: invalid ray warp");
        return {
            static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]), static_cast<float>(values[3]),
            static_cast<float>(values[4]), static_cast<float>(values[5]), static_cast<float>(values[6])
        };
    }

    static int n_blocks(const int64_t n) { return static_cast<int>((n + block_size - 1) / block_size); }

    static void check_n_intervals(const int64_t n) {
        if (n < 2 || n > max_intervals) throw std::runtime_error("ZipNeRF: unsupported number of samples per level");
    }

    std::vector<torch::Tensor> generate_rays_wrapper(
        const int64_t seed,
        const int n_rays,
        const torch::Tensor& view_offsets,
        const torch::Tensor& view_cameras,
        const torch::Tensor& camera_offsets,
        const torch::Tensor& local_directions,
        const torch::Tensor& c2w,
        const torch::Tensor& camera_radii,
        const torch::Tensor& pool_rgb,
        const int direction_degree,
        const int n_samples,
        const std::vector<double>& ray_warp)
    {
        check_input(view_offsets, "view_offsets", torch::kLong);
        check_input(view_cameras, "view_cameras", torch::kInt);
        check_input(camera_offsets, "camera_offsets", torch::kLong);
        check_input(local_directions, "local_directions", torch::kFloat);
        check_input(c2w, "c2w", torch::kFloat);
        check_input(camera_radii, "camera_radii", torch::kFloat);
        check_input(pool_rgb, "pool_rgb", torch::kByte);
        check_n_intervals(n_samples);
        const torch::TensorOptions options = local_directions.options();
        torch::Tensor origins = torch::empty({n_rays, 3}, options);
        torch::Tensor directions = torch::empty({n_rays, 3}, options);
        torch::Tensor radii = torch::empty({n_rays}, options);
        torch::Tensor rgb = torch::empty({n_rays, 3}, options);
        torch::Tensor encoded_directions = torch::empty({n_rays, config::direction_block_width(direction_degree)}, options.dtype(torch::kBFloat16));
        torch::Tensor sdist = torch::empty({n_rays, n_samples + 1}, options);
        torch::Tensor tdist = torch::empty({n_rays, n_samples + 1}, options);
        if (n_rays > 0) {
            generate_rays_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
                static_cast<uint64_t>(seed), pool_rgb.size(0), view_offsets.data_ptr<int64_t>(), static_cast<int>(view_cameras.size(0)),
                view_cameras.data_ptr<int>(), camera_offsets.data_ptr<int64_t>(), local_directions.data_ptr<float>(),
                c2w.data_ptr<float>(), camera_radii.data_ptr<float>(), pool_rgb.data_ptr<uint8_t>(),
                origins.data_ptr<float>(), directions.data_ptr<float>(), radii.data_ptr<float>(), rgb.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(encoded_directions.data_ptr<torch::BFloat16>()), sdist.data_ptr<float>(), tdist.data_ptr<float>(),
                direction_degree, n_samples, make_ray_warp(ray_warp), n_rays);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {origins, directions, radii, rgb, encoded_directions, sdist, tdist};
    }

    std::vector<torch::Tensor> view_rays_wrapper(
        const torch::Tensor& pixel_indices,
        const torch::Tensor& local_directions,
        const torch::Tensor& c2w,
        const double radius,
        const int direction_degree,
        const int n_samples,
        const std::vector<double>& ray_warp)
    {
        check_input(pixel_indices, "pixel_indices", torch::kLong);
        check_input(local_directions, "local_directions", torch::kFloat);
        check_input(c2w, "c2w", torch::kFloat);
        if (c2w.numel() != 12) throw std::runtime_error("ZipNeRF: c2w must be a 3x4 matrix");
        check_n_intervals(n_samples);
        const int n_rays = static_cast<int>(pixel_indices.numel());
        const torch::TensorOptions options = local_directions.options();
        torch::Tensor origins = torch::empty({n_rays, 3}, options);
        torch::Tensor directions = torch::empty({n_rays, 3}, options);
        torch::Tensor radii = torch::empty({n_rays}, options);
        torch::Tensor encoded_directions = torch::empty({n_rays, config::direction_block_width(direction_degree)}, options.dtype(torch::kBFloat16));
        torch::Tensor sdist = torch::empty({n_rays, n_samples + 1}, options);
        torch::Tensor tdist = torch::empty({n_rays, n_samples + 1}, options);
        if (n_rays > 0) {
            view_rays_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
                pixel_indices.data_ptr<int64_t>(), local_directions.data_ptr<float>(), c2w.data_ptr<float>(), static_cast<float>(radius),
                origins.data_ptr<float>(), directions.data_ptr<float>(), radii.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(encoded_directions.data_ptr<torch::BFloat16>()), sdist.data_ptr<float>(), tdist.data_ptr<float>(),
                direction_degree, n_samples, make_ray_warp(ray_warp), n_rays);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return {origins, directions, radii, encoded_directions, sdist, tdist};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> initialize_rays_wrapper(
        const int64_t seed,
        const torch::Tensor& directions,
        const int direction_degree,
        const int n_samples,
        const std::vector<double>& ray_warp)
    {
        check_input(directions, "directions", torch::kFloat);
        check_n_intervals(n_samples);
        const int n_rays = static_cast<int>(directions.size(0));
        torch::Tensor encoded_directions = torch::empty({n_rays, config::direction_block_width(direction_degree)}, directions.options().dtype(torch::kBFloat16));
        torch::Tensor sdist = torch::empty({n_rays, n_samples + 1}, directions.options());
        torch::Tensor tdist = torch::empty({n_rays, n_samples + 1}, directions.options());
        if (n_rays == 0) return {encoded_directions, sdist, tdist};
        initialize_rays_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            seed, directions.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(encoded_directions.data_ptr<torch::BFloat16>()), sdist.data_ptr<float>(), tdist.data_ptr<float>(),
            direction_degree, n_samples, make_ray_warp(ray_warp), n_rays);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {encoded_directions, sdist, tdist};
    }

    torch::Tensor sample_uniforms_wrapper(const int64_t seed, const int64_t stream, const int64_t n) {
        torch::Tensor values = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        if (n == 0) return values;
        sample_uniforms_kernel<<<n_blocks(n), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            static_cast<uint64_t>(seed), values.data_ptr<float>(), static_cast<uint64_t>(stream), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return values;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> composite_and_resample_wrapper(
        const torch::Tensor& raw_density,
        const torch::Tensor& sdist,
        const torch::Tensor& tdist,
        const torch::Tensor& directions,
        const double anneal,
        const int n_samples,
        const int64_t seed,
        const int next_level,
        const std::vector<double>& ray_warp)
    {
        check_input(raw_density, "raw_density", torch::kFloat);
        check_input(sdist, "sdist", torch::kFloat);
        check_input(tdist, "tdist", torch::kFloat);
        check_input(directions, "directions", torch::kFloat);
        check_n_intervals(n_samples);
        const int n_rays = static_cast<int>(raw_density.size(0));
        const int n_intervals = static_cast<int>(raw_density.size(1));
        check_n_intervals(n_intervals);
        torch::Tensor weights = torch::empty_like(raw_density);
        torch::Tensor sdist_next = torch::empty({n_rays, n_samples + 1}, raw_density.options());
        torch::Tensor tdist_next = torch::empty({n_rays, n_samples + 1}, raw_density.options());
        if (n_rays == 0) return {weights, sdist_next, tdist_next};
        composite_and_resample_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            raw_density.data_ptr<float>(), sdist.data_ptr<float>(), tdist.data_ptr<float>(), directions.data_ptr<float>(),
            weights.data_ptr<float>(), sdist_next.data_ptr<float>(), tdist_next.data_ptr<float>(),
            n_intervals, static_cast<float>(anneal), n_samples, seed, next_level, make_ray_warp(ray_warp), n_rays);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {weights, sdist_next, tdist_next};
    }

    std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> train_final_wrapper(
        const torch::Tensor& nerf_outputs,
        const torch::Tensor& sdist,
        const torch::Tensor& tdist,
        const std::vector<torch::Tensor>& proposal_raw_densities,
        const std::vector<torch::Tensor>& proposal_sdists,
        const std::vector<torch::Tensor>& proposal_tdists,
        const std::vector<torch::Tensor>& proposal_weights,
        const torch::Tensor& directions,
        const int64_t seed,
        const torch::Tensor& rgb_target,
        const double charbonnier_padding,
        const std::vector<double>& interlevel_weights,
        const std::vector<double>& interlevel_blurs,
        const double distortion_weight,
        const double distortion_power,
        const double distortion_premult,
        const bool compute_loss_sums)
    {
        check_input(nerf_outputs, "nerf_outputs", torch::kBFloat16);
        check_input(sdist, "sdist", torch::kFloat);
        check_input(tdist, "tdist", torch::kFloat);
        check_input(directions, "directions", torch::kFloat);
        check_input(rgb_target, "rgb_target", torch::kFloat);
        const int n_rays = static_cast<int>(tdist.size(0));
        const int n_intervals = static_cast<int>(tdist.size(1)) - 1;
        check_n_intervals(n_intervals);
        if (nerf_outputs.numel() != static_cast<int64_t>(n_rays) * n_intervals * n_nerf_outputs) throw std::runtime_error("ZipNeRF: invalid nerf_outputs");
        const size_t n_levels = proposal_raw_densities.size();
        if (n_levels > max_proposal_levels || proposal_sdists.size() != n_levels
            || proposal_tdists.size() != n_levels || proposal_weights.size() != n_levels
            || interlevel_weights.size() != n_levels || interlevel_blurs.size() != n_levels) {
            throw std::runtime_error("ZipNeRF: inconsistent proposal levels");
        }
        ProposalLevels proposals{};
        proposals.n_levels = static_cast<int>(n_levels);
        std::vector<torch::Tensor> proposal_raw_density_grads;
        for (size_t level = 0; level < n_levels; level++) {
            check_input(proposal_raw_densities[level], "proposal_raw_density", torch::kFloat);
            check_input(proposal_sdists[level], "proposal_sdist", torch::kFloat);
            check_input(proposal_tdists[level], "proposal_tdist", torch::kFloat);
            check_input(proposal_weights[level], "proposal_weights", torch::kFloat);
            check_n_intervals(proposal_raw_densities[level].size(1));
            proposal_raw_density_grads.push_back(torch::empty_like(proposal_raw_densities[level]));
            proposals.n_intervals[level] = static_cast<int>(proposal_raw_densities[level].size(1));
            proposals.raw_density[level] = proposal_raw_densities[level].data_ptr<float>();
            proposals.sdist[level] = proposal_sdists[level].data_ptr<float>();
            proposals.tdist[level] = proposal_tdists[level].data_ptr<float>();
            proposals.weights[level] = proposal_weights[level].data_ptr<float>();
            proposals.raw_density_grad[level] = proposal_raw_density_grads[level].data_ptr<float>();
            proposals.interlevel_weight[level] = static_cast<float>(interlevel_weights[level]);
            proposals.interlevel_blur[level] = static_cast<float>(interlevel_blurs[level]);
        }
        torch::Tensor nerf_outputs_grad = torch::empty_like(nerf_outputs);
        torch::Tensor loss_sums = compute_loss_sums ? torch::zeros({4}, tdist.options()) : torch::Tensor();
        if (n_rays == 0) return {nerf_outputs_grad, proposal_raw_density_grads, loss_sums};
        train_final_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(nerf_outputs.data_ptr<torch::BFloat16>()), sdist.data_ptr<float>(), tdist.data_ptr<float>(),
            proposals, directions.data_ptr<float>(), static_cast<uint64_t>(seed), rgb_target.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(nerf_outputs_grad.data_ptr<torch::BFloat16>()),
            compute_loss_sums ? loss_sums.data_ptr<float>() : nullptr, n_intervals, static_cast<float>(charbonnier_padding), static_cast<float>(distortion_weight),
            static_cast<float>(distortion_power), static_cast<float>(distortion_premult), n_rays);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {nerf_outputs_grad, proposal_raw_density_grads, loss_sums};
    }

    void composite_final_wrapper(
        const torch::Tensor& nerf_outputs,
        const torch::Tensor& tdist,
        const torch::Tensor& directions,
        const torch::Tensor& background,
        const torch::Tensor& pixel_indices,
        torch::Tensor& rgb,
        torch::Tensor& alpha,
        torch::Tensor& depth,
        const bool use_median_depth)
    {
        check_input(nerf_outputs, "nerf_outputs", torch::kBFloat16);
        check_input(tdist, "tdist", torch::kFloat);
        check_input(directions, "directions", torch::kFloat);
        check_input(background, "background", torch::kFloat);
        check_input(pixel_indices, "pixel_indices", torch::kLong);
        check_input(rgb, "rgb", torch::kFloat);
        if (alpha.numel() > 0) check_input(alpha, "alpha", torch::kFloat);
        if (depth.numel() > 0) check_input(depth, "depth", torch::kFloat);
        const int n_rays = static_cast<int>(tdist.size(0));
        const int n_intervals = static_cast<int>(tdist.size(1)) - 1;
        check_n_intervals(n_intervals);
        if (nerf_outputs.numel() != static_cast<int64_t>(n_rays) * n_intervals * n_nerf_outputs || pixel_indices.numel() != n_rays) throw std::runtime_error("ZipNeRF: invalid composite_final inputs");
        if (n_rays == 0) return;
        composite_final_kernel<<<n_blocks(n_rays), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(nerf_outputs.data_ptr<torch::BFloat16>()), tdist.data_ptr<float>(), directions.data_ptr<float>(),
            background.data_ptr<float>(), pixel_indices.data_ptr<int64_t>(), rgb.data_ptr<float>(),
            alpha.numel() > 0 ? alpha.data_ptr<float>() : nullptr, depth.numel() > 0 ? depth.data_ptr<float>() : nullptr,
            n_intervals, n_rays, use_median_depth);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

}
