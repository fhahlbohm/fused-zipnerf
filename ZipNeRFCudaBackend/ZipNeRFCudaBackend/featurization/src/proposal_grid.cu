#include "proposal_grid.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <climits>
#include <stdexcept>

namespace zipnerf::featurization {

    namespace {

        struct Pyramid {
            const float* data[12];
            int sizes[12];
            int count;
        };

        __device__ float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
        __device__ float3 mul(float a, float3 b) { return make_float3(a * b.x, a * b.y, a * b.z); }
        __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __device__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
        __device__ float3 normalized(float3 a) { return mul(rsqrtf(dot(a, a)), a); }

        // Cell centers in the same [-2, 2] contracted sphere used by native grids.
        // Outside the reachable sphere, extend the closest angular direction rather
        // than introducing an artificial zero-density boundary in interpolation.
        __global__ void proposal_grid_points_kernel(float3* points, int resolution, int64_t start, int count) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count) return;
            int64_t index = start + i;
            float3 q = make_float3(
                4.f * ((index % resolution + .5f) / resolution - .5f),
                4.f * (((index / resolution) % resolution + .5f) / resolution - .5f),
                4.f * ((index / (resolution * resolution) + .5f) / resolution - .5f));
            float r = sqrtf(dot(q, q));
            if (r > 1.f) {
                float clamped = fminf(r, 2.f - 1.e-5f);
                q = mul(1.f / (r * (2.f - clamped)), q);
            }
            points[i] = q;
        }

        __device__ float trilinear(const float* table, int n, float3 x) {
            float p[3] = {x.x * n - .5f, x.y * n - .5f, x.z * n - .5f};
            int base[3];
            float f[3];
            #pragma unroll
            for (int d = 0; d < 3; d++) {
                float v = fminf(fmaxf(p[d], 0.f), n - 1.f);
                base[d] = min(static_cast<int>(v), n - 1);
                f[d] = v - base[d];
            }
            float value = 0.f;
            #pragma unroll
            for (int corner = 0; corner < 8; corner++) {
                int ix = min(base[0] + ((corner >> 2) & 1), n - 1);
                int iy = min(base[1] + ((corner >> 1) & 1), n - 1);
                int iz = min(base[2] + (corner & 1), n - 1);
                float weight = ((corner & 4) ? f[0] : 1.f - f[0]) *
                    ((corner & 2) ? f[1] : 1.f - f[1]) * ((corner & 1) ? f[2] : 1.f - f[2]);
                value += weight * __ldg(table + ix + n * (iy + n * iz));
            }
            return value;
        }

        __device__ float lookup(Pyramid pyramid, float3 world, float footprint, bool use_mips) {
            float norm_sq = fmaxf(dot(world, world), 1.f);
            float norm = sqrtf(norm_sq);
            float contraction = (2.f * norm - 1.f) / norm_sq;
            float3 x = add(mul(.25f * contraction, world), make_float3(.5f, .5f, .5f));
            float isoscale = expf((2.f / 3.f) * logf(2.f * norm - 1.f) - logf(norm_sq));
            // Approximate diameter of the footprint, in base-grid cells.
            float lod = use_mips ? log2f(fmaxf(2.f * footprint * isoscale * .25f * pyramid.sizes[0], 1.f)) : 0.f;
            lod = fminf(lod, pyramid.count - 1.f);
            int level = static_cast<int>(lod);
            float value = trilinear(pyramid.data[level], pyramid.sizes[level], x);
            if (level + 1 < pyramid.count && lod > level) {
                float other = trilinear(pyramid.data[level + 1], pyramid.sizes[level + 1], x);
                value += (lod - level) * (other - value);
            }
            return value;
        }

        // Native deterministic conical-frustum pattern, with density read from a
        // precomputed grid instead of evaluating an MLP on averaged hash features.
        // Averaging densities is an approximation to that nonlinear operation.
        __global__ void sample_proposal_grid_kernel(Pyramid pyramid, const float3* origins, const float3* directions,
            const float* radii, const float* tdist, float* raw, int rays, int intervals, bool use_mips, bool multisample, bool raw_space) {
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= rays * intervals) return;
            int group = id / (32 * intervals), offset = id % (32 * intervals);
            int group_rays = min(32, rays - 32 * group);
            int ray = 32 * group + offset % group_rays;
            int interval = offset / group_rays;
            const float* t = tdist + ray * (intervals + 1) + interval;
            float s = .5f * (t[0] + t[1]), h = .5f * (t[1] - t[0]);
            float denominator = h * h + 3.f * s * s;
            float ratio = fabsf(denominator) < 1.17549435e-38f ? 0.f : h / denominator;
            float base = t[1] * t[1] + 2.f * s * s;
            float q = h * h - s * s;
            float root = sqrtf(fmaxf(q * q + 4.f * s * s * s * s, 0.f));
            float3 d = directions[ray], origin = origins[ray];
            float3 cross1 = cross(d, make_float3(0, 0, 1));
            if (fabsf(cross1.x) < 1.1920929e-7f && fabsf(cross1.y) < 1.1920929e-7f && fabsf(cross1.z) < 1.1920929e-7f)
                cross1 = cross(d, make_float3(1, 1, 1));
            float3 axis1 = normalized(cross1), axis2 = normalized(cross(d, cross1));
            bool flip = interval % 2 == 1;
            float value = 0.f;
            if (multisample) {
                constexpr float pattern[6] = {0.f, 2.f, 4.f, 3.f, 5.f, 1.f};
                #pragma unroll
                for (int j = 0; j < 6; j++) {
                    float theta = (3.14159265358979f / 3.f) * pattern[flip ? 5 - j : j] + (flip ? 3.14159265358979f / 6.f : 0.f);
                    float cz = t[0] + ratio * (base + (3.f / sqrtf(7.f)) * (j * .4f - 1.f) * root);
                    float perp = sqrtf(.5f) * radii[ray] * cz;
                    float st, ct;
                    sincosf(theta, &st, &ct);
                    float3 point = add(add(add(origin, mul(perp * ct, axis1)), mul(perp * st, axis2)), mul(cz, d));
                    value += lookup(pyramid, point, fabsf(perp), use_mips) * (1.f / 6.f);
                }
            } else {
                float mean = s + (denominator > 0.f ? 2.f * s * h * h / denominator : 0.f);
                value = lookup(pyramid, add(origin, mul(mean, d)), fabsf(sqrtf(.5f) * radii[ray] * mean), use_mips);
            }
            raw[ray * intervals + interval] = raw_space ? value : logf(fmaxf(value, 1.e-30f)) + 1.f;
        }


        void check_input(const torch::Tensor& tensor) {
            if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != torch::kFloat)
                throw std::runtime_error("ZipNeRF: proposal grids require contiguous CUDA float32 tensors");
        }

    }

    torch::Tensor proposal_grid_points_wrapper(const torch::Tensor& reference, const int resolution, const int64_t start, const int count) {
        check_input(reference);
        if (resolution < 2 || resolution > 1024 || (resolution & (resolution - 1)) != 0 || count < 0 || start < 0 ||
            start > static_cast<int64_t>(resolution) * resolution * resolution - count)
            throw std::runtime_error("ZipNeRF: invalid proposal grid point range");
        const c10::cuda::CUDAGuard guard(reference.device());
        auto result = torch::empty({count, 3}, reference.options());
        if (count > 0) {
            proposal_grid_points_kernel<<<(count + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<float3*>(result.data_ptr<float>()), resolution, start, count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return result;
    }

    void sample_proposal_grid_wrapper(
        const std::vector<torch::Tensor>& levels, const torch::Tensor& origins, const torch::Tensor& directions,
        const torch::Tensor& radii, const torch::Tensor& tdist, torch::Tensor& raw_density)
    {
        for (const auto& tensor : {origins, directions, radii, tdist, raw_density}) {
            check_input(tensor);
            if (tensor.device() != origins.device()) throw std::runtime_error("ZipNeRF: proposal grid device mismatch");
        }
        if (origins.dim() != 2 || origins.size(1) != 3 || directions.sizes() != origins.sizes() ||
            tdist.dim() != 2 || tdist.size(0) != origins.size(0) || tdist.size(1) < 2 ||
            radii.numel() != origins.size(0) || raw_density.dim() != 2 || raw_density.size(0) != origins.size(0) ||
            raw_density.size(1) != tdist.size(1) - 1 || raw_density.numel() > INT_MAX - 127 || levels.empty() || levels.size() > 11)
            throw std::runtime_error("ZipNeRF: invalid proposal grid inputs");
        Pyramid pyramid{};
        pyramid.count = static_cast<int>(levels.size());
        for (int i = 0; i < pyramid.count; i++) {
            check_input(levels[i]);
            if (levels[i].device() != origins.device() || levels[i].dim() != 3 || levels[i].size(0) < 1 ||
                levels[i].size(0) > 1024 || levels[i].size(0) != levels[i].size(1) || levels[i].size(0) != levels[i].size(2) ||
                (levels[i].size(0) & (levels[i].size(0) - 1)) != 0 || (i > 0 && pyramid.sizes[i - 1] != 2 * levels[i].size(0)))
                throw std::runtime_error("ZipNeRF: invalid proposal grid mip pyramid");
            pyramid.data[i] = levels[i].data_ptr<float>();
            pyramid.sizes[i] = static_cast<int>(levels[i].size(0));
        }
        const c10::cuda::CUDAGuard guard(origins.device());
        const int rays = static_cast<int>(origins.size(0)), intervals = static_cast<int>(tdist.size(1) - 1);
        if (raw_density.numel() > 0) {
            sample_proposal_grid_kernel<<<(raw_density.numel() + 127) / 128, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
                pyramid, reinterpret_cast<const float3*>(origins.data_ptr<float>()), reinterpret_cast<const float3*>(directions.data_ptr<float>()),
                radii.data_ptr<float>(), tdist.data_ptr<float>(), raw_density.data_ptr<float>(), rays, intervals, true, true, false);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

}
