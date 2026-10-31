// helpers of the runtime-compiled featurization kernels, inlined into their source (no includes, only nvrtc built-ins)
//
// tokens defined by the preamble generated in featurizer.cu:
//   JIT_N_LEVELS, JIT_N_FEATURES: grid levels and features per level (each level adds one scale feature)
//   JIT_MLP_WIDTH: hidden width of the fused density mlp (0 without)
//   JIT_N_FP32_GRAD_LEVELS: leading levels whose gradient featurize_backward accumulates in fp32 (the others in fp16)
//   JIT_PRECONDITION_SCALING: scale of the grid features (see zipnerf_config.h)
//   JIT_PATTERN_STREAM: random stream of the level's multisampling patterns (see random.cuh, prepended by featurizer.cu)
//   zipnerf::config::level_size/level_entries/level_offset/level_hashed: per-level grid layout

namespace zipnerf {

    constexpr uint32_t n_multisamples = 6;
    constexpr uint32_t n_level_features = JIT_N_FEATURES + 1;
    constexpr uint32_t n_output_features = JIT_N_LEVELS * n_level_features;
    constexpr float grid_bbox_size = 4.0f;  // grids cover the contracted space [-2, 2]^3
    constexpr float multisample_scale = 0.5f;
    constexpr uint32_t hash_prime_y = 19349663u;
    constexpr uint32_t hash_prime_z = 83492791u;
    constexpr float tiny = 1.17549435e-38f;
    constexpr float feature_scale = JIT_PRECONDITION_SCALING / n_multisamples;

    // fp16 (grid gradients) and bf16 (NeRF MLP) conversions without cuda_fp16.h and cuda_bf16.h
    __device__ inline uint16_t float_to_half(const float value) {
        uint16_t result;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
        return result;
    }

    __device__ inline uint16_t float_to_bf16(const float value) {
        uint16_t result;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
        return result;
    }

    __device__ inline float bf16_to_float(const uint16_t value) { return __uint_as_float(static_cast<uint32_t>(value) << 16); }

    __device__ inline float3 operator+(const float3& a, const float3& b) { return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ inline float3 operator*(const float a, const float3& b) { return ::make_float3(a * b.x, a * b.y, a * b.z); }
    __device__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ inline float3 cross(const float3& a, const float3& b) { return ::make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

    __device__ inline float3 normalize(const float3& a) { return rsqrtf(dot(a, a)) * a; }

    __device__ inline float approx_erf(const float x) {
        return copysignf(sqrtf(-expm1f(-(4.0f / 3.14159265358979f) * x * x)), x);
    }

    // per-interval setup shared by the six multisamples of the hexagonal pattern
    struct RayInterval {
        float3 origin, direction, perp_axis1, perp_axis2;
        float radius, t0, ratio, base, root, rotation;
        bool flip;
    };

    __device__ inline RayInterval setup_interval(
        const float3* __restrict__ origins,
        const float3* __restrict__ directions,
        const float* __restrict__ radii,
        const float* __restrict__ tdist,
        const long long pattern_seed,
        const uint32_t sample_idx,
        const uint32_t n_intervals)
    {
        RayInterval interval;
        const uint32_t ray_idx = sample_idx / n_intervals;
        interval.origin = origins[ray_idx];
        interval.direction = directions[ray_idx];
        interval.radius = radii[ray_idx];
        // perpendicular basis
        const float3 d = interval.direction;
        float3 cross1 = cross(d, ::make_float3(0.0f, 0.0f, 1.0f));
        constexpr float eps = 1.1920929e-07f;
        if (fabsf(cross1.x) < eps && fabsf(cross1.y) < eps && fabsf(cross1.z) < eps) cross1 = cross(d, ::make_float3(1.0f, 1.0f, 1.0f));
        interval.perp_axis1 = normalize(cross1);
        interval.perp_axis2 = normalize(cross(d, cross1));
        // conical frustum moments
        const float* t = tdist + ray_idx * (n_intervals + 1) + sample_idx % n_intervals;
        const float t0 = t[0], t1 = t[1];
        const float s = 0.5f * (t0 + t1);
        const float h = 0.5f * (t1 - t0);
        const float denominator = h * h + 3.0f * s * s;
        interval.t0 = t0;
        interval.ratio = fabsf(denominator) < tiny ? 0.0f : h / denominator;
        interval.base = t1 * t1 + 2.0f * s * s;
        const float q = h * h - s * s;
        interval.root = sqrtf(fmaxf(q * q + 4.0f * s * s * s * s, 0.0f));
        // random pattern flip and rotation (pattern_seed >= 0), alternating flips with a pi/6 rotation otherwise
        if (pattern_seed >= 0) {
            interval.flip = random::uniform(pattern_seed, JIT_PATTERN_STREAM, 2ull * sample_idx) < 0.5f;
            interval.rotation = 2.0f * 3.14159265358979f * random::uniform(pattern_seed, JIT_PATTERN_STREAM, 2ull * sample_idx + 1);
        }
        else {
            interval.flip = (sample_idx % n_intervals) % 2 == 1;
            interval.rotation = interval.flip ? 3.14159265358979f / 6.0f : 0.0f;
        }
        return interval;
    }

    // contracted multisample position mapped to [0, 1]^3 and its radius in the same units
    __device__ inline void multisample(const RayInterval& interval, const uint32_t j, float3& x, float& scale) {
        // pattern angles (pi / 3) * (0, 2, 4, 3, 5, 1), reversed when flipped
        constexpr float pattern[n_multisamples] = {0.0f, 2.0f, 4.0f, 3.0f, 5.0f, 1.0f};
        const float theta = (3.14159265358979f / 3.0f) * pattern[interval.flip ? n_multisamples - 1 - j : j] + interval.rotation;
        const float cz = interval.t0 + interval.ratio * (interval.base + (3.0f / sqrtf(7.0f)) * (j * 0.4f - 1.0f) * interval.root);
        const float perp_mag = sqrtf(0.5f) * interval.radius * cz;
        float sin_theta, cos_theta;
        sincosf(theta, &sin_theta, &cos_theta);
        const float3 p = interval.origin + (perp_mag * cos_theta) * interval.perp_axis1 + (perp_mag * sin_theta) * interval.perp_axis2 + cz * interval.direction;
        // contraction and isotropic scaling of its jacobian
        const float norm_sq = fmaxf(dot(p, p), 1.0f);
        const float norm = sqrtf(norm_sq);
        const float isoscale = expf((2.0f / 3.0f) * logf(2.0f * norm - 1.0f) - logf(norm_sq));
        const float contraction = (2.0f * norm - 1.0f) / norm_sq;
        x = (1.0f / grid_bbox_size) * (contraction * p) + ::make_float3(0.5f, 0.5f, 0.5f);
        scale = multisample_scale * perp_mag * isoscale * (1.0f / grid_bbox_size);
    }

    constexpr uint32_t max_level_corners = 8 * n_multisamples;

    // table rows and weights (including the downweighting) of the corners of one level for all multisamples of an
    // interval; multisamples in the same cell share corners, whose weights are merged to save lookups and atomics;
    // returns the number of corners
    __device__ inline uint32_t level_corners(const uint32_t level, const float3* x, const float* weighting, uint32_t* rows, float* weights) {
        const uint32_t size = config::level_size(level);
        int cells[n_multisamples][3];
        float fractions[n_multisamples][3];
        #pragma unroll
        for (uint32_t j = 0; j < n_multisamples; j++) {
            const float p[3] = {x[j].x * size - 0.5f, x[j].y * size - 0.5f, x[j].z * size - 0.5f};
            #pragma unroll
            for (uint32_t d = 0; d < 3; d++) {
                const float floored = floorf(p[d]);
                cells[j][d] = static_cast<int>(floored);
                fractions[j][d] = p[d] - floored;
            }
        }
        uint32_t n_corners = 0, merged = 0;
        #pragma unroll
        for (uint32_t j = 0; j < n_multisamples; j++) {
            if (merged & (1u << j)) continue;
            float corner_weights[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (uint32_t k = j; k < n_multisamples; k++) {
                if (cells[k][0] != cells[j][0] || cells[k][1] != cells[j][1] || cells[k][2] != cells[j][2]) continue;
                merged |= 1u << k;
                #pragma unroll
                for (uint32_t corner = 0; corner < 8; corner++) {
                    const float wx = corner & 4u ? fractions[k][0] : 1.0f - fractions[k][0];
                    const float wy = corner & 2u ? fractions[k][1] : 1.0f - fractions[k][1];
                    const float wz = corner & 1u ? fractions[k][2] : 1.0f - fractions[k][2];
                    corner_weights[corner] += weighting[k] * (wx * wy * wz);
                }
            }
            #pragma unroll
            for (uint32_t corner = 0; corner < 8; corner++) {
                const int ix = cells[j][0] + static_cast<int>((corner >> 2) & 1u);
                const int iy = cells[j][1] + static_cast<int>((corner >> 1) & 1u);
                const int iz = cells[j][2] + static_cast<int>(corner & 1u);
                uint32_t index;
                float weight = corner_weights[corner];
                if (config::level_hashed(level)) {
                    index = (static_cast<uint32_t>(ix) ^ (static_cast<uint32_t>(iy) * hash_prime_y) ^ (static_cast<uint32_t>(iz) * hash_prime_z)) % config::level_entries(level);
                }
                else {
                    const int n = static_cast<int>(size);
                    const bool valid = ix >= 0 && ix < n && iy >= 0 && iy < n && iz >= 0 && iz < n;
                    if (!valid) weight = 0.0f;
                    index = valid ? static_cast<uint32_t>(ix + n * (iy + n * iz)) : 0u;
                }
                rows[n_corners] = config::level_offset(level) + index;
                weights[n_corners] = weight;
                n_corners++;
            }
        }
        return n_corners;
    }

    // downweighting of a level's features by the multisample's radius
    __device__ inline float level_weighting(const uint32_t level, const float scale) {
        return approx_erf(1.0f / (sqrtf(8.0f) * scale * config::level_size(level)));
    }

    // downweighted grid features of an interval, averaged over the multisamples and preconditioned
    __device__ inline void featurize(const RayInterval& interval, const float* __restrict__ table, const float* __restrict__ scale_magnitudes, float* features) {
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
            uint32_t rows[max_level_corners];
            float weights[max_level_corners];
            const uint32_t n_corners = level_corners(level, x, weighting, rows, weights);
            float level_features[JIT_N_FEATURES];
            #pragma unroll
            for (uint32_t f = 0; f < JIT_N_FEATURES; f++) level_features[f] = 0.0f;
            for (uint32_t c = 0; c < n_corners; c++) {
                if (weights[c] == 0.0f) continue;
                // vector loads for rows of 2 or 4 features (8 / 16 byte aligned)
                if (JIT_N_FEATURES == 4) {
                    const float4 value = reinterpret_cast<const float4*>(table)[rows[c]];
                    level_features[0] += weights[c] * value.x;
                    level_features[1 % JIT_N_FEATURES] += weights[c] * value.y;
                    level_features[2 % JIT_N_FEATURES] += weights[c] * value.z;
                    level_features[3 % JIT_N_FEATURES] += weights[c] * value.w;
                }
                else if (JIT_N_FEATURES == 2) {
                    const float2 value = reinterpret_cast<const float2*>(table)[rows[c]];
                    level_features[0] += weights[c] * value.x;
                    level_features[1 % JIT_N_FEATURES] += weights[c] * value.y;
                }
                else {
                    #pragma unroll
                    for (uint32_t f = 0; f < JIT_N_FEATURES; f++) level_features[f] += weights[c] * table[rows[c] * JIT_N_FEATURES + f];
                }
            }
            #pragma unroll
            for (uint32_t f = 0; f < JIT_N_FEATURES; f++) features[level * n_level_features + f] = feature_scale * level_features[f];
            features[level * n_level_features + JIT_N_FEATURES] = feature_scale * scale_feature * scale_magnitudes[level];
        }
    }

    // adds the weighted feature gradient to a row of an fp32 grid gradient
    __device__ inline void add_row_gradient(float* __restrict__ grad_table, const uint32_t row, const float weight, const float* g) {
        float* grad_row = grad_table + row * JIT_N_FEATURES;
#if __CUDA_ARCH__ >= 900 && JIT_N_FEATURES == 4
        // vector atomics for rows of 4 features (16 byte aligned)
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(grad_row),
            "f"(weight * g[0]), "f"(weight * g[1]), "f"(weight * g[2]), "f"(weight * g[3]) : "memory");
#else
        #pragma unroll
        for (uint32_t f = 0; f < JIT_N_FEATURES; f++) {
            if (g[f] != 0.0f) atomicAdd(grad_row + f, weight * g[f]);
        }
#endif
    }

    // adds the weighted feature gradient to a row of an fp16 grid gradient: one 8 byte vector atomic for rows of 4
    // features (sm_90+), else packed half2 atomics (an atomic costs one L2 request regardless of its width)
    __device__ inline void add_row_gradient(uint16_t* __restrict__ grad_table, const uint32_t row, const float weight, const float* g) {
#if JIT_N_FEATURES % 2 == 0
        uint32_t* grad_row = reinterpret_cast<uint32_t*>(grad_table + row * JIT_N_FEATURES);
        uint32_t pairs[JIT_N_FEATURES / 2];
        #pragma unroll
        for (uint32_t p = 0; p < JIT_N_FEATURES / 2; p++) {
            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(pairs[p]) : "f"(weight * g[2 * p + 1]), "f"(weight * g[2 * p]));
        }
#if __CUDA_ARCH__ >= 900 && JIT_N_FEATURES == 4
        asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1, %2};" :: "l"(grad_row), "r"(pairs[0]), "r"(pairs[1]) : "memory");
#else
        #pragma unroll
        for (uint32_t p = 0; p < JIT_N_FEATURES / 2; p++) {
            asm volatile("red.global.add.noftz.f16x2 [%0], %1;" :: "l"(grad_row + p), "r"(pairs[p]) : "memory");
        }
#endif
#else
        #pragma unroll
        for (uint32_t f = 0; f < JIT_N_FEATURES; f++) {
            if (g[f] != 0.0f) asm volatile("red.global.add.noftz.f16 [%0], %1;" :: "l"(grad_table + row * JIT_N_FEATURES + f), "h"(float_to_half(weight * g[f])) : "memory");
        }
#endif
    }

    // accumulates the grid gradient of an interval's feature gradient (scale features carry no gradient): the first
    // n_fp32_levels levels into grad_table (fp32), the others into grad_table_half (fp16, rows from the first fp16 level)
    template <uint32_t n_fp32_levels>
    __device__ inline void scatter_feature_gradients(const RayInterval& interval, const float* grad_features, float* __restrict__ grad_table, uint16_t* __restrict__ grad_table_half) {
        float3 x[n_multisamples];
        float scale[n_multisamples];
        #pragma unroll
        for (uint32_t j = 0; j < n_multisamples; j++) multisample(interval, j, x[j], scale[j]);
        #pragma unroll
        for (uint32_t level = 0; level < JIT_N_LEVELS; level++) {
            float weighting[n_multisamples];
            #pragma unroll
            for (uint32_t j = 0; j < n_multisamples; j++) weighting[j] = level_weighting(level, scale[j]);
            uint32_t rows[max_level_corners];
            float weights[max_level_corners];
            const uint32_t n_corners = level_corners(level, x, weighting, rows, weights);
            const float* g = grad_features + level * n_level_features;
            for (uint32_t c = 0; c < n_corners; c++) {
                if (weights[c] == 0.0f) continue;
                if (level < n_fp32_levels) add_row_gradient(grad_table, rows[c], weights[c], g);
                else add_row_gradient(grad_table_half, rows[c] - config::level_offset(n_fp32_levels), weights[c], g);
            }
        }
    }

}
