#pragma once

#include <cstdint>
#include <cfloat>

// per-ray device functions of the sampling kernels (one thread per ray), following stepfun.py and render.py of camp_zipnerf

namespace zipnerf::sampling {

    inline constexpr int max_intervals = 64;
    inline constexpr int max_proposal_levels = 4;
    inline constexpr float tiny = FLT_MIN;
    inline constexpr float eps = FLT_EPSILON;
    inline constexpr float max_density_exponent = 88.72283f;  // exp clipped to finite outputs

    struct RayWarp {
        float near, far, fn_near, fn_far, power, premult, y_max;
    };

    __device__ inline float power_ladder(const float x, const float p, const float premult) {
        const float xp = fabsf(x * premult);
        const float y = fabsf(p - 1.0f) / p * (powf(xp / fabsf(p - 1.0f) + 1.0f, p) - 1.0f);
        return copysignf(y, x);
    }

    // s in [0, 1] -> metric distance via the inverse power ladder
    __device__ inline float s_to_t(const float s, const RayWarp& warp) {
        const float y = s * warp.fn_far + (1.0f - s) * warp.fn_near;
        const float yp = warp.power < 0.0f ? fminf(fabsf(y), warp.y_max) : fabsf(y);
        const float p = warp.power;
        const float x = copysignf(fabsf(p - 1.0f) * (powf(p / fabsf(p - 1.0f) * yp + 1.0f, 1.0f / p) - 1.0f), y) / warp.premult;
        return fminf(fmaxf(x, warp.near), warp.far);
    }

    __device__ inline float activate_density(const float raw) {
        return expf(fminf(raw - 1.0f, max_density_exponent));
    }

    // samples n_out intervals from the step function (t, softmax(anneal * log(w))) of n bins; jitter < 0 selects the
    // deterministic centered pattern
    __device__ inline void sample_intervals(const float* t, const float* w, const int n, const float anneal, const int n_out, const float jitter, float* s_out) {
        // resampling weights (softmax of the annealed log weights, empty intervals get zero weight)
        float p[max_intervals];
        float max_logit = -INFINITY;
        for (int i = 0; i < n; i++) {
            p[i] = t[i + 1] > t[i] ? anneal * logf(fmaxf(w[i], tiny)) : -INFINITY;
            max_logit = fmaxf(max_logit, p[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            p[i] = p[i] == -INFINITY ? 0.0f : expf(p[i] - max_logit);
            sum += p[i];
        }
        // cdf with explicit 0 and 1 endpoints
        float cw[max_intervals + 1];
        cw[0] = 0.0f;
        float acc = 0.0f;
        for (int i = 1; i < n; i++) {
            acc += p[i - 1] / sum;
            cw[i] = fminf(acc, 1.0f);
        }
        cw[n] = 1.0f;
        // stratified (single jitter per ray) or centered sample positions in the cdf
        float u_start, u_step;
        if (jitter >= 0.0f) {
            const float u_max = eps + (1.0f - eps) / n_out;
            u_start = jitter * ((1.0f - u_max) / (n_out - 1) - eps);
            u_step = (1.0f - u_max) / (n_out - 1);
        }
        else {
            const float pad = 1.0f / (2 * n_out);
            u_start = pad;
            u_step = (1.0f - 2.0f * pad - eps) / (n_out - 1);
        }
        // inverse cdf at the sorted positions, then interval endpoints from the sample centers
        float centers[max_intervals];
        int idx = 0;  // number of cdf values <= u
        for (int j = 0; j < n_out; j++) {
            const float u = u_start + j * u_step;
            while (idx <= n && cw[idx] <= u) idx++;
            const int i0 = max(idx - 1, 0), i1 = min(idx, n);
            const float offset = fminf(fmaxf((u - cw[i0]) / fmaxf(eps * eps, cw[i1] - cw[i0]), 0.0f), 1.0f);
            centers[j] = t[i0] + offset * (t[i1] - t[i0]);
        }
        for (int j = 1; j < n_out; j++) s_out[j] = fminf(fmaxf(0.5f * (centers[j - 1] + centers[j]), 0.0f), 1.0f);
        s_out[0] = fminf(fmaxf(2.0f * centers[0] - 0.5f * (centers[0] + centers[1]), 0.0f), 1.0f);
        s_out[n_out] = fminf(fmaxf(2.0f * centers[n_out - 1] - 0.5f * (centers[n_out - 2] + centers[n_out - 1]), 0.0f), 1.0f);
    }

    // alpha compositing weights of n intervals; x_out receives the optical depths sigma * dt * |d|
    __device__ inline void composite(const float* density, const float* t, const int n, const float direction_norm, float* w_out, float* x_out) {
        float optical_depth = 0.0f;
        for (int i = 0; i < n; i++) {
            const float x = density[i] * (t[i + 1] - t[i]) * direction_norm;
            w_out[i] = -expm1f(-x) * expf(-optical_depth);
            optical_depth += x;
            if (x_out != nullptr) x_out[i] = x;
        }
    }

    // gradient of the weights w.r.t. the optical depths (dL/dx_k = g_k T_k exp(-x_k) - sum_{i>k} g_i w_i), in place
    __device__ inline void composite_backward(float* grad, const float* x, const float* w, const int n) {
        float transmittance[max_intervals];  // T_k exp(-x_k) = exp(-sum_{j<=k} x_j)
        float optical_depth = 0.0f;
        for (int i = 0; i < n; i++) {
            optical_depth += x[i];
            transmittance[i] = expf(-optical_depth);
        }
        float suffix = 0.0f;
        for (int i = n - 1; i >= 0; i--) {
            const float g = grad[i];
            grad[i] = g * transmittance[i] - suffix;
            suffix += g * w[i];
        }
    }

    __device__ inline float plus_eps(const float x) { return fabsf(x) < tiny ? tiny : nextafterf(x, INFINITY); }
    __device__ inline float minus_eps(const float x) { return fabsf(x) < tiny ? -tiny : nextafterf(x, -INFINITY); }

    // blurs the histogram (t, w) of n bins with a box filter of the given halfwidth and resamples it into the nq - 1 bins
    // of the sorted endpoints tq
    __device__ inline void blur_and_resample(const float* tq, const int nq, const float* t, const float* w, const int n, const float halfwidth, float* wq_out) {
        // lower and upper box edges of each endpoint and the pdf slope change at both
        float edge_lo[max_intervals + 1], edge_hi[max_intervals + 1], slope_change[max_intervals + 1];
        float pdf_previous = 0.0f;
        for (int i = 0; i <= n; i++) {
            const float dt = i < n ? t[i + 1] - t[i] : 0.0f;
            const float pdf = i < n && dt >= tiny ? w[i] / dt : 0.0f;
            edge_lo[i] = t[i] - halfwidth;
            edge_hi[i] = fmaxf(plus_eps(t[i]), t[i] + halfwidth);
            slope_change[i] = (pdf - pdf_previous) / (edge_hi[i] - edge_lo[i]);
            pdf_previous = pdf;
        }
        // piecewise linear spline (tp, yp) of the blurred pdf by merging the sorted edges; the slope changes of the
        // last two knots are not needed since the spline ends at zero
        constexpr int max_knots = 2 * (max_intervals + 1);
        float tp[max_knots], yp[max_knots];
        const int n_knots = 2 * (n + 1);
        int i_lo = 0, i_hi = 0;
        float slope = 0.0f;
        for (int k = 0; k < n_knots; k++) {
            const bool take_lo = i_hi > n || (i_lo <= n && edge_lo[i_lo] <= edge_hi[i_hi]);
            const float knot = take_lo ? edge_lo[i_lo] : edge_hi[i_hi];
            yp[k] = k == 0 || k == n_knots - 1 ? 0.0f : yp[k - 1] + (knot - tp[k - 1]) * slope;
            tp[k] = knot;
            if (k < n_knots - 2) slope += take_lo ? slope_change[i_lo] : -slope_change[i_hi];
            if (take_lo) i_lo++;
            else i_hi++;
        }
        // integral of the spline (piecewise quadratic) at the sorted query points
        const float t_min = tp[0], t_max = minus_eps(tp[n_knots - 1]);
        int k = 0;
        float integral = 0.0f, previous = 0.0f;  // integral up to tp[k]
        for (int j = 0; j < nq; j++) {
            const float q = fminf(fmaxf(tq[j], t_min), t_max);
            while (k + 2 < n_knots && tp[k + 1] <= q) {
                integral += 0.5f * (tp[k + 1] - tp[k]) * (yp[k] + yp[k + 1]);
                k++;
            }
            const float a = (yp[k + 1] - yp[k]) / fmaxf(eps * eps, 2.0f * (tp[k + 1] - tp[k]));
            const float td = q - tp[k];
            const float value = (a * td + yp[k]) * td + integral;
            if (j > 0) wq_out[j - 1] = fmaxf(value - previous, 0.0f);
            previous = value;
        }
    }

}
