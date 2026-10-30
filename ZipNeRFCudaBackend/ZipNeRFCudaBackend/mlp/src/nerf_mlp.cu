#include "nerf_mlp.h"
#include "zipnerf_config.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace zipnerf::mlp {

    constexpr int64_t workspace_bytes = 32 * 1024 * 1024;

    static void check_cublas(const cublasStatus_t status, const char* call) {
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("ZipNeRF: ") + call + " failed with status " + std::to_string(static_cast<int>(status)));
    }

    // a column-major operand: data with rows x cols and leading dimension ld (row-major tensors are their transposes)
    struct Operand {
        const void* data;
        int64_t rows, cols, ld;
        cudaDataType_t type;
    };

    struct Epilogue {
        cublasLtEpilogue_t type = CUBLASLT_EPILOGUE_DEFAULT;
        const void* bias = nullptr;  // fp16 bias, or fp16 bias gradient output for bgrad epilogues (cuBLASLt requires the output type)
        void* aux = nullptr;  // relu bitmask
        int64_t aux_ld = 0;
    };

    // D = op(A) op(B) + beta * D with an epilogue (fp32 compute and scale type)
    static void matmul(const Operand& a, const bool transpose_a, const Operand& b, const bool transpose_b, void* d, const int64_t ldd, const cudaDataType_t d_type, const float beta, const Epilogue& epilogue = {}) {
        const int64_t m = transpose_a ? a.cols : a.rows;
        const int64_t k = transpose_a ? a.rows : a.cols;
        const int64_t n = transpose_b ? b.rows : b.cols;
        if ((transpose_b ? b.cols : b.rows) != k) throw std::runtime_error("ZipNeRF: matmul shape mismatch");
        if (m == 0 || n == 0) return;
        cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
        cublasLtMatmulDesc_t description;
        check_cublas(cublasLtMatmulDescCreate(&description, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
        const cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N, op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
        cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
        if (epilogue.type != CUBLASLT_EPILOGUE_DEFAULT) {
            check_cublas(cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue.type, sizeof(epilogue.type)), "set epilogue");
            if (epilogue.bias != nullptr) {
                check_cublas(cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &epilogue.bias, sizeof(epilogue.bias)), "set bias");
            }
            if (epilogue.aux != nullptr) {
                check_cublas(cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &epilogue.aux, sizeof(epilogue.aux)), "set aux");
                check_cublas(cublasLtMatmulDescSetAttribute(description, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &epilogue.aux_ld, sizeof(epilogue.aux_ld)), "set aux ld");
            }
        }
        cublasLtMatrixLayout_t layout_a, layout_b, layout_d;
        cublasLtMatrixLayoutCreate(&layout_a, a.type, a.rows, a.cols, a.ld);
        cublasLtMatrixLayoutCreate(&layout_b, b.type, b.rows, b.cols, b.ld);
        cublasLtMatrixLayoutCreate(&layout_d, d_type, m, n, ldd);
        // the fastest of the heuristic candidates is chosen by timing them once per problem (not for accumulating GEMMs,
        // whose output the timing runs would change) and cached
        static std::unordered_map<std::string, cublasLtMatmulAlgo_t> algorithms;
        const std::string key = std::to_string(m) + "," + std::to_string(n) + "," + std::to_string(k) + "," + std::to_string(a.ld) + "," + std::to_string(b.ld) + ","
            + std::to_string(ldd) + "," + std::to_string(transpose_a) + std::to_string(transpose_b) + "," + std::to_string(epilogue.type) + "," + std::to_string(d_type) + ","
            + std::to_string(beta != 0.0f) + "," + std::to_string(epilogue.aux_ld);
        const torch::Tensor workspace = torch::empty({workspace_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        const float alpha = 1.0f;
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        auto entry = algorithms.find(key);
        if (entry == algorithms.end()) {
            cublasLtMatmulPreference_t preference;
            cublasLtMatmulPreferenceCreate(&preference);
            const uint64_t max_workspace = workspace_bytes;
            cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace));
            constexpr int max_candidates = 8;
            cublasLtMatmulHeuristicResult_t results[max_candidates];
            int n_results = 0;
            const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle, description, layout_a, layout_b, layout_d, layout_d, preference, beta != 0.0f ? 1 : max_candidates, results, &n_results);
            cublasLtMatmulPreferenceDestroy(preference);
            if (status != CUBLAS_STATUS_SUCCESS || n_results == 0) throw std::runtime_error("ZipNeRF: no cublasLt algorithm for " + key);
            int best = 0;
            if (n_results > 1) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                float best_time = INFINITY;
                for (int i = 0; i < n_results; i++) {
                    if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
                    cudaEventRecord(start, stream);
                    for (int repeat = 0; repeat < 3; repeat++) {
                        cublasLtMatmul(handle, description, &alpha, a.data, layout_a, b.data, layout_b, &beta, d, layout_d, d, layout_d,
                            &results[i].algo, workspace.data_ptr(), workspace_bytes, stream);
                    }
                    cudaEventRecord(stop, stream);
                    cudaEventSynchronize(stop);
                    float time;
                    cudaEventElapsedTime(&time, start, stop);
                    if (time < best_time) {
                        best_time = time;
                        best = i;
                    }
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            if (algorithms.size() > 4096) algorithms.clear();
            entry = algorithms.emplace(key, results[best].algo).first;
        }
        check_cublas(cublasLtMatmul(handle, description, &alpha, a.data, layout_a, b.data, layout_b, &beta, d, layout_d, d, layout_d,
            &entry->second, workspace.data_ptr(), workspace_bytes, stream), "cublasLtMatmul");
        cublasLtMatrixLayoutDestroy(layout_a);
        cublasLtMatrixLayoutDestroy(layout_b);
        cublasLtMatrixLayoutDestroy(layout_d);
        cublasLtMatmulDescDestroy(description);
    }

    constexpr int block_size = 256;
    constexpr int n_outputs = 8;  // raw density, rgb logits, zero padding
    constexpr int64_t segment_alignment = 128;  // elements, so that all GEMM operands start at 256-byte boundaries

    static int64_t round_up(const int64_t value, const int64_t multiple) { return (value + multiple - 1) / multiple * multiple; }

    // output and input sizes of the packed weights (biases have a single input)
    __host__ __device__ inline int64_t weight_rows(const int weight, const Layout& layout) {
        switch (weight) {
            case w_density: case b_density: return layout.density_width;
            case w_combined: case b_1: case w_1: case w_2: case b_2: return layout.view_width;
            case w_skip: case w_rgb: return n_outputs;
            default: return 0;
        }
    }

    __host__ __device__ inline int64_t weight_cols(const int weight, const Layout& layout) {
        switch (weight) {
            case w_density: return layout.n_features;
            case w_combined: case w_skip: return layout.n_inputs;
            case w_1: case w_2: case w_rgb: return layout.view_width;
            default: return 1;
        }
    }

    // segment of a flat index in an offset table (segments are visited in order by consecutive threads)
    __device__ inline int find_segment(const int64_t* offsets, const int n_segments, const int64_t idx) {
        int segment = 0;
        while (segment < n_segments - 1 && idx >= offsets[segment + 1]) segment++;
        return segment;
    }

    // a column of the bottleneck folded into a row of weights over the bottleneck outputs (the bias column folds the
    // bottleneck bias)
    __device__ inline float fold_bottleneck(
        const float* __restrict__ row, const float* __restrict__ bottleneck, const float* __restrict__ bottleneck_bias, const int64_t column,
        const Layout& layout)
    {
        const bool bias = column == layout.n_inputs - 1;
        float sum = 0.0f;
        for (int64_t k = 0; k < layout.bottleneck_width; k++) sum += row[k] * (bias ? bottleneck_bias[k] : bottleneck[k * layout.density_width + column]);
        return sum;
    }

    // gradient of a bottleneck input weight (or of the bias for the bias column) from the gradients of the folded
    // weights of the first view layer and the rgb skip rows (rows 1-3 of w_skip)
    __device__ inline float unfold_bottleneck(
        const float* __restrict__ view_0, const float* __restrict__ rgb, const float* __restrict__ grad_combined, const float* __restrict__ grad_skip,
        const int64_t k, const int64_t column, const Layout& layout)
    {
        float sum = 0.0f;
        for (int64_t o = 0; o < layout.view_width; o++) sum += view_0[o * layout.n_view_inputs + k] * grad_combined[o * layout.n_inputs + column];
        for (int r = 0; r < 3; r++) sum += rgb[r * layout.n_rgb_inputs + layout.view_width + k] * grad_skip[(r + 1) * layout.n_inputs + column];
        return sum;
    }

    // gradient of a weight over the bottleneck outputs from the gradient row of its folded weights
    __device__ inline float unfold_row(
        const float* __restrict__ grad_row, const float* __restrict__ bottleneck, const float* __restrict__ bottleneck_bias, const int64_t k,
        const Layout& layout)
    {
        float sum = grad_row[layout.n_inputs - 1] * bottleneck_bias[k];
        for (int64_t j = 0; j < layout.density_width; j++) sum += grad_row[j] * bottleneck[k * layout.density_width + j];
        return sum;
    }

    __global__ void pack_kernel(const float* __restrict__ params, const Layout layout, __half* __restrict__ packed) {
        const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= layout.weights[n_weights]) return;
        const int weight = find_segment(layout.weights, n_weights, idx);
        const int64_t element = idx - layout.weights[weight], cols = weight_cols(weight, layout);
        // alignment padding between the segments
        if (element >= weight_rows(weight, layout) * cols) {
            packed[idx] = __float2half(0.0f);
            return;
        }
        const int64_t row = element / cols, column = element % cols;
        const float* bottleneck = params + layout.params[bottleneck_weight];
        const float* bottleneck_b = params + layout.params[bottleneck_bias];
        const bool bias = column == layout.n_inputs - 1;
        const bool direction = column >= layout.density_width && column < layout.density_width + layout.n_direction_features;
        const int64_t direction_column = layout.bottleneck_width + column - layout.density_width;
        float value = 0.0f;
        switch (weight) {
            case w_density: value = params[layout.params[density_0_weight] + element]; break;
            case b_density: value = params[layout.params[density_0_bias] + element]; break;
            case w_combined: {
                const float* view_0 = params + layout.params[view_0_weight] + row * layout.n_view_inputs;
                if (column < layout.density_width || bias) value = fold_bottleneck(view_0, bottleneck, bottleneck_b, column, layout);
                if (bias) value += params[layout.params[view_0_bias] + row];
                if (direction) value = view_0[direction_column];
                break;
            }
            case w_skip: {
                if (row == 0) {
                    if (column < layout.density_width) value = params[layout.params[density_1_weight] + column];
                    if (bias) value = params[layout.params[density_1_bias]];
                }
                else if (row < 4) {
                    const float* rgb = params + layout.params[rgb_weight] + (row - 1) * layout.n_rgb_inputs + layout.view_width;
                    if (column < layout.density_width || bias) value = fold_bottleneck(rgb, bottleneck, bottleneck_b, column, layout);
                    if (bias) value += params[layout.params[rgb_bias] + row - 1];
                    if (direction) value = rgb[direction_column];
                }
                break;
            }
            case w_1: value = params[layout.params[view_1_weight] + element]; break;
            case b_1: value = params[layout.params[view_1_bias] + element]; break;
            case w_2: value = params[layout.params[view_2_weight] + element]; break;
            case b_2: value = params[layout.params[view_2_bias] + element]; break;
            case w_rgb: if (row >= 1 && row < 4) value = params[layout.params[rgb_weight] + (row - 1) * layout.n_rgb_inputs + column]; break;
            default: __trap();
        }
        packed[idx] = __float2half(value);
    }

    __global__ void unpack_kernel(
        const float* __restrict__ params,
        const float* __restrict__ weight_grads,
        const __half* __restrict__ bias_grads,
        const Layout layout,
        float* __restrict__ grad_params)
    {
        const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= layout.params[n_parameters]) return;
        const int parameter = find_segment(layout.params, n_parameters, idx);
        const int64_t element = idx - layout.params[parameter];
        const int64_t bias_column = layout.n_inputs - 1;
        const float* bottleneck = params + layout.params[bottleneck_weight];
        const float* bottleneck_b = params + layout.params[bottleneck_bias];
        const float* view_0 = params + layout.params[view_0_weight];
        const float* rgb = params + layout.params[rgb_weight];
        const float* grad_combined = weight_grads + layout.weights[w_combined];
        const float* grad_skip = weight_grads + layout.weights[w_skip];
        float grad = 0.0f;
        switch (parameter) {
            case density_0_weight: grad = weight_grads[layout.weights[w_density] + element]; break;
            case density_0_bias: grad = __half2float(bias_grads[layout.bias_grads[b_density] + element]); break;
            case density_1_weight: grad = grad_skip[element]; break;
            case density_1_bias: grad = grad_skip[bias_column]; break;
            case bottleneck_weight: grad = unfold_bottleneck(view_0, rgb, grad_combined, grad_skip, element / layout.density_width, element % layout.density_width, layout); break;
            case bottleneck_bias: grad = unfold_bottleneck(view_0, rgb, grad_combined, grad_skip, element, bias_column, layout); break;
            case view_0_weight: {
                const int64_t row = element / layout.n_view_inputs, column = element % layout.n_view_inputs;
                if (column < layout.bottleneck_width) grad = unfold_row(grad_combined + row * layout.n_inputs, bottleneck, bottleneck_b, column, layout);
                else if (column < layout.bottleneck_width + layout.n_direction_features) grad = grad_combined[row * layout.n_inputs + layout.density_width + column - layout.bottleneck_width];
                break;
            }
            case view_0_bias: grad = grad_combined[element * layout.n_inputs + bias_column]; break;
            case view_1_weight: grad = weight_grads[layout.weights[w_1] + element]; break;
            case view_1_bias: grad = __half2float(bias_grads[layout.bias_grads[b_1] + element]); break;
            case view_2_weight: grad = weight_grads[layout.weights[w_2] + element]; break;
            case view_2_bias: grad = __half2float(bias_grads[layout.bias_grads[b_2] + element]); break;
            case rgb_weight: {
                const int64_t row = element / layout.n_rgb_inputs, column = element % layout.n_rgb_inputs;
                const float* grad_skip_row = grad_skip + (row + 1) * layout.n_inputs;
                if (column < layout.view_width) grad = weight_grads[layout.weights[w_rgb] + (row + 1) * layout.view_width + column];
                else if (column < layout.view_width + layout.bottleneck_width) grad = unfold_row(grad_skip_row, bottleneck, bottleneck_b, column - layout.view_width, layout);
                else if (column < layout.view_width + layout.bottleneck_width + layout.n_direction_features) grad = grad_skip_row[layout.density_width + column - layout.view_width - layout.bottleneck_width];
                break;
            }
            case rgb_bias: grad = grad_skip[(element + 1) * layout.n_inputs + bias_column]; break;
            default: __trap();
        }
        grad_params[idx] += grad;
    }
    static void check_input(const torch::Tensor& tensor, const char* name, const torch::ScalarType dtype) {
        if (!tensor.is_cuda() || !tensor.is_contiguous() || tensor.scalar_type() != dtype) {
            throw std::runtime_error(std::string("ZipNeRF: ") + name + " must be a contiguous cuda tensor of the expected type");
        }
    }

    // a row-major (rows x cols, row stride) tensor seen as its column-major transpose
    static Operand operand(const torch::Tensor& tensor) {
        if (tensor.dim() != 2 || tensor.stride(1) != 1) throw std::runtime_error("ZipNeRF: matmul operands must be row-major 2D tensors");
        return {tensor.data_ptr(), tensor.size(1), tensor.size(0), tensor.stride(0), tensor.scalar_type() == torch::kHalf ? CUDA_R_16F : CUDA_R_32F};
    }

    // relu bitmask for rows x n (column-major), leading dimension rounded up to 128 bits
    static torch::Tensor relu_mask(const int64_t rows, const int64_t n) {
        return torch::empty({n, (rows + 127) / 128 * 16}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }

    static int64_t mask_ld(const int64_t rows) { return (rows + 127) / 128 * 128; }

    static unsigned int n_blocks(const int64_t n) { return static_cast<unsigned int>((n + block_size - 1) / block_size); }

    std::vector<std::tuple<std::string, std::vector<int64_t>, int64_t>> nerf_mlp_parameters(
        const int64_t n_features, const int64_t density_width, const int64_t bottleneck_width, const int64_t view_width, const int64_t direction_degree)
    {
        // input columns are zero-padded to multiples of 8; the padding columns come last and never receive gradients
        const int64_t n_direction_features = 3 + 6 * direction_degree;
        const int64_t n_view_inputs = bottleneck_width + config::direction_block_width(direction_degree);
        return {
            {"density_0.weight", {density_width, round_up(n_features, 8)}, n_features}, {"density_0.bias", {density_width}, 0},
            {"density_1.weight", {1, density_width}, density_width}, {"density_1.bias", {1}, 0},
            {"bottleneck.weight", {bottleneck_width, density_width}, density_width}, {"bottleneck.bias", {bottleneck_width}, 0},
            {"view_0.weight", {view_width, n_view_inputs}, bottleneck_width + n_direction_features}, {"view_0.bias", {view_width}, 0},
            {"view_1.weight", {view_width, view_width}, view_width}, {"view_1.bias", {view_width}, 0},
            {"view_2.weight", {view_width, view_width}, view_width}, {"view_2.bias", {view_width}, 0},
            {"rgb.weight", {3, view_width + n_view_inputs}, view_width + bottleneck_width + n_direction_features}, {"rgb.bias", {3}, 0},
        };
    }

    NerfMlp::NerfMlp(const int64_t n_features, const int64_t density_width, const int64_t bottleneck_width, const int64_t view_width, const int64_t direction_degree) {
        if (density_width % 8 != 0 || bottleneck_width % 8 != 0 || view_width % 8 != 0) throw std::runtime_error("ZipNeRF: nerf mlp widths must be multiples of 8");
        const auto parameters = nerf_mlp_parameters(n_features, density_width, bottleneck_width, view_width, direction_degree);
        m_layout.n_features = std::get<1>(parameters[density_0_weight])[1];
        m_layout.density_width = density_width;
        m_layout.bottleneck_width = bottleneck_width;
        m_layout.view_width = view_width;
        m_layout.n_direction_features = 3 + 6 * direction_degree;
        m_layout.n_inputs = density_width + config::direction_block_width(direction_degree);
        m_layout.n_view_inputs = std::get<1>(parameters[view_0_weight])[1];
        m_layout.n_rgb_inputs = view_width + m_layout.n_view_inputs;
        m_layout.weights[0] = 0;
        for (int weight = 0; weight < n_weights; weight++) {
            m_layout.weights[weight + 1] = round_up(m_layout.weights[weight] + weight_rows(weight, m_layout) * weight_cols(weight, m_layout), segment_alignment);
        }
        int64_t n_bias_grads = 0;
        for (int weight = 0; weight <= n_weights; weight++) {
            m_layout.bias_grads[weight] = n_bias_grads;
            if (weight == b_density || weight == b_1 || weight == b_2) n_bias_grads += round_up(weight_rows(weight, m_layout), segment_alignment);
        }
        m_layout.params[0] = 0;
        for (int parameter = 0; parameter < n_parameters; parameter++) {
            int64_t size = 1;
            for (const int64_t dim : std::get<1>(parameters[parameter])) size *= dim;
            m_layout.params[parameter + 1] = m_layout.params[parameter] + size;
        }
    }

    // the column-major (in x out) view of a packed row-major (out x in) weight, so that op T yields the weight
    static Operand weight(const torch::Tensor& packed, const Layout& layout, const int w) {
        const int64_t cols = weight_cols(w, layout);
        return {static_cast<const __half*>(packed.data_ptr()) + layout.weights[w], cols, weight_rows(w, layout), cols, CUDA_R_16F};
    }
    torch::Tensor NerfMlp::pack(const torch::Tensor& params) const {
        check_input(params, "params", torch::kFloat);
        if (params.numel() < m_layout.params[n_parameters]) throw std::runtime_error("ZipNeRF: invalid nerf mlp parameters");
        torch::Tensor packed = torch::empty({m_layout.weights[n_weights]}, params.options().dtype(torch::kHalf));
        pack_kernel<<<n_blocks(m_layout.weights[n_weights]), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            params.data_ptr<float>(), m_layout, reinterpret_cast<__half*>(packed.data_ptr<torch::Half>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return packed;
    }

    std::tuple<torch::Tensor, std::vector<torch::Tensor>> NerfMlp::forward(
        const torch::Tensor& features,
        torch::Tensor& inputs,
        const torch::Tensor& packed,
        const bool training) const
    {
        if (features.size(1) != m_layout.n_features || inputs.size(1) != m_layout.n_inputs || packed.numel() != m_layout.weights[n_weights]) {
            throw std::runtime_error("ZipNeRF: invalid nerf mlp inputs");
        }
        const int64_t n = features.size(0);
        const torch::TensorOptions half = features.options();
        const __half* packed_data = static_cast<const __half*>(packed.data_ptr());
        torch::Tensor v0 = torch::empty({n, m_layout.view_width}, half), v1 = torch::empty({n, m_layout.view_width}, half), v2 = torch::empty({n, m_layout.view_width}, half);
        torch::Tensor outputs = torch::empty({n, n_outputs}, half);
        std::vector<torch::Tensor> masks;
        for (const int64_t rows : {m_layout.density_width, m_layout.view_width, m_layout.view_width, m_layout.view_width}) masks.push_back(training ? relu_mask(rows, n) : torch::Tensor());
        const auto relu = [&](const int bias, const int mask) {
            Epilogue epilogue;
            epilogue.type = bias >= 0 ? (training ? CUBLASLT_EPILOGUE_RELU_AUX_BIAS : CUBLASLT_EPILOGUE_RELU_BIAS) : (training ? CUBLASLT_EPILOGUE_RELU_AUX : CUBLASLT_EPILOGUE_RELU);
            if (bias >= 0) epilogue.bias = packed_data + m_layout.weights[bias];
            if (training) {
                epilogue.aux = masks[mask].data_ptr();
                epilogue.aux_ld = mask_ld(mask == 0 ? m_layout.density_width : m_layout.view_width);
            }
            return epilogue;
        };
        matmul(weight(packed, m_layout, w_density), true, operand(features), false, inputs.data_ptr(), m_layout.n_inputs, CUDA_R_16F, 0.0f, relu(b_density, 0));
        matmul(weight(packed, m_layout, w_combined), true, operand(inputs), false, v0.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, relu(-1, 1));
        matmul(weight(packed, m_layout, w_skip), true, operand(inputs), false, outputs.data_ptr(), n_outputs, CUDA_R_16F, 0.0f);
        matmul(weight(packed, m_layout, w_1), true, operand(v0), false, v1.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, relu(b_1, 2));
        matmul(weight(packed, m_layout, w_2), true, operand(v1), false, v2.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, relu(b_2, 3));
        matmul(weight(packed, m_layout, w_rgb), true, operand(v2), false, outputs.data_ptr(), n_outputs, CUDA_R_16F, 1.0f);
        std::vector<torch::Tensor> context = {inputs, v0, v1, v2};
        context.insert(context.end(), masks.begin(), masks.end());
        return {outputs, context};
    }

    torch::Tensor NerfMlp::backward(
        const torch::Tensor& grad_outputs,
        const torch::Tensor& features,
        const torch::Tensor& packed,
        const std::vector<torch::Tensor>& context,
        const torch::Tensor& params,
        torch::Tensor& grad_params) const
    {
        if (context.size() != 8 || packed.numel() != m_layout.weights[n_weights]) throw std::runtime_error("ZipNeRF: invalid nerf mlp backward inputs");
        check_input(params, "params", torch::kFloat);
        check_input(grad_params, "grad_params", torch::kFloat);
        const torch::Tensor& inputs = context[0];
        const torch::Tensor& v0 = context[1];
        const torch::Tensor& v1 = context[2];
        const torch::Tensor& v2 = context[3];
        const int64_t n = features.size(0);
        const torch::TensorOptions half = features.options();
        // weight gradients in fp32 (in the packed layout), bias gradients (reduced in fp32 by the epilogues) in fp16
        torch::Tensor weight_grads = torch::empty({m_layout.weights[n_weights]}, half.dtype(torch::kFloat));
        torch::Tensor bias_grads = torch::empty({m_layout.bias_grads[n_weights]}, half);
        float* weight_grads_data = weight_grads.data_ptr<float>();
        __half* bias_grads_data = reinterpret_cast<__half*>(bias_grads.data_ptr<torch::Half>());
        const auto drelu = [&](const int mask, const int64_t rows, const int bias) {
            Epilogue epilogue;
            epilogue.type = bias >= 0 ? CUBLASLT_EPILOGUE_DRELU_BGRAD : CUBLASLT_EPILOGUE_DRELU;
            if (bias >= 0) epilogue.bias = bias_grads_data + m_layout.bias_grads[bias];
            epilogue.aux = context[4 + mask].data_ptr();
            epilogue.aux_ld = mask_ld(rows);
            return epilogue;
        };
        // weight gradients (out x in, row-major) = grad_out^T @ in, i.e., column-major (in x out) = in^T-op chain below
        const auto weight_grad = [&](const torch::Tensor& grad_out, const torch::Tensor& in, const int w) {
            matmul(operand(in), false, operand(grad_out), true, weight_grads_data + m_layout.weights[w], weight_cols(w, m_layout), CUDA_R_32F, 0.0f);
        };
        torch::Tensor grad_v2 = torch::empty({n, m_layout.view_width}, half), grad_v1 = torch::empty({n, m_layout.view_width}, half), grad_v0 = torch::empty({n, m_layout.view_width}, half);
        torch::Tensor grad_hidden = torch::empty({n, m_layout.density_width}, half), grad_features = torch::empty({n, features.size(1)}, half);
        // a weight operand with op N is w^T, which maps output to input gradients
        matmul(weight(packed, m_layout, w_rgb), false, operand(grad_outputs), false, grad_v2.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, drelu(3, m_layout.view_width, b_2));
        weight_grad(grad_outputs, v2, w_rgb);
        matmul(weight(packed, m_layout, w_2), false, operand(grad_v2), false, grad_v1.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, drelu(2, m_layout.view_width, b_1));
        weight_grad(grad_v2, v1, w_2);
        matmul(weight(packed, m_layout, w_1), false, operand(grad_v1), false, grad_v0.data_ptr(), m_layout.view_width, CUDA_R_16F, 0.0f, drelu(1, m_layout.view_width, -1));
        weight_grad(grad_v1, v0, w_1);
        weight_grad(grad_v0, inputs, w_combined);
        weight_grad(grad_outputs, inputs, w_skip);
        // hidden gradient from both layers reading inputs (only the first density_width input columns are hidden features)
        const Operand combined = weight(packed, m_layout, w_combined), skip = weight(packed, m_layout, w_skip);
        matmul({combined.data, m_layout.density_width, combined.cols, combined.ld, combined.type}, false, operand(grad_v0), false, grad_hidden.data_ptr(), m_layout.density_width, CUDA_R_16F, 0.0f);
        matmul({skip.data, m_layout.density_width, skip.cols, skip.ld, skip.type}, false, operand(grad_outputs), false, grad_hidden.data_ptr(), m_layout.density_width, CUDA_R_16F, 1.0f, drelu(0, m_layout.density_width, b_density));
        weight_grad(grad_hidden, features, w_density);
        matmul(weight(packed, m_layout, w_density), false, operand(grad_hidden), false, grad_features.data_ptr(), features.size(1), CUDA_R_16F, 0.0f);
        unpack_kernel<<<n_blocks(m_layout.params[n_parameters]), block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
            params.data_ptr<float>(), weight_grads_data, bias_grads_data, m_layout, grad_params.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return grad_features;
    }

}
