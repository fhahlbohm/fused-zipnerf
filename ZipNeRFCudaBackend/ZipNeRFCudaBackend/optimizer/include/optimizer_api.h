#pragma once

#include <torch/extension.h>

#include <vector>

namespace zipnerf::optimizer {

    // Adam of several flat buffers in one launch per step; gradients (fp32, or fp16 clamped to the fp16 range so that an
    // overflow cannot reach the moments) are unscaled, nan_to_num-ed, and zeroed after the update; buffers with level
    // offsets are hash grid tables, whose levels get the regularizer 0.5 * reg_weight * mean(p^2), and the last block
    // writes the scale feature magnitudes sqrt(max_value^2 + mean(p^2)) and the regularizer
    class Adam {
    public:
        Adam(
            const std::vector<torch::Tensor>& params,
            const std::vector<torch::Tensor>& grads,
            const std::vector<double>& grad_scales,
            const std::vector<std::vector<int64_t>>& level_offsets,
            double reg_weight,
            double max_value,
            double beta1,
            double beta2,
            double eps,
            torch::Tensor magnitudes);

        void step(double lr);

        torch::Tensor regularization() const { return m_regularization; }

    private:
        std::vector<torch::Tensor> m_params, m_grads, m_exp_avgs, m_exp_avg_sqs;
        std::vector<double> m_grad_scales;
        std::vector<std::vector<int64_t>> m_level_offsets;
        double m_reg_weight, m_max_value;
        float m_beta1, m_beta2, m_eps;
        int64_t m_step = 0;
        int m_n_levels = 0;
        int m_n_sms = 0;
        torch::Tensor m_magnitudes, m_regularization, m_level_sums, m_counter;
    };

}
