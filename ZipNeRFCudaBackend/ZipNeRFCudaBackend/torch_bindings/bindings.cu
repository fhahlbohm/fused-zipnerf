#include <torch/extension.h>
#include "featurizer.h"
#include "sampling_api.h"
#include "optimizer_api.h"
#include "nerf_mlp.h"
#include "zipnerf_config.h"

namespace featurization_api = zipnerf::featurization;
namespace sampling_api = zipnerf::sampling;
namespace optimizer_api = zipnerf::optimizer;
namespace mlp_api = zipnerf::mlp;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("grid_max_value") = zipnerf::config::hash_init_range / zipnerf::config::precondition_scaling;
    m.attr("nerf_mlp_grad_scale") = zipnerf::config::nerf_mlp_grad_scale;
    m.def("set_rtc_dirs", &featurization_api::set_rtc_dirs);
    py::class_<featurization_api::Featurizer>(m, "Featurizer")
        .def(py::init<int, const std::vector<int64_t>&, const std::vector<int64_t>&, int, int>())
        .def("forward", &featurization_api::Featurizer::forward)
        .def("backward", &featurization_api::Featurizer::backward)
        .def("backward_mlp", &featurization_api::Featurizer::backward_mlp);
    m.def("nerf_mlp_parameters", &mlp_api::nerf_mlp_parameters);
    py::class_<mlp_api::NerfMlp>(m, "NerfMlp")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
        .def_property_readonly("n_features", &mlp_api::NerfMlp::n_features)
        .def_property_readonly("n_inputs", &mlp_api::NerfMlp::n_inputs)
        .def("pack", &mlp_api::NerfMlp::pack)
        .def("forward", &mlp_api::NerfMlp::forward)
        .def("backward", &mlp_api::NerfMlp::backward);
    py::class_<optimizer_api::Adam>(m, "Adam")
        .def(py::init<const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&, const std::vector<double>&,
            const std::vector<std::vector<int64_t>>&, double, double, double, double, double, torch::Tensor>())
        .def("step", &optimizer_api::Adam::step)
        .def("regularization", &optimizer_api::Adam::regularization);
    m.def("generate_rays", &sampling_api::generate_rays_wrapper);
    m.def("view_rays", &sampling_api::view_rays_wrapper);
    m.def("initialize_rays", &sampling_api::initialize_rays_wrapper);
    m.def("sample_uniforms", &sampling_api::sample_uniforms_wrapper);
    m.def("composite_and_resample", &sampling_api::composite_and_resample_wrapper);
    m.def("train_final", &sampling_api::train_final_wrapper);
    m.def("composite_final", &sampling_api::composite_final_wrapper);
}
