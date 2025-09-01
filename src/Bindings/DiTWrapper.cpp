#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"
#include "NN/EpsilonPredictor.hpp"
#include "Diffusion_model.hpp"
#include "Diffusion_Sample.hpp"
#include "NormalDist.hpp"
#include "sampleData.hpp"
//#include "train_diffusion_model.hpp"

namespace py = pybind11;
PYBIND11_MODULE(dit_wrapper, m) {
    py::class_<BetaSchedule>(m, "BetaSchedule")
        .def(py::init<double, int>()) // Initial Beta, total epochs
        .def("update", &BetaSchedule::update, py::arg("nll_losses"), py::arg("entropy_losses"), py::arg("epoch"));

    //Wrapping GaussianDiffusion
    py::class_<GaussianDiffusion>(m, "GaussianDiffusion")
        .def(py::init<int, double, double>(), py::arg("num_timesteps"), py::arg("beta_start"), py::arg("beta_end"))
        .def("train", &GaussianDiffusion::train, py::arg("data"), py::arg("epochs"))
        .def("forward", &GaussianDiffusion::forward, py::arg("x_prev"), py::arg("t"));
    
    //Wrapping DiffusionModel
    py::class_<DiffusionModel>(m, "DiffusionModel")
        .def(py::init<int, int>()) // input size, output size
        .def("compute_mean_variance", &DiffusionModel::compute_mean_variance, py::arg("x_t"), py::arg("t"), py::arg("mean"), py::arg("variance"));
    
    //Wrapping EpisilonPredictor
    py::class_<EpsilonPredictor>(m, "EpsilonPredictor")
        .def(py::init<int, int>()) // input size, output size
        .def("predictEpsilon", &EpsilonPredictor::predictEpsilon, py::arg("x_t"), py::arg("t"));
    
    //Wrapping DiffusionSample
    py::class_<DiffusionSample>(m, "DiffusionSample")
        .def(py::init<DiffusionModel&, const std::vector<double>&>(), py::arg("model"), py::arg("noise_schedule"))
        .def("p_sample", &DiffusionSample::p_sample, py::arg("shape"), py::arg("clip_denoised"), py::arg("denoised_fn"), py::arg("model_kwargs"), py::arg("device") = "cpu")
        .def("p_sample_loop_progressive", &DiffusionSample::p_sample_loop_progressive, py::arg("shape"), py::arg("clip_denoised"), py::arg("denoised_fn"), py::arg("model_kwargs"), py::arg("device") = "cpu");
    
    //Wrapping NormalDist

    m.def("log_prob_from_predictions", &NormalDist::log_prob_from_predictions, py::arg("y"), py::arg("x_start_pred"), py::arg("eps_pred"));
    m.def("gradients", &NormalDist::gradients, py::arg("y"), py::arg("x_start_pred"), py::arg("eps_pred"), py::arg("dfd_y"), py::arg("dfd_mu"), py::arg("dfd_sigma"));
    
    // Wrape main loop function (for training)
   //m.def("train_diffusion_model", &train_diffusion_model, 
    //      py::arg("initial_beta"),
    //      py::arg("total_epochs"),
    //      py::arg("learning_rate"),
    //      py::arg("beta1"),
    //      py::arg("beta2"),
    //      py::arg("epsilon"),
    //      py::arg("input_size"),
    //      py::arg("output_size"),
    //      py::arg("epochs"),
    //      py::arg("sample_data_path"));
}
