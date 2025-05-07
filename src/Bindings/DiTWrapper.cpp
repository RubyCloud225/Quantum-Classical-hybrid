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

namespace py = pybind11;
PYBIND11_MODULE(dit_wrapper, m) {
    py::class_<BetaSchedule>(m, "BetaSchedule")
        .def(py::init<double, int>()) // Initial Beta, total epochs
        .def("update", &BetaSchedule::update, py::arg("nll_losses"), py::arg("entropy_losses"), py::arg("epoch"));

    //Wrapping GaussianDiffusion
    py::class_<GaussianDiffusion>(m, "GaussianDiffusion")
        .def(py::init<int, const std::vector<double>>()) // number of timesteps, beta schedule
        .def("train", &GaussianDiffusion::train, py::arg("training_noise"), py::arg("targets"), py::arg("nll_losses") py::arg("epochs"))
        .def("forward", &GaussianDiffusion::forward, py::arg("x_prev"), py::arg("t"));
    
    //Wrapping DiffusionModel
    py::class_<DiffusionModel>(m, "DiffusionModel")
        .def(py::init<int, int>()) // input size, output size
        .def("forward", &DiffusionModel::forward, py::arg("x_t"), py::arg("t"), py::arg("clip_denoised"), py::arg("denoised_fn"), py::arg("model_kwargs"));
    
    //Wrapping EpisilonPredictor
    py::class_<EpsilonPredictor>(m, "EpsilonPredictor")
        .def(py::init<int, int>()) // input size, output size
        .def("predict", &EpsilionPredictor::predict, py::arg("x_t"), py::arg("t"));
    
    //Wrapping DiffusionSample
    py::class_<DiffusionSample>(m, "DiffusionSample")
        .def(py::init<>())
        .def("p_sample", &DiffusionSample::p_sample, py::arg("shape"), py::arg("clip_denoised"), py::arg("denoised_fn"), py::arg("model_kwargs"), py::arg("device") = "cpu")
        .def("p_sample_loop_progressive", &DiffusionSample::p_sample_loop_progressive, py::arg("shape"), py::arg("clip_denoised"), py::arg("denoised_fn"), py::arg("model_kwargs"), py::arg("device") = "cpu");
    
    //Wrapping NormalDist
    py::class_<NormalDist>(m, "NormalDist")
        .def_static("log_prob_from_predictions", &NormalDist::log_prob_from_predictions, py::arg("y"), py::arg("x_start_pred"), py::arg("eps_pred"))
        .def_static("gradients", &NormalDist::gradients, py::arg("y"), py::arg("x_start_pred"), py::arg("eps_pred"), py::arg("dfd_y"), py::arg("dfd_mu"), py::arg("dfd_sigma"));
    
    // Wrape main loop function (for training)
    m.def("train_diffusion_model", &train_diffusion_model, 
          py::arg("initial_beta"),
          py::arg("total_epochs"),
          py::arg("learning_rate"),
          py::arg("beta1"),
          py::arg("beta2"),
          py::arg("epsilon"),
          py::arg("input_size"),
          py::arg("output_size"),
          py::arg("epochs"),
          py::arg("sample_data_path"));
}
