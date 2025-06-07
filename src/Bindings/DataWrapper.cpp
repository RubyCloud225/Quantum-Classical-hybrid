#include "GaussianNoise.hpp"
#include "LayerNormalization.hpp"
#include "LinearRegression.hpp"
#include "tokenizer.hpp"
#include "sampleData.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include "PreprocessingBindings.cpp"

namespace py = pybind11;
PYBIND11_MODULE(Data_wrapper, m) {
    py::class_<GaussianNoise>(m, "GaussianNoise")
        .def(py::init<const std::vector<double>&, const std::vector<std::vector<double>>&,
             const std::vector<double>&>(), py::arg("mean"), py::arg("covariance"), py::arg("weights"))
        .def("generateNoise", &GaussianNoise::generateNoise)
        .def("calculateDensity", &GaussianNoise::calculateDensity, py::arg("sample"))
        .def("negativeLogLikelihood", &GaussianNoise::negativeLogLikelihood, py::arg("sample"))
        .def("calculateEntropy", &GaussianNoise::calculateEntropy);
    
    py::class_<LayerNormalization>(m, "LayerNormalization")
        .def(py::init<double, double>(), py::arg("features"), py::arg("epsilon"))
        .def("resetParameters", &LayerNormalization::resetParameters)
        .def("forward", &LayerNormalization::forward, py::arg("input"))
        .def("getGamma", &LayerNormalization::getGamma)
        .def("getBeta", &LayerNormalization::getBeta);
    
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit, py::arg("data"))
        .def("predict", &LinearRegression::predict, py::arg("x"))
        .def("reshapeData", &LinearRegression::reshapeData, py::arg("x"), py::arg("y"), py::arg("reshapeData"));

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("countTokens", &Tokenizer::countTokens, py::arg("tokens"))
        .def("countUniqueTokens", &Tokenizer::countUniqueTokens, py::arg("tokens"))
        .def("countSentences", &Tokenizer::countSentences, py::arg("input"))
        .def("countWords", &Tokenizer::countWords, py::arg("tokens"))
        .def("countCharacters", &Tokenizer::countCharacters, py::arg("input"))
        .def("countPunctuation", &Tokenizer::countPunctuation, py::arg("input"))
        .def("createPositionalEmbeddings", &Tokenizer::createPositionalEmbeddings, py::arg("tokens"))
        .def_readonly("tokens", &Tokenizer::tokens)
        .def_readonly("input", &Tokenizer::input);

    py::class_<SampleData>(m, "SampleData")
        .def(py::init<>())
        .def_readwrite("token_embedding", &SampleData::token_embedding)
        .def_readwrite("noise", &SampleData::noise)
        .def_readwrite("target_value", &SampleData::target_value)
        .def_readwrite("normalized_noise", &SampleData::normalized_noise)
        .def_readwrite("density", &SampleData::density)
        .def_readwrite("nll", &SampleData::nll)
        .def_readwrite("entopy", &SampleData::entopy);
    
    m.def("saveSamples", &saveSamples, "Save SampleData to binary file");

    m.def("run_preprocessing", &run_preprocessing, "Run full preprocessing and save SampleData to file");
}
