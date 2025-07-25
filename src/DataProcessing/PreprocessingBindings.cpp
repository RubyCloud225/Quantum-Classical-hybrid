#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer/tokenizer.hpp"
#include "normaliser/bert.hpp"
#include "normaliser/byte_level.hpp"
#include "normaliser/digits.hpp"
#include "normaliser/metaspace.hpp"
#include "normaliser/prepend.hpp"
#include "normaliser/replace.hpp"
#include "normaliser/unicode_processor.hpp"
#include "models/GaussianNoise.hpp"
#include "models/LinearRegression.hpp"
#include "models/LayerNormalization.hpp"
#include "models/sampleData.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <cstring>

namespace py = pybind11;

py::dict run_preprocessing(const std::string& input) {
    // Tokenization
    Tokenizer tokenizer;
    std::vector<std::string> tokens = tokenizer.tokenize(input);
    int totalTokens = tokenizer.countTokens(tokens);
    int uniqueTokens = tokenizer.countUniqueTokens(tokens);
    int totalWords = tokenizer.countWords(tokens);
    int sentences = tokenizer.countSentences(input);
    int totalPunctuation = tokenizer.countPunctuation(input);

    // Positional Embeddings (optional to expose in return)
    auto positionalEmbeddings = tokenizer.createPositionalEmbeddings(tokens);

    // Gaussian noise setup
    std::vector<double> mean = {static_cast<double>(totalTokens), static_cast<double>(uniqueTokens)};
    std::vector<std::vector<double>> covariance = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> weights = {1.0, 1.0};
    GaussianNoise noise(mean, covariance, weights);

    // Generate noise samples
    std::vector<std::vector<double>> noiseSamples;
    for (int i = 0; i < 10; ++i) {
        noiseSamples.push_back(noise.generateNoise());
    }

    // Linear regression data
    std::vector<std::pair<double, double>> regressionData;
    for (const auto& ns : noiseSamples) {
        regressionData.emplace_back(ns[0], static_cast<double>(totalTokens));
    }

    // Train/test split
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(regressionData.begin(), regressionData.end(), g);
    size_t trainSize = static_cast<size_t>(0.8 * regressionData.size());
    std::vector<std::pair<double, double>> trainData(regressionData.begin(), regressionData.begin() + trainSize);
    std::vector<std::pair<double, double>> testData(regressionData.begin() + trainSize, regressionData.end());

    // Fit regression model
    LinearRegression lr;
    lr.fit(trainData);

    // Predict Y for test data
    std::vector<double> predictedY;
    for (const auto& [x, _] : testData) {
        predictedY.push_back(lr.predict(x));
    }

    // Normalize NLL
    std::vector<double> nllValues;
    for (const auto& sample : noiseSamples) {
        try {
            nllValues.push_back(noise.negativeLogLikelihood(sample));
        } catch (...) {
            nllValues.push_back(0.0);
        }
    }

    LayerNormalization norm(nllValues.size());
    norm.resetParameters();
    std::vector<double> normalizedNLL = norm.forward(nllValues);

    // Compute density and entropy (reuse last sample)
    double density = noise.calculateDensity(noiseSamples[0]);
    double entropy = noise.calculateEntropy();

    // Build SampleData set
    std::vector<SampleData> dataset;
    for (size_t i = 0; i < noiseSamples.size(); ++i) {
        SampleData sample;
        sample.token_embedding = {
            static_cast<double>(totalTokens),
            static_cast<double>(uniqueTokens),
            static_cast<double>(totalWords),
            static_cast<double>(totalPunctuation)
        };
        sample.noise = noiseSamples[i];
        sample.target_value = static_cast<double>(totalTokens);
        sample.normalized_noise = normalizedNLL;  // technically normalized NLL
        sample.density = density;
        sample.nll = nllValues[i];
        sample.entopy = entropy;
        dataset.push_back(sample);
    }

    // Save dataset to file
    saveSamples(dataset, "sample_data.bin");

    // Return metadata to Python
    py::dict result;
    result["tokens"] = totalTokens;
    result["unique_tokens"] = uniqueTokens;
    result["words"] = totalWords;
    result["punctuation"] = totalPunctuation;
    return result;
}

PYBIND11_MODULE(preprocessing, m) {
    m.def("run_preprocessing", &run_preprocessing, py::arg("input"), "Preprocess text and generate data for DiT training");
    py::class_<BertNormaliser>(m, "BertNormaliser")
        .def(py::init<>())
        .def("bertCleaning", &BertNormaliser::bertCleaning, py::arg("input"))
        .def("stripAccents", &BertNormaliser::stripAccents, py::arg("input"))
        .def("utf8ToUtf32", &BertNormaliser::utf8ToUtf32, py::arg("input"))
        .def("utf32ToUtf8", &BertNormaliser::utf32ToUtf8, py::arg("input"))
        .def("pretok", &BertNormaliser::pretok, py::arg("input"));
    py::class_<ByteNormalizer>(m, "ByteNormalizer")
        .def("ByteNormalise", &ByteNormalizer::ByteNormalise, py::arg("input"), py::arg("debug") = false)
        .def("pretok", &ByteNormalizer::pretok, py::arg("input"), py::arg("debug") = false);
    py::class_<DigitNormaliser>(m, "DigitNormaliser")
        .def("normaliseDigits", &DigitNormaliser::normaliseDigits, py::arg("input"), py::arg("debug") = false);
    py::class_<MetaspaceNormaliser>(m, "MetaspaceNormaliser")
        .def(py::init<>())
        .def("setReplacement", &MetaspaceNormaliser::setReplacement, py::arg("replacement"), py::arg("prependScheme") = false)
        .def("getReplacement", &MetaspaceNormaliser::getReplacement)
        .def("getPrependScheme", &MetaspaceNormaliser::getPrependScheme)
        .def("setPrependScheme", &MetaspaceNormaliser::setPrependScheme)
        .def("pretok", &MetaspaceNormaliser::pretok, py::arg("input"), py::arg("debug") = false);
    py::class_<Prepend>(m, "Prepend")
        .def(py::init<const std::string&, const std::string&>(), py::arg("filename"), py::arg("text"))
        .def("extract_normalised", &Prepend::extract_normalised, py::arg("text"))
        .def("build_comment_block", &Prepend::build_comment_block, py::arg("values"))
        .def("write_comment_block", &Prepend::write_comment_block, py::arg("filename"), py::arg("content"));
    py::class_<Replace>(m, "Replace")
        .def(py::init<const std::string&, const std::string&>(), py::arg("regexPattern"), py::arg("replaceWith"))
        .def("applyReplace", &Replace::applyReplace, py::arg("content"))
        .def("clone", &Replace::clone)
        .def("serialise", &Replace::serialise)
        .def("deserialise", &Replace::deserialise, py::arg("serialisedData"))
        .def_property("pattern", &Replace::getPattern, nullptr)
        .def_property("replace", &Replace::getReplace, nullptr);
    py::class_<UnicodeProcessor>(m, "UnicodeProcessor")
        .def_static("normaliseString", &UnicodeProcessor::normaliseString, py::arg("input"), py::arg("mode") = UNormalization2Mode::UNORM2_COMPOSE)
        .def_static("removeDiacritics", &UnicodeProcessor::removeDiacritics, py::arg("input"));
}
