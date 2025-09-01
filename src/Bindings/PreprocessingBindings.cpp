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
#include "json.hpp"
#include "dotenv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <random>
#include <algorithm>

namespace py = pybind11;

// Preprocessing pipeline function
py::dict run_preprocessing(const std::string& input) {
    // --- Bert normalization ---
    BertNormaliser bertNormaliser;
    std::string normalizedText = bertNormaliser.bertCleaning(input);
    normalizedText = bertNormaliser.stripAccents(normalizedText);

    // --- Byte normalization ---
    ByteNormalizer byteNormalizer;
    std::vector<std::string> byteNormalizedTokens = byteNormalizer.ByteNormalise(normalizedText, true);

    // --- Digit normalization ---
    std::vector<std::string> digitNormalizedTokens;
    try {
        DigitNormaliser digitNormaliser;
        digitNormalizedTokens = digitNormaliser.normaliseDigits(normalizedText, true);
    } catch (const std::exception& e) {
        std::cerr << "DigitNormaliser failed: " << e.what() << std::endl;
        digitNormalizedTokens.push_back(normalizedText);
    }

    // Join digit tokens for Metaspace normalization
    std::string digitNormalizedText;
    for (const auto& t : digitNormalizedTokens) digitNormalizedText += t;

    // --- Metaspace normalization ---
    std::vector<std::string> metaspaceTokens;
    try {
        MetaspaceNormaliser metaspaceNormaliser;
        // Convert first UTF-8 character to char if possible, otherwise only use ASCII
        metaspaceNormaliser.setReplacement(' ', true);
        metaspaceTokens = metaspaceNormaliser.pretok(digitNormalizedText, true);
    } catch (const std::exception& e) {
        std::cerr << "MetaspaceNormaliser failed: " << e.what() << std::endl;
        metaspaceTokens = digitNormalizedTokens;
    }

    // Join metaspace tokens
    std::string metaspaceNormalizedText;
    for (const auto& t : metaspaceTokens) metaspaceNormalizedText += t;

    // --- Prepend normalization ---
    Prepend prepend("dummy.txt", metaspaceNormalizedText);
    std::set<std::string> normalizedValues = prepend.extract_normalised(metaspaceNormalizedText);

    // --- Tokenization ---
    Tokenizer tokenizer;
    std::vector<std::string> tokens = tokenizer.tokenize(normalizedText);
    int totalTokens = tokenizer.countTokens(tokens);
    int uniqueTokens = tokenizer.countUniqueTokens(tokens);
    int totalWords = tokenizer.countWords(tokens);
    int totalPunctuation = tokenizer.countPunctuation(normalizedText);
    int sentences = tokenizer.countSentences(normalizedText);

    // --- Gaussian noise, regression, normalization ---
    std::vector<double> mean = {static_cast<double>(totalTokens), static_cast<double>(uniqueTokens)};
    std::vector<std::vector<double>> covariance = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<double> weights = {1.0, 1.0};
    GaussianNoise noise(mean, covariance, weights);

    std::vector<std::vector<double>> noiseSamples;
    for (int i = 0; i < 10; ++i) {
        noiseSamples.push_back(noise.generateNoise());
    }

    std::vector<std::pair<double, double>> regressionData;
    for (const auto& ns : noiseSamples) {
        regressionData.emplace_back(ns[0], static_cast<double>(totalTokens));
    }

    // Shuffle and split train/test
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(regressionData.begin(), regressionData.end(), g);
    size_t trainSize = static_cast<size_t>(0.8 * regressionData.size());
    std::vector<std::pair<double, double>> trainData(regressionData.begin(), regressionData.begin() + trainSize);
    std::vector<std::pair<double, double>> testData(regressionData.begin() + trainSize, regressionData.end());

    LinearRegression lr;
    lr.fit(trainData);

    std::vector<double> predictedY;
    for (const auto& [x, _] : testData) {
        predictedY.push_back(lr.predict(x));
    }

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

    double density = noise.calculateDensity(noiseSamples[0]);
    double entropy = noise.calculateEntropy();

    // Build SampleData set
    std::vector<SampleData> dataset;
    for (size_t i = 0; i < noiseSamples.size(); ++i) {
        SampleData sample;
        sample.token_embedding = {static_cast<double>(totalTokens),
                                  static_cast<double>(uniqueTokens),
                                  static_cast<double>(totalWords),
                                  static_cast<double>(totalPunctuation)};
        sample.noise = noiseSamples[i];
        sample.target_value = static_cast<double>(totalTokens);
        sample.normalized_noise = normalizedNLL;
        sample.density = density;
        sample.nll = nllValues[i];
        sample.entopy = entropy;
        dataset.push_back(sample);
    }

    saveSamples(dataset, "sample_data.bin");

    py::dict result;
    result["tokens"] = totalTokens;
    result["unique_tokens"] = uniqueTokens;
    result["words"] = totalWords;
    result["punctuation"] = totalPunctuation;
    return result;
}

// --- Pybind11 module ---
PYBIND11_MODULE(quantum_classical_hybrid, m) {
    m.def("run_preprocessing", &run_preprocessing, py::arg("input"),
          "Preprocess text and generate data for DiT training");

    py::class_<BertNormaliser>(m, "BertNormaliser")
        .def(py::init<>())
        .def("bertCleaning", &BertNormaliser::bertCleaning)
        .def("stripAccents", &BertNormaliser::stripAccents);

    py::class_<ByteNormalizer>(m, "ByteNormalizer")
        .def(py::init<>())
        .def("ByteNormalise", &ByteNormalizer::ByteNormalise, py::arg("input"), py::arg("debug") = false)
        .def("pretok", &ByteNormalizer::pretok, py::arg("input"), py::arg("debug") = false);

    py::class_<DigitNormaliser>(m, "DigitNormaliser")
        .def(py::init<>())
        .def("normaliseDigits", &DigitNormaliser::normaliseDigits, py::arg("input"), py::arg("debug") = false);

    py::class_<MetaspaceNormaliser>(m, "MetaspaceNormaliser")
        .def(py::init<>())
        .def("setReplacement", &MetaspaceNormaliser::setReplacement, py::arg("replacement"), py::arg("prependScheme") = false)
        .def("pretok", &MetaspaceNormaliser::pretok, py::arg("input"), py::arg("debug") = false);

    py::class_<Prepend>(m, "Prepend")
        .def(py::init<const std::string&, const std::string&>())
        .def("extract_normalised", &Prepend::extract_normalised)
        .def("build_comment_block", &Prepend::build_comment_block)
        .def("write_comment_block", &Prepend::write_comment_block);

    py::class_<Replace>(m, "Replace")
        .def(py::init<const std::string&, const std::string&>())
        .def("applyReplace", &Replace::applyReplace)
        .def("clone", &Replace::clone)
        .def("serialise", &Replace::serialise)
        .def_static("deserialise", &Replace::deserialise)
        .def_property("pattern", &Replace::getPattern, nullptr)
        .def_property("replace", &Replace::getReplace, nullptr);

    py::class_<UnicodeProcessor>(m, "UnicodeProcessor")
        .def_static("normaliseString", &UnicodeProcessor::normaliseString)
        .def_static("removeDiacritics", &UnicodeProcessor::removeDiacritics);

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("tokenize", &Tokenizer::tokenize)
        .def("countTokens", &Tokenizer::countTokens)
        .def("countUniqueTokens", &Tokenizer::countUniqueTokens)
        .def("countSentences", &Tokenizer::countSentences)
        .def("countPunctuation", &Tokenizer::countPunctuation);

    py::class_<GaussianNoise>(m, "GaussianNoise")
        .def(py::init<const std::vector<double>&, const std::vector<std::vector<double>>&, const std::vector<double>&>())
        .def("generateNoise", &GaussianNoise::generateNoise)
        .def("negativeLogLikelihood", &GaussianNoise::negativeLogLikelihood)
        .def("calculateDensity", &GaussianNoise::calculateDensity)
        .def("calculateEntropy", &GaussianNoise::calculateEntropy);

    py::class_<LayerNormalization>(m, "LayerNormalization")
        .def(py::init<double, double>(), py::arg("features"), py::arg("epsilon"))
        .def("resetParameters", &LayerNormalization::resetParameters)
        .def("forward", &LayerNormalization::forward);

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", [](LinearRegression &lr, const std::vector<std::pair<double,double>>& data){ lr.fit(data); })
        .def("predict", &LinearRegression::predict);

    m.def("load_dotenv", &load_dotenv, py::arg("path") = "");

    py::class_<json::jsonValue>(m, "jsonValue")
        .def("to_py", &json::jsonValue::to_py);
}
