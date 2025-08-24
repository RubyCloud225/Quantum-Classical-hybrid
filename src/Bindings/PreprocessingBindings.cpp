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
#include <cstring>

namespace py = pybind11;

py::dict run_preprocessing(const std::string& input) {
    // === Normalisation step ===
    BertNormaliser bertNormaliser;
    std::string normalizedText = bertNormaliser.bertCleaning(input);
    normalizedText = bertNormaliser.stripAccents(normalizedText);

    // --- Byte normalization ---//
    ByteNormalizer byteNormalizer;
    std::vector<std::string> byteNormalizedTokens = byteNormalizer.ByteNormalise(normalizedText, true);
    // join tokens back to a single string
    //-------- Digits ---------------//
    DigitNormaliser digitNormaliser;
    std::string digitNormalizedText = digitNormaliser.normaliseDigits(byteNormalizedTokens, true);
    // --- Metaspace normalization ---//
    MetaspaceNormaliser metaspaceNormaliser;
    metaspaceNormaliser.setReplacement("Ġ", true);
    metaspaceNormaliser.setReplacement(" ", true);
    std::string metaspaceNormalizedText = metaspaceNormaliser.pretok(digitNormalizedText, true);

    
    //===== Prepend normalization ===//
    Prepend prepend;
    std::set<std::string> normalizedValues = prepend.extract_normalised(digitNormalizedText);
    
    // === Tokenization on normalized text ===//
    Tokenizer tokenizer;
    std::vector<std::string> tokens = tokenizer.tokenize(normalizedText);
    int totalTokens = tokenizer.countTokens(tokens);
    int uniqueTokens = tokenizer.countUniqueTokens(tokens);
    int totalWords = tokenizer.countWords(tokens);
    int sentences = tokenizer.countSentences(normalizedText);
    int totalPunctuation = tokenizer.countPunctuation(normalizedText);

    // === Gaussian noise generation, regression, normalization ===
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
        sample.token_embedding = {
            static_cast<double>(totalTokens),
            static_cast<double>(uniqueTokens),
            static_cast<double>(totalWords),
            static_cast<double>(totalPunctuation)
        };
        sample.noise = noiseSamples[i];
        sample.target_value = static_cast<double>(totalTokens);
        sample.normalized_noise = normalizedNLL;
        sample.density = density;
        sample.nll = nllValues[i];
        sample.entropy = entropy;
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
        .def_static("deserialise", &Replace::deserialise, py::arg("serialisedData"))
        .def_property("pattern", &Replace::getPattern, nullptr)
        .def_property("replace", &Replace::getReplace, nullptr);
    py::class_<UnicodeProcessor>(m, "UnicodeProcessor")
        .def_static("normaliseString", &UnicodeProcessor::normaliseString, py::arg("input"), py::arg("mode") = UNormalization2Mode::UNORM2_COMPOSE)
        .def_static("removeDiacritics", &UnicodeProcessor::removeDiacritics, py::arg("input"));
    py::class_<Tokenizer>(m, "Tokeniser")
        .def(py::init<>())
        .def("tokenize", &Tokenizer::tokenize, py::arg("text"))
        .def("countTokens", &Tokenizer::countTokens, py::arg("text"))
        .def("countUniqueTokens", &Tokenizer::countUniqueTokens, py::arg("text"))
        .def("countSentences", &Tokenizer::countSentences, py::arg("text"))
        .def("countPunctuation", &Tokenizer::countPunctuation, py::arg("text"));
    py::class_<ImageNormalizer>(m, "ImageNormalizer")
        .def(py::init<>())
        .def("loadImage", &ImageNormalizer::loadImage, py::arg("image.jpg"));
    
    py::class_<PatchToken>(m, "PatchToken")
        .def(py::init<>())
        .def_readwrite("embedding", &PatchToken::embedding)
        .def_readwrite("row", &PatchToken::row)
        .def_readwrite("col", &PatchToken::col);

    py::class_<SampleData>(m, "SampleData")
        .def(py::init<>())
        .def_readwrite("token_embedding", &SampleData::token_embedding)
        .def_readwrite("noise", &SampleData::noise)
        .def_readwrite("target_value", &SampleData::target_value)
        .def_readwrite("normalized_noise", &SampleData::normalized_noise)
        .def_readwrite("density", &SampleData::density)
        .def_readwrite("nll", &SampleData::nll)
        .def_readwrite("entropy", &SampleData::entropy);

    py::class_<ImageSequentialTokenizer>(m, "ImageSequentialTokenizer")
        .def(py::init<>())
        .def_readwrite("textTokens", &ImageSequentialTokenizer::textTokens)
        .def_readwrite("imageTokens", &ImageSequentialTokenizer::imageTokens)
        .def("tokenizeImage", &ImageSequentialTokenizer::tokenizeImage);

    // Bindings from DataWrapper.cpp merged here:
    py::class_<GaussianNoise>(m, "GaussianNoise")
        .def(py::init<const std::vector<double>&, const std::vector<std::vector<double>>&, const std::vector<double>&>())
        .def("generateNoise", &GaussianNoise::generateNoise)
        .def("negativeLogLikelihood", &GaussianNoise::negativeLogLikelihood)
        .def("calculateDensity", &GaussianNoise::calculateDensity)
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
}
