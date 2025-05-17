#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.hpp"
#include "GaussianNoise.hpp"
#include "LinearRegression.hpp"
#include "LayerNormalization.hpp"
#include "sampleData.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <string>

namespace py = pybind11;


py::dict run_preprocessing(
    const std::string& input, 
            std::string& tokens, 
            std::string& totalTokens,
            std::string& uniqueTokens,
            std::string& totalWords,
            std::string& sentences,
            std::string& totalPunctuation) {
    // Tokenization
    Tokenizer tokenizer;
    tokens = tokenizer.tokenize(input);
    totalTokens = tokenizer.countTokens(tokens);
    uniqueTokens = tokenizer.countUniqueTokens(tokens);
    totalWords = tokenizer.countWords(tokens);
    sentences = tokenizer.countSentences(input);
    totalPunctuation = tokenizer.countPunctuation(input);

    // Positional Embeddings (optional to expose in return)
    auto positionalEmbeddings = token.createPositionalEmbeddings(tokens);

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
    result = py::dict("tokens" = totalTokens, "unique_tokens" = uniqueTokens, "words" = totalWords, "punctuation" = totalPunctuation);
    if (!result.empty()) {
        result["warning"]
    }
    return result;
                                
}

PYBIND11_MODULE(preprocessing, m) {
    m.def("run_preprocessing", &run_preprocessing, py::arg("input"), "Preprocess text and generate data for DiT training");
}
