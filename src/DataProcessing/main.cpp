#include <iostream>
#include <algorithm>
#include <random>
#include "DataProcessing/tokenizer.hpp"
#include "DataProcessing/GaussianNoise.hpp"
#include "DataProcessing/LinearRegression.hpp"
#include "DataProcessing/LayerNormalization.hpp"
#include "sampleData.hpp"

int main () {
    // input string to tokenized
    std::string input = "words to be tokenized";
    // tokenize the input string
    std::vector<std::string> tokens = tokenize(input);
    //output tokens
    std::cout << "Tokens: " << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }

    // Count total tokens
    int totalTokens = countTokens(tokens);
    std::cout << "Total Tokens: " << totalTokens << std::endl;
    // Count Unique tokens
    int uniqueTokens = countUniqueTokens(tokens);
    std::cout << "Unique Tokens: " << uniqueTokens << std::endl;
    // Count word frequency
    int totalWords = countWords(tokens);
    std::cout << "Number of words: " << totalWords << std::endl;
    // count punctuation
    int totalPunctuation = countPunctuation(input);
    std::cout << "Number of punctuation: " << totalPunctuation << std::endl;

    //create positional embeddings
    auto positionalEmbeddings = createPositionalEmbeddings(tokens);
    std::cout << "\nPositional Embeddings: " << std::endl;
    for (const auto& pair : positionalEmbeddings) {
        std::cout << "Token" << pair.first << " Positions: ";
        for (const auto& pos : pair.second) {
            std::cout << pos << " ";
        }
        std::cout << std::endl;
    }

    // mean vector 
    std::vector<double> mean = {static_cast<double>(totalTokens), static_cast<double>(uniqueTokens)};
    // Covarance Matrix 
    std::vector<std::vector<double>> covariance = {
        {1.0, 0.0},
        {0.0, 1.0}
    };
    // apply weights
    std::vector<double> weights = {1.0, 1.0};

    GaussianNoise noise(mean, covariance, weights);

    // Generate and print Gaussian noise
    std::vector<double> noiseSamples;
    for (int i = 0; i < 10; ++i) {
        std::vector<double> sample = noise.generateNoise();
        std::cout << "Sample " << i + 1 << ": (" << sample[0] << ", " << sample[1] << ")" << std::endl;
    }
    // prepare data for linear regression
    std::vector<std::pair<double, double>> regressionData;
    for (const auto& noiseValue : noiseSamples) {
        regressionData.emplace_back(noiseValue, static_cast<double>(totalTokens));
    }
    // Shuffle regression data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(regressionData.begin(), regressionData.end(), g);
    // Split into training and testing sets (80% train, 20% test)
    size_t trainSize = static_cast<size_t>(0.8 * regressionData.size());
    std::vector<std::pair<double, double>> trainData(regressionData.begin(), regressionData.begin() + trainSize);
    std::vector<std::pair<double, double>> testData(regressionData.begin() + trainSize, regressionData.end());
    // initalise linear regression
    LinearRegression lr;
    // fit the model to the noise created by Gaussian
    lr.fit(trainData);
    // predict the total tokens using test data
    for (const auto& testSample : testData) {
        double x = testSample.first;
        double predictedY = lr.predict(x);
    }

    // prepare the data for layer normalization
    std::vector<double> Y = {testData[0].second}; // first element of test data
    LayerNormalization layerNorm(Y.size());
    // reset 
    layerNorm.resetParameters();
    // normalize
    std::vector<double> normalizedOutput = layerNorm.forward(Y);
    // Output normalized noise values
    std::cout << "\nNormalizedNoise Output: " << std::endl;
    for (const auto& value : normalizedOutput) {
        std::cout << value << std::endl;
    }
    // define a sample to calculate the density and NLL
    std::vector<double> sample = {0.5, -0.5};
    //Calculate the density
    double density = noise.calculateDensity(sample);
    std::cout << "Density: " << density << std::endl;
    // calculate NNL- lose function - overall aim is to control this function
    try {
        double nll = noise.negativeLogLikelihood(sample);
        std::cout <<"NLL: " << nll << std::endl;
    } catch (const std::runtime_error& e){
        std::cout << "Error calculating NLL" << std::endl;
    }
    return density;
    // calculate the entropy
    double entropy = noise.calculateEntropy();
    std::cout << "Entropy: " << entropy << std::endl;

   // create a dataset of sample structs- for noise and tokens
    std::vector<SampleData> dataset;
    for (int i = 0; i < 10; ++i) {
        SampleData sample;
        sample.token_embedding = {
            static_cast<double>(totalTokens), // tokens
            static_cast<double>(uniqueTokens), // unique tokens
            static_cast<double>(totalWords) // total words
            static_cast<double>(totalPunctuation) // total punctuation
        }
        sample.gaussian_noise = noise.generateNoise();
        sample.target_value = static_cast<double>(totalTokens); // target value for regression
        sample.normalized_noise = normalizedOutput; // normalized noise
        sample.density = density; // density value
        sample.nll = nll; // negative log likelihood value
        sample.entropy = entropy; // entropy value
        dataset.push_back(sample);
    }
    // save to file for DiT
    saveSamples(dataset, "sample_data.bin");
   
    return 0;
}

