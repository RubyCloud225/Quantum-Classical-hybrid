#include <iostream>
#include "tokenizer.hpp"
#include "GaussianNoise.hpp"
#include "LinearRegression.hpp"
#include "LayerNormalization.hpp"

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

    GaussianNoise noise(mean, covariance);

    // Generate and print Gaussian noise
    std::vector<double> noiseSamples;
    for (int i = 0; i < 10; ++i) {
        std::vector<double> sample = noise.generateNoise();
        std::cout << "Sample " << i + 1 << ": (" << sample[0] << ", " << sample[1] << ")" << std::endl;
    }
    // prepaire data for linear regression
    std::vector<std::pair<double, double>> regressionData;
    for (const auto& noiseValue : noiseSamples) {
        regressionData.emplace_back(noiseValue, (totalTokens));
    }
    // initalise linear regression
    LinearRegression lr;
    // fit the model to the noise created by Gaussian
    lr.fit(regressionData);
    // predict the total tokens
    double x = static_cast<double>(totalTokens);
    double y = lr.predict(x);
    std::cout << "predicted noise value at total tokens = " << x << ":" << y << std::endl;

    // prepare the data for layer normalization
    std::vector<double> Y = {y};
    LayerNormalization layerNorm(Y.size());
    // reset 
    layerNorm.resetParameters();
    // normalize
    std::vector<double> normalizedOutput = layerNorm.forward(Y);
    // Output normalized noise values
    std::cout << "\nNormalizedNoise Output: " << std::endl;
    for (const auto& value : Y) {
        std::cout << value << std::endl;
    }
    
    return 0;
}