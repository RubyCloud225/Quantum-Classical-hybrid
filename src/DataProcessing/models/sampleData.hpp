#ifndef SAMPLEDATA_HPP
#define SAMPLEDATA_HPP

#pragma once
#include <vector>
#include <string>

struct PatchToken {
    int row;
    int col;
    std::vector<double> embedding;
};

struct SampleData {
    std::vector<double> token_embedding; // include frequency, positional embedding, and token id
    std::vector<double> noise; // noise vector
    double target_value; // target value for regression
    std::vector<double> normalized_noise; // normalized noise vector
    double density; // density value
    double nll; // negative log likelihood value
    double entopy; // KL divergence value
    std::vector<PatchToken> imageTokens; // image tokens
};

// Function to generate sample data
void saveSamples(const std::vector<SampleData>& samples, const std::string& filename);
// Function to load sample data
std::vector<SampleData> loadSamples(const std::string& filename);

#endif // SAMPLEDATA_HPP
