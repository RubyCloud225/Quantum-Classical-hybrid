#include "sampleData.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include "utils/logger.hpp"

void saveSamples(const std::vector<SampleData>& samples, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    for (const auto& sample : samples) {
        size_t size = sample.token_embedding.size();
        size_t noise_size = sample.noise.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(reinterpret_cast<const char*>(sample.token_embedding.data()), size * sizeof(double));
        out.write(reinterpret_cast<const char*>(&noise_size), sizeof(noise_size));
        out.write(reinterpret_cast<const char*>(sample.noise.data()), noise_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(&sample.target_value), sizeof(sample.target_value));
        out.write(reinterpret_cast<const char*>(sample.normalized_noise.data()), sample.normalized_noise.size() * sizeof(double));
        out.write(reinterpret_cast<const char*>(&sample.density), sizeof(sample.density));
        out.write(reinterpret_cast<const char*>(&sample.nll), sizeof(sample.nll));
        out.write(reinterpret_cast<const char*>(&sample.entopy), sizeof(sample.entopy));
    }
    Logger::log("Saved " + std::to_string(samples.size()) + " samples to " + filename, LogLevel::INFO, __FILE__, __LINE__);
    out.close();
}

std::vector<SampleData> loadSamples(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    std::vector<SampleData> samples;
    while (in.peek() != EOF) {
        SampleData sample;
        size_t size, noise_size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        sample.token_embedding.resize(size);
        in.read(reinterpret_cast<char*>(sample.token_embedding.data()), size * sizeof(double));
        in.read(reinterpret_cast<char*>(&noise_size), sizeof(noise_size));
        sample.noise.resize(noise_size);
        in.read(reinterpret_cast<char*>(sample.noise.data()), noise_size * sizeof(double));
        in.read(reinterpret_cast<char*>(&sample.target_value), sizeof(sample.target_value));
        sample.normalized_noise.resize(noise_size);
        in.read(reinterpret_cast<char*>(sample.normalized_noise.data()), noise_size * sizeof(double));
        in.read(reinterpret_cast<char*>(&sample.density), sizeof(sample.density));
        in.read(reinterpret_cast<char*>(&sample.nll), sizeof(sample.nll));
        in.read(reinterpret_cast<char*>(&sample.entopy), sizeof(sample.entopy));
        samples.push_back(sample);
        // Check for read errors
        if (in.fail()) {
            throw std::runtime_error("Error reading sample from file");
        }
        Logger::log("Loaded sample with token embedding size: " + std::to_string(size) + 
                    ", noise size: " + std::to_string(noise_size) + 
                    ", target value: " + std::to_string(sample.target_value), LogLevel::INFO, __FILE__, __LINE__);
    }
    Logger::log("Loaded " + std::to_string(samples.size()) + " samples from " + filename, LogLevel::INFO, __FILE__, __LINE__);
    in.close();
    return samples;
}
