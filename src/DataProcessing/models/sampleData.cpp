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
        // Image tokens
        size_t image_tokens_count = sample.imageTokens.size();
        out.write(reinterpret_cast<const char*>(&image_tokens_count), sizeof(image_tokens_count));
        for (const auto& token : sample.imageTokens) {
            out.write(reinterpret_cast<const char*>(&token.row), sizeof(token.row));
            out.write(reinterpret_cast<const char*>(&token.col), sizeof(token.col));
            int embedding_size = static_cast<int>(token.embedding.size());
            out.write(reinterpret_cast<const char*>(&embedding_size), sizeof(embedding_size));
            out.write(reinterpret_cast<const char*>(token.embedding.data()), embedding_size * sizeof(double));
        }
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
        // read imageTokens
        size_t image_tokens_count;
        in.read(reinterpret_cast<char*>(&image_tokens_count), sizeof(image_tokens_count));
        sample.imageTokens.resize(image_tokens_count);
        for (size_t i = 0; i < image_tokens_count; ++i) {
            PatchToken token;
            in.read(reinterpret_cast<char*>(&token.row), sizeof(token.row));
            in.read(reinterpret_cast<char*>(&token.col), sizeof(token.col));
            int embedding_size;
            in.read(reinterpret_cast<char*>(&embedding_size), sizeof(embedding_size));
            token.embedding.resize(embedding_size);
            in.read(reinterpret_cast<char*>(token.embedding.data()), embedding_size * sizeof(double));
            sample.imageTokens[i] = std::move(token);
        }
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
