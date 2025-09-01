// normalise strings to byte level
#include "byte_level.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include "utils/logger.hpp"
#include <omp.h>

std::vector<std::string> ByteNormalizer::pretok(const std::string& input, bool debug) {
    // OpenMP parallel processing
    std::vector<std::string> tokens;
    int nThreads = omp_get_max_threads();
    std::vector<std::vector<std::string>> threadTokens(nThreads);

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        int tid = omp_get_thread_num();
        unsigned char ch = input[i];
        std::string token = (ch == ' ') ? "Ġ" : std::string(1, ch);

        #pragma omp critical
        {
            Logger::log("Processing byte: " + std::to_string(static_cast<int>(ch)), LogLevel::INFO, __FILE__, __LINE__);
            if (debug) {
                std::cout << "Byte: " << static_cast<int>(ch) << " -> " << token << std::endl;
            }
        }
        threadTokens[tid].push_back(token);
    }

    // Merge thread-local vectors
    for (const auto& tvec : threadTokens) {
        tokens.insert(tokens.end(), tvec.begin(), tvec.end());
    }
    return tokens;
}
std::vector<std::string> ByteNormalizer::ByteNormalise(const std::string& input, bool debug) {
    // OpenMP parallel ByteNormalise
    std::vector<std::string> output;
    std::vector<std::vector<std::string>> threadOutput(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        int tid = omp_get_thread_num();
        unsigned char ch = input[i];
        std::string token = (ch == ' ') ? "Ġ" : std::string(1, ch);

        #pragma omp critical
        {
            Logger::log("Normalizing byte: " + std::to_string(static_cast<int>(ch)), LogLevel::INFO, __FILE__, __LINE__);
            if (debug) {
                std::cout << "Byte: " << static_cast<int>(ch) << " -> " << token << std::endl;
            }
        }
        threadOutput[tid].push_back(token);
    }

    for (const auto& tvec : threadOutput) {
        output.insert(output.end(), tvec.begin(), tvec.end());
    }
    return output;
}
