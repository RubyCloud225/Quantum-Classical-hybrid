// prepending scheme
// replaces all whitespaces with a provided meta char and then splits on this
// meta char

// get replacement
// set replacement
// get spilt
// set spilt
// get prepend scheme
// set prepend scheme

#include "utils/logger.hpp"
#include "metaspace.hpp"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

MetaspaceNormaliser::MetaspaceNormaliser(char replacement, bool prependScheme)
    : replacement_(replacement), prependScheme_(prependScheme) {}

void MetaspaceNormaliser::setReplacement(char replacement, bool prependScheme) {
    replacement_ = replacement;
    prependScheme_ = prependScheme;
    Logger::log("MetaspaceNormaliser replacement set to: " + std::string(1, replacement) +
                     ", prepend scheme: " + (prependScheme ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

char MetaspaceNormaliser::getReplacement() const {
    return replacement_;
    Logger::log("MetaspaceNormaliser getReplacement called, returning: " + std::string(1, replacement_), LogLevel::INFO, __FILE__, __LINE__);
}

void MetaspaceNormaliser::setPrependScheme(bool scheme) {
    prependScheme_ = scheme;
    Logger::log("MetaspaceNormaliser prepend scheme set to: " + std::string(scheme ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

bool MetaspaceNormaliser::getPrependScheme() const {
    return prependScheme_;
    Logger::log("MetaspaceNormaliser getPrependScheme called, returning: " + std::string(prependScheme_ ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

std::vector<std::string> MetaspaceNormaliser::pretok(const std::string& input, bool debug) const {
    // Parallel transformation using OpenMP
    size_t n = input.size();
    std::vector<char> transformedChars(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        transformedChars[i] = (input[i] == ' ') ? replacement_ : input[i];
    }

    std::string transformed(transformedChars.begin(), transformedChars.end());
    if (prependScheme_ && !transformed.empty() && transformed[0] != replacement_) {
        transformed = replacement_ + transformed;
    }

    // Parallel token splitting
    // First, count tokens in parallel for allocation
    std::vector<size_t> split_indices;
    split_indices.push_back(0);
    for (size_t i = 0; i < transformed.size(); ++i) {
        if (transformed[i] == replacement_) {
            split_indices.push_back(i + 1);
        }
    }
    split_indices.push_back(transformed.size() + 1);

    std::vector<std::string> tokens;
    for (size_t idx = 0; idx + 1 < split_indices.size(); ++idx) {
        size_t start = split_indices[idx];
        size_t end = split_indices[idx + 1] - 1;
        if (end > start) {
            std::string token = transformed.substr(start, end - start);
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
    }

    if (debug) {
        #pragma omp critical
        {
            for (const std::string& t : tokens) {
                std::cout << "Metaspace token: " << t << std::endl;
            }
        }
    }
    #pragma omp critical
    Logger::log("MetaspaceNormaliser pretok completed with " + std::to_string(tokens.size()) + " tokens", LogLevel::INFO, __FILE__, __LINE__);
    return tokens;
}