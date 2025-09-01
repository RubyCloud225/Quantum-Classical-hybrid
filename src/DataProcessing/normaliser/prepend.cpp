// strip normalised values // and prepend them to the original text
#include "prepend.hpp"
#include <string>
#include <regex>
#include <set>
#include <vector>
#include <omp.h>
#include "logger.hpp"

Prepend::Prepend(std::string filename, std::string text) {
    this-> filename = filename;
}

std::set<std::string> Prepend::extract_normalised(const std::string& text) {
    // Parallel regex matching using OpenMP
    std::set<std::string> matches;
    size_t nThreads = omp_get_max_threads();
    std::vector<std::set<std::string>> threadMatches(nThreads);
    std::regex re("\\d+(?:\\.\\d+)?");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::sregex_iterator words_begin(text.begin(), text.end(), re);
        std::sregex_iterator words_end;
        for (auto it = words_begin; it != words_end; ++it) {
            threadMatches[tid].insert(it->str());
        }
    }

    // Merge thread-local sets
    for (const auto& tset : threadMatches) {
        matches.insert(tset.begin(), tset.end());
    }
    Logger::log("Extracted " + std::to_string(matches.size()) + " normalised values from text", LogLevel::INFO, __FILE__, __LINE__);
    return matches;
}

std::string Prepend::build_comment_block(const std::set<std::string>& values) {
    // Parallel concatenation using OpenMP
    int nThreads = omp_get_max_threads();
    std::vector<std::string> threadStrings(nThreads);

    #pragma omp parallel for
    for (size_t i = 0; i < values.size(); ++i) {
        int tid = omp_get_thread_num();
        auto it = values.begin();
        std::advance(it, i);
        threadStrings[tid] += *it + "\n";
    }

    std::string comment_block;
    for (const auto& str : threadStrings) {
        comment_block += str;
    }
    comment_block += "===\n";
    Logger::log("Built comment block with " + std::to_string(values.size()) + " values", LogLevel::INFO, __FILE__, __LINE__);
    return comment_block;
}

void Prepend::write_comment_block(const std::string& text, const std::string& filename) {
    std::set<std::string> normalised_values = extract_normalised(text);
    std::string comment_block = build_comment_block(normalised_values);
    std::ofstream file(filename);
    if (file.is_open()) {
        file << comment_block;
        file.close();
    } else {
        Logger::log("Failed to open file " + filename + " for writing", LogLevel::ERROR, __FILE__, __LINE__);
    }
    Logger::log("Wrote comment block to " + filename, LogLevel::INFO, __FILE__, __LINE__);
}
