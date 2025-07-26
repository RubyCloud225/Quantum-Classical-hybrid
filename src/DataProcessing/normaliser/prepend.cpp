// strip normalised values // and prepend them to the original text
#include "prepend.hpp"
#include <string>
#include <regex>
#include <set>
#include "logger.hpp"

Prepend::Prepend(std::string filename, std::string text) {
    this-> filename = filename;
}

std::set<std::string> Prepend::extract_normalised(const std::string& text) {
    std::set<std::string> matches;
    // match floating-pont numbers- (TODO this is basic)
    std::regex re("\\d+(?:\\.\\d+)?");
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        matches.insert(match.str());
    }
    Logger::log("Extracted " + std::to_string(matches.size()) + " normalised values from text", LogLevel::INFO, __FILE__, __LINE__);
    return matches;
}

std::string Prepend::build_comment_block(const std::set<std::string>& values) {
    std::string comment_block = ""; // === Normalised values ===\n";
    for (const auto& match : values) {
        comment_block += match + "\n";
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
