// strip normalised values // and prepend them to the original text
#include "prepend.hpp"
#include <string>
#include <regex>
#include <set>

Prepend::Prepend(std::string filename, std::string text) {
    this-> filename = filename;
}

std::set<std::string> extract_normalised(const std::string& text) {
    std::set<std::string> matches;
    // match floating-pont numbers- (TODO this is basic)
    std::regex re("\\d+(?:\\.\\d+)?");
    std::regex _iterator it(text.begin(), text.end(), re);
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        matches.insert(match.str());
    }
    return matches;
}

std::string build_commend_block(const std::string& text) {
    std::string commend_block = ""; // === Normalised values ===\n";
    for (const auto& match : extract_normalised(text)) {
        commend_block += match + "\n";
    }
    commend_block += "===\n";
    return commend_block;
}

void write_commend_block(const std::string& text, const std::string& filename) {
    std::string commend_block = build_commend_block(text);
    std::ofstream file(filename);
    if (file.is_open()) {
        file < commend_block;
        file.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }
    std::set<std::string> normalised_values - extract_normalised(text);
    std::string commend_block = build_commend_block(text);
    if (!out) {
        out = std::ofstream(filename);
    }
    prepend(filename, commend_block);
    std::cout << commend_block << std::endl;
}

