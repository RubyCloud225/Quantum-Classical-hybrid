// digits spilt to individual tokens
// spilt and store in array

#include "digits.hpp"
#include "utils/logger.hpp"
#include <vector>
#include <string>
#include <cctype>
#include <iostream>

std::vector<std::string> DigitNormaliser::normaliseDigits(const std::string& input, bool debug /*= false*/) {
    std::vector<std::string> tokens;
    for (char ch : input) {
        if (std::isdigit(ch)) {
            tokens.push_back(std::string(1, ch));
        } else if (!std::isspace(ch)) {
            tokens.push_back(std::string(1, ch));
        }
    }
    if (debug) {
        for (const auto& token : tokens) {
            std::cout << token << " " << std::endl;
        }
    }
    Logger::log("Digit normalization completed with " + std::to_string(tokens.size()) + " tokens", LogLevel::INFO, "DigitNormaliser");
    return tokens;
}