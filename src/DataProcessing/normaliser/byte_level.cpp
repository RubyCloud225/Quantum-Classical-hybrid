// normalise strings to byte level
#include "byte_level.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include "utils/logger.hpp"

std::vector<std::string> ByteNormalizer::pretok(const std::string& input, bool debug) {
    std::ostringstream oss;
    for (unsigned char ch : input) {
        Logger::log("Processing byte: " + std::to_string(static_cast<int>(ch)), LogLevel::INFO, __FILE__, __LINE__);
        if (ch == ' ') {
            oss << "Ġ";
        } else {
            oss << static_cast<char>(ch);
        }
    }
    std::string byte_str = oss.str();
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(byte_str);
    while (iss >> token) {
        Logger::log("Tokenizing byte: " + token, LogLevel::INFO, __FILE__, __LINE__);
        if (debug) {
            std::cout << "Byte: " << static_cast<int >( token[0]) << " -> " << token << std::endl;
        }
        tokens.push_back(token);
    }
    return tokens;
}
std::vector<std::string> ByteNormalizer::ByteNormalise(const std::string& input, bool debug) {
    std::vector<std::string> output;
    for (unsigned char ch : input) {
        std::string token;
        if (ch == ' ') {
            token = "Ġ";
        } else {
            token = std::string(1, ch);
        }
        if (debug) {
            std::cout << "Byte: " << static_cast<int>(ch) << " -> " << token << std::endl;
        }
        Logger::log("Normalizing byte: " + std::to_string(static_cast<int>(ch)), LogLevel::INFO, __FILE__, __LINE__);
        output.push_back(token);
    }
    return output;
}


