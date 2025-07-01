// normalise strings to byte level
#include "byte_level.hpp"
#include <sstream>
#include <iomanip>

std::vector<std::string> ByteNormaliser::pretok(const std::string& input, bool debug) {
    std::ostringstream oss;
    for (unsigned char ch : input) {
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
        if (debug) {
            std::cout << "Byte: " << static_cast<int >( token[0]) << " -> " << token << std::endl;
        }
        tokens.push_back(token);
    }
    return tokens;
}
std::vector<std::string> ByteNormaliser::ByteNormalise(const std::string& input, bool debug) {
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
        output.push_back(token);
    }
    return output;
}


