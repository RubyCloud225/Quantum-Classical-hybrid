// normalise strings to byte level
#include "byte_level.hpp"
#include <sstream>
#include <iomanip>

std::vector<std::string> ByteNormaliser::ByteNormalise(const std::string& input, bool debug) {
    std::vector<std::string> output;
    for (unsigned char ch : input) {
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

