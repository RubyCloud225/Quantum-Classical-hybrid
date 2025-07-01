#ifndef BYTE_LEVEL_HPP
#define BYTE_LEVEL_HPP

#include <string>
#include <vector>

class ByteNormalizer {
    public:
    // converts input string to list of visible normalized byte chars
    static std::vector<std::string> ByteNormalise(const std::string& input, bool debug = false);
    std::vector<std::string> pretok(const std::string& input, bool debug = false);
};
#endif // BYTE_LEVEL_HPP