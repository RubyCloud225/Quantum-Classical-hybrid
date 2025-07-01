#ifndef METASPACE_HPP
#define METASPACE_HPP

#include <string>
#include <iostream>

class MetaspaceNormaliser {
    void setReplacement(char replacement);
    char getReplacement();
    void setPrependScheme(bool enabled);
    bool getPrependScheme();
    std::vector<std::string> pretok(const std::string& input, bool debug);
}
#endif