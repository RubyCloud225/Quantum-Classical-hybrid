#ifndef METASPACE_HPP
#define METASPACE_HPP

#include <string>
#include <iostream>
#include <vector>

class MetaspaceNormaliser {
    public:
    MetaspaceNormaliser(char replacement = '_', bool prependScheme = false);
    void setReplacement(char replacement, bool prependScheme = false);
    char getReplacement() const;
    void setPrependScheme(bool scheme);
    bool getPrependScheme() const;
    std::vector<std::string> pretok(const std::string& input, bool debug) const;
    private:
    char replacement_; // Default replacement character
    bool prependScheme_; // Whether to prepend the replacement character
};
#endif