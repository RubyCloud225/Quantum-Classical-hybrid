// show serialize and deserialize functions for the Replace class
// provide the value for regex and replace
// take the pattern and replace every occurance with content
// clone the data
// partial template specialization for regex and replace
// Serialise the data
// Deserialize the data
// return the data

#include "replace.hpp"
#include "utils/logger.hpp"
#include <sstream>
#include <omp.h>

// creates an object and initializes it with the regex pattern and replacement string
Replace::Replace(const std::string& regexPattern, const std::string& replaceWith)
    : regexPattern(regexPattern), replaceWith(replaceWith) {}

// applied a find and replace operation on the content using the regex pattern and replacement string
std::string Replace::applyReplace(const std::string& content) const {
    std::regex re(regexPattern);
    #pragma omp critical
    Logger::log("Applying regex replace with pattern: " + regexPattern + " and replacement: " + replaceWith, LogLevel::INFO, __FILE__, __LINE__);
    return std::regex_replace(content, re, replaceWith); // sequential to be thread-safe
}

// creates a clone of the Replace object with the same regex pattern and replacement string
Replace Replace::clone() const {
    Logger::log("Cloning Replace object with pattern: " + regexPattern + " and replacement: " + replaceWith, LogLevel::INFO, __FILE__, __LINE__);
    return Replace(regexPattern, replaceWith);
}

// serialize the Replace object
std::string Replace::serialise() const {
    Logger::log("Serializing Replace object with pattern: " + regexPattern + " and replacement: " + replaceWith, LogLevel::INFO, __FILE__, __LINE__);
    std::ostringstream oss;
    oss << regexPattern << "\n" << replaceWith;
    return oss.str();
}
// deserialize a Replace object from a string
Replace Replace::deserialise(const std::string& serialisedData) {
    std::istringstream iss(serialisedData);
    std::string pattern, replace;
    Logger::log("Deserializing Replace object from data: " + serialisedData, LogLevel::INFO, __FILE__, __LINE__);
    if (!std::getline(iss, pattern) || !std::getline(iss, replace)) {
        throw std::runtime_error("Failed to deserialize Replace object");
    }
    return Replace(pattern, replace);
}