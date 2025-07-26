// prepending scheme
// replaces all whitespaces with a provided meta char and then splits on this
// meta char

// get replacement
// set replacement
// get spilt
// set spilt
// get prepend scheme
// set prepend scheme


#include "metaspace.hpp"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

MetaspaceNormaliser::MetaspaceNormaliser(char replacement, bool prependScheme)
    : replacement_(replacement), prependScheme_(prependScheme) {}

void MetaspaceNormaliser::setReplacement(char replacement, bool prependScheme) {
    replacement_ = replacement;
    prependScheme_ = prependScheme;
    Logger::log("MetaspaceNormaliser replacement set to: " + std::string(1, replacement) +
                     ", prepend scheme: " + (prependScheme ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

char MetaspaceNormaliser::getReplacement() const {
    return replacement_;
    Logger::log("MetaspaceNormaliser getReplacement called, returning: " + std::string(1, replacement_), LogLevel::INFO, __FILE__, __LINE__);
}

void MetaspaceNormaliser::setPrependScheme(bool scheme) {
    prependScheme_ = scheme;
    Logger::log("MetaspaceNormaliser prepend scheme set to: " + std::string(scheme ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

bool MetaspaceNormaliser::getPrependScheme() const {
    return prependScheme_;
    Logger::log("MetaspaceNormaliser getPrependScheme called, returning: " + std::string(prependScheme_ ? "true" : "false"), LogLevel::INFO, __FILE__, __LINE__);
}

std::vector<std::string> MetaspaceNormaliser::pretok(const std::string& input, bool debug) const {
    std::string transformed;
    for (char ch : input) {
        if (ch == ' ') {
            transformed += replacement_;
        } else {
            transformed += ch;
        }
    }

    if (prependScheme_ && !transformed.empty() && transformed[0] != replacement_) {
        transformed = replacement_ + transformed;
    }

    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(transformed);
    while (std::getline(iss, token, replacement_)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    if (debug) {
        for (const std::string& t : tokens) {
            std::cout << "Metaspace token: " << t << std::endl;
        }
    }
    Logger::log("MetaspaceNormaliser pretok completed with " + std::to_string(tokens.size()) + " tokens", LogLevel::INFO, __FILE__, __LINE__);
    return tokens;
}