#include "tokenizer.hpp"
#include <regex>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include "utils/logger.hpp"

std::vector<std::string> Tokenizer::tokenize(const std::string& input) {
    // Regular expression to match delimiters (spaces, commas, semicolons, periods)
    std::regex delimiter_regex("[ ,;\\.]+");
    std::vector<std::string> tokens;
    std::sregex_token_iterator iter(input.begin(), input.end(), delimiter_regex, -1);
    std::sregex_token_iterator end;
    for (; iter != end; ++iter) {
        if (!iter->str().empty()) {
            tokens.push_back(iter->str());
        }
    }
    Logger::log("Tokenized input into " + std::to_string(tokens.size()) + " tokens", LogLevel::INFO, __FILE__, __LINE__);
    return tokens;
}

int Tokenizer::countTokens(const std::vector<std::string>& tokens) {
    Logger::log("Counting tokens: " + std::to_string(tokens.size()), LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(tokens.size());
}

int Tokenizer::countUniqueTokens(const std::vector<std::string>& tokens) {
    std::unordered_set<std::string> uniqueTokens(tokens.begin(), tokens.end());
    Logger::log("Counting unique tokens: " + std::to_string(uniqueTokens.size()), LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(uniqueTokens.size());
}

int Tokenizer::countSentences(const std::string& input) {
    std::regex sentence_regex("[.!?]+");
    std::sregex_token_iterator iter(input.begin(), input.end(), sentence_regex);
    std::sregex_token_iterator end;
    Logger::log("Counting sentences in input", LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(std::distance(iter, end));
}

int Tokenizer::countWords(const std::vector<std::string>& tokens) {
    Logger::log("Counting words in tokens: " + std::to_string(tokens.size()), LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(tokens.size());
}

int Tokenizer::countCharacters(const std::string& input) {
    Logger::log("Counting characters in input: " + std::to_string(input.size()), LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(input.size());
}

int Tokenizer::countPunctuation(const std::string& input) {
    std::regex punctuation_regex("[^\\w\\s]");
    std::sregex_token_iterator iter(input.begin(), input.end(), punctuation_regex);
    std::sregex_token_iterator end;
    Logger::log("Counting punctuation in input", LogLevel::INFO, __FILE__, __LINE__);
    return static_cast<int>(std::distance(iter, end));
}

std::unordered_map<std::string, std::vector<int>> Tokenizer::createPositionalEmbeddings(const std::vector<std::string>& tokens) {
    std::unordered_map<std::string, std::vector<int>> positionalEmbeddings;
    for (size_t i = 0; i < tokens.size(); ++i) {
        positionalEmbeddings[tokens[i]].push_back(static_cast<int>(i));
    }
    Logger::log("Created positional embeddings for " + std::to_string(positionalEmbeddings.size()) + " unique tokens", LogLevel::INFO, __FILE__, __LINE__);
    return positionalEmbeddings;
}
