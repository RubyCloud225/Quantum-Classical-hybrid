#include "tokenizer.hpp"
#include <regex>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

std::vector<std::string> Tokenizer::tokenize(const std::string& input) {
    // Regular expression to match delimiters (spaces, Commas, Semicolons)
    std::regex delimiter_regex("[ , ;\\.]+");
    // Vector to hold the tokens
    std::vector<std::string> tokens;
    // tokenize the input string 
    std::sregex_token_iterator iter(input.begin(), input.end(), delimiter_regex, -1);
    std::sregex_token_iterator end;
    // Add the tokens to the vector
    for (; iter != end; ++iter) {
        // only add non-empty tokens
        if (!iter ->str().empty()) {
            tokens.push_back(iter -> str());
        }
    }
    return tokens;
}

int Tokenizer::countToken(const std::vector<std::string> & tokens) {
    std::unordered_set<std::string> uniqueTokens(tokens.begin(), tokens.end());
    return uniqueTokens.size();
}

// Tokenize sentences
int Tokenizer::countSentences(const std::vector<std::string> & input) {
    std::regex sentence_regex("[.!?]+");
    std::sregex_token_iterator iter(input.begin(), input.end(), sentence_regex);
    std::sregex_token_iterator end;
    return std::distance(iter, end);
}

int Tokenizer::countWords(const std::vector<std::string>& tokens) {
    return tokens.size();
}

int Tokenizer::countPunctuation(const std::vector<std::string>& input) {
    std::regex punctuation_regex("[^\\w\\s]");
    std::sregex_token_iterator iter(input.begin(), input.end(), punctuation_regex);
    std::sregex_token_iterator end;
    return std::distance(iter, end);
}

std::unordered_map<std::string, std::vector<int>> Tokenizer::createPositionalEmbeddings(const std::vector<std::string>& tokens) {
    std::unordered_map<std::string, std::vector<int>> positionalEmbeddings;
    for (size_t i = 0; i < tokens.size(); ++i) {
        positionalEmbeddings[tokens[i]].push_back(static_cast<int>(i));
    }
    return positionalEmbeddings;
}
