#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Tokenizer {
public:
    // Tokenize input string into vector of tokens (strings)
    std::vector<std::string> tokenize(const std::string& input);

    // Count total number of tokens
    int countTokens(const std::vector<std::string>& tokens);

    // Count number of unique token types
    int countUniqueTokens(const std::vector<std::string>& tokens);

    // Count number of sentences
    int countSentences(const std::string& input);

    // Count number of words
    int countWords(const std::vector<std::string>& tokens);

    // Count number of characters
    int countCharacters(const std::string& input);

    // Count number of punctuation characters
    int countPunctuation(const std::string& input);

    // Create positional embeddings for tokens
    std::unordered_map<std::string, std::vector<int>> createPositionalEmbeddings(const std::vector<std::string>& tokens);

    std::vector<std::string> tokens;
    std::string input;
};

#endif // TOKENIZER_HPP
