#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Tokenizer {
    public:
    std::vector<int> tokenize(const std::string& input);
    // count total number of tokens
    std::vector<int> countTokens(const std::string& tokens);
    // count number of unique tokens types
    std::vector<int> countUniqueTokens(const std::string& tokens);
    // count number of sentences
    std::vector<int> countSentences(const std::string& input);
    // count number of words
    std::vector<int> void countWords(const std::string& tokens);
    // count number of characters
    void countCharacters(const std::string& tokens);
    // number of punctuation
    void countPunctuation(const std::string& input);
    // create positional embedding
    std::unordered_map<std::string, std::vector<int>> createPositionalEmbeddings(const std::string& tokens);
    std::vector<std::string> tokens;
    std::string input;
    
};

#endif // TOKENIZER_HPP