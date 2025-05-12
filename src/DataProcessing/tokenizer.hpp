#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Tokenizer {
    public:
    std::vector<std::string> tokenize(const std::string& input);
    // count total number of tokens
    int countTokens(const std::vector<std::string>& tokens);
    // count number of unique tokens types
    int countUniqueTokens(const std::vector<std::string>& tokens);
    // count number of sentences
    int countSentences(const std::vector<std::string>& tokens);
    // count number of words
    int countWords(const std::vector<std::string>& tokens);
    // count number of characters
    int countCharacters(const std::vector<std::string>& tokens);
    // number of punctuation
    int countPunctuation(const std::string & input);
    // create positional embedding
    std::unordered_map<std::string, std::vector<int>> createPositionalEmbeddings(const std::vector<std::string>& tokens);
    private:
    std::vector<std::string> tokens_;
    std::string input_;
};

#endif // TOKENIZER_HPP