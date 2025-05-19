#include "tokenizer.hpp"
#include <iostream>

int main() {
    Tokenizer tokenizer;
    std::string text = "Hello, world! This is a test. Testing, one, two, three.";

    // Test tokenize
    auto tokens = tokenizer.tokenize(text);
    std::cout << "Tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }

    // Test countTokens
    std::cout << "Total tokens: " << tokenizer.countTokens(tokens) << std::endl;

    // Test countUniqueTokens
    std::cout << "Unique tokens: " << tokenizer.countUniqueTokens(tokens) << std::endl;

    // Test countSentences
    std::cout << "Sentences: " << tokenizer.countSentences(text) << std::endl;

    // Test countWords
    std::cout << "Words: " << tokenizer.countWords(tokens) << std::endl;

    // Test countCharacters
    std::cout << "Characters: " << tokenizer.countCharacters(text) << std::endl;

    // Test countPunctuation
    std::cout << "Punctuation: " << tokenizer.countPunctuation(text) << std::endl;

    // Test createPositionalEmbeddings
    auto embeddings = tokenizer.createPositionalEmbeddings(tokens);
    std::cout << "Positional Embeddings:" << std::endl;
    for (const auto& pair : embeddings) {
        std::cout << pair.first << ": ";
        for (int pos : pair.second) {
            std::cout << pos << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
