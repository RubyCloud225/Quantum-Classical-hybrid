#ifndef BERT_HPP
#define BERT_HPP

#include <string>
#include <vector>

class BertNormaliser {
    public:
    // check for whitespace
    bool isWhitespace(char32_t ch);
    // check for control characters
    bool isControl(char32_t ch);
    // check for chinese characters
    bool isChineseChar(char32_t ch);
    std::string bertCleaning(const std::string& input);
    std::string stripAccents(const std::string& input);
    std::u32string utf8ToUtf32(const std::string& input);
    std::string utf32ToUtf8(const std::u32string& input);
    std::vector<std::string> pretok(const std::string& input);
};

#endif // BERT_HPP