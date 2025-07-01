#ifndef BERT_HPP
#define BERT_HPP

#include <string>

class BertNormaliser {
    public:
    // check for whitespace
    static bool isWhitespace(char32_t ch);
    // check for control characters
    static bool isControl(char32_t ch);
    // check for chinese characters
    static bool isChineseChar(char32_t ch);
    static std::string bertCleaning(const std::string& input);
    static std::string stripAccents(const std::string& input);
    static std::u32string utf8ToUtf32(const std::string& input);
    static std::string utf32ToUtf8(const std::u32string& input);
    static std::vector<std::string> pretok(const std::string& input);
};

#endif // BERT_HPP