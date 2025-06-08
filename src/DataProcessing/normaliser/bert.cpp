#include "bert.hpp"
#include <sstream>
#include <locale>
#include <unordered_set>
#include <cwctype>
#include <cctype>
#include <iostream>
#include <exception>
#include <algorithm>
#include <stdexcept>

struct GlobalInitLogger {
    GlobalInitLogger() {
        std::cout << "GlobalInitLogger constructor called" << std::endl;
    }
    ~GlobalInitLogger() {
        std::cout << "GlobalInitLogger destructor called" << std::endl;
    }
};

GlobalInitLogger globalInitLoggerInstance;

static char32_t toLowerAscii(char32_t ch) {
    if (ch >= U'A' && ch <= U'Z') {
        return ch + 32;
    }
    return ch;
}

bool BertNormaliser::isWhitespace(char32_t ch) {
    return ch == 0x0009 || ch == 0x000A || ch == 0x000B || ch == 0x000C ||
           ch == 0x000D || ch == 0x0020 || ch == 0x00A0 || ch == 0x1680 ||
           ch == 0x180E || ch == 0x2000 || ch == 0x2001 || ch == 0x2002 ||
           ch == 0x2003 || ch == 0x2004 || ch == 0x2005 || ch == 0x2006 ||
           ch == 0x2007 || ch == 0x2008 || ch == 0x2009 || ch == 0x200A ||
           ch == 0x202F || ch == 0x205F || ch == 0x3000;
}

bool BertNormaliser::isControl(char32_t ch) {
    return (ch >= 0x0000 && ch <= 0x001F) || (ch >= 0x007F && ch <= 0x009F);
}

bool BertNormaliser::isChineseChar(char32_t ch) {
    return (ch >= 0x4E00 && ch <= 0x9FFF);
}

std::string BertNormaliser::bertCleaning(const std::string& input) {
    std::string output;
    bool inWhitespace = false;
    for (char ch : input) {
        if (isControl(ch)) {
            continue;
        }
        if (isspace(static_cast<unsigned char>(ch))) {
            if (!inWhitespace) {
                output += ' ';
                inWhitespace = true;
            }
        } else {
            output += ch;
            inWhitespace = false;
        }
    }
    size_t start = output.find_first_not_of(' ');
    size_t end = output.find_last_not_of(' ');
    if (start == std::string::npos) {
        return "";
    }
    return output.substr(start, end - start + 1);
}

std::string BertNormaliser::stripAccents(const std::string& input) {
    std::string output;
    for (unsigned char ch : input) {
        switch (ch) {
            case 0xC3:
                continue;
            case 0xA9:
                output += 'e';
                break;
            case 0xA0:
                output += 'a';
                break;
            case 0xA8:
                output += 'e';
                break;
            case 0xAF:
                output += 'i';
                break;
            case 0xB4:
                output += 'o';
                break;
            case 0xB6:
                output += 'o';
                break;
            case 0xB9:
                output += 'u';
                break;
            default:
                output += ch;
                break;
        }
    }
    return output;
}

std::u32string BertNormaliser::utf8ToUtf32(const std::string& input) {
    std::u32string output;
    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = input[i];
        char32_t ch = 0;
        size_t extraBytes = 0;
        if ((c & 0x80) == 0) {
            ch = c;
            extraBytes = 0;
        } else if ((c & 0xE0) == 0xC0) {
            ch = c & 0x1F;
            extraBytes = 1;
        } else if ((c & 0xF0) == 0xE0) {
            ch = c & 0x0F;
            extraBytes = 2;
        } else if ((c & 0xF8) == 0xF0) {
            ch = c & 0x07;
            extraBytes = 3;
        } else {
            throw std::runtime_error("Invalid UTF-8 encoding");
        }
        if (i + extraBytes >= input.size()) {
            throw std::runtime_error("Truncated UTF-8 sequence");
        }
        for (size_t j = 1; j <= extraBytes; ++j) {
            unsigned char cc = input[i + j];
            if ((cc & 0xC0) != 0x80) {
                throw std::runtime_error("Invalid UTF-8 continuation byte");
            }
            ch = (ch << 6) | (cc & 0x3F);
        }
        output.push_back(ch);
        i += extraBytes + 1;
    }
    return output;
}

std::string BertNormaliser::utf32ToUtf8(const std::u32string& input) {
    std::string output;
    for (char32_t ch : input) {
        if (ch <= 0x7F) {
            output.push_back(static_cast<char>(ch));
        } else if (ch <= 0x7FF) {
            output.push_back(static_cast<char>(0xC0 | ((ch >> 6) & 0x1F)));
            output.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0xFFFF) {
            output.push_back(static_cast<char>(0xE0 | ((ch >> 12) & 0x0F)));
            output.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0x10FFFF) {
            output.push_back(static_cast<char>(0xF0 | ((ch >> 18) & 0x07)));
            output.push_back(static_cast<char>(0x80 | ((ch >> 12) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
            output.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
        } else {
            throw std::runtime_error("Invalid UTF-32 code point");
        }
    }
    return output;
}
