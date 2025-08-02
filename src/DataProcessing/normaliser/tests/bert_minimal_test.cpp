#include <iostream>
#include <cassert>
#include <chrono>
#include "utils/logger.hpp"
#include "../bert.hpp"

BertNormaliser normaliser;

void testWhitespaceHandling() {
    // Test trimming and collapsing whitespace
    std::string input = "  Hello   \t\n  World!  ";
    std::string expected = "Hello World!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Whitespace handling test passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test multiple consecutive spaces
    input = "Hello     World";
    expected = "Hello World";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Whitespace handling multiple spaces test passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test leading and trailing spaces only
    input = "    Hello World    ";
    expected = "Hello World";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Whitespace handling leading/trailing spaces test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testControlCharacterRemoval() {
    // Test removal of control characters
    std::string input = "Hello\x01World\x02!";
    std::string expected = "HelloWorld!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Control character removal test passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test control characters mixed with whitespace
    input = "Hello\x01 \x02World!";
    expected = "Hello World!";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Control character removal with whitespace test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testChineseCharacterHandling() {
    // Test Chinese characters are preserved
    std::string input = "Hello 你好 World!";
    std::string expected = "Hello 你好 World!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Chinese character handling test passed", LogLevel::INFO, __FILE__, __LINE__);

    // Edge case: Chinese characters only
    input = "你好世界";
    expected = "你好世界";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Chinese character only test passed", LogLevel::INFO, __FILE__, __LINE__);

    // Edge case: Chinese characters with control characters
    input = "你\x01好\x02世\x03界";
    expected = "你好世界";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Chinese characters with control characters test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testIsFunctions() {
    // Test isWhitespace
    assert(normaliser.isWhitespace(U' ') == true);
    assert(normaliser.isWhitespace(U'\t') == true);
    assert(normaliser.isWhitespace(U'A') == false);
    Logger::log("testIsWhitespace passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test isControl
    assert(normaliser.isControl(U'\x01') == true);
    assert(normaliser.isControl(U'A') == false);
    Logger::log("testIsControl passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test isChineseChar
    assert(normaliser.isChineseChar(U'你') == true);
    assert(normaliser.isChineseChar(U'A') == false);
    assert(normaliser.isChineseChar(U'\u4E00') == true); // CJK Unified Ideograph
    assert(normaliser.isChineseChar(U'\u9FFF') == true); // CJK Unified Ideograph Extension A
    assert(normaliser.isChineseChar(U'\u3000') == false); // CJK Symbols and Punctuation
    Logger::log("testIsChineseChar passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testUtfConversion() {
    // Test utf8ToUtf32 and utf32ToUtf8 round trip
    std::string utf8Str = u8"Hello 你好";
    std::u32string utf32Str = normaliser.utf8ToUtf32(utf8Str);
    std::string utf8Result = normaliser.utf32ToUtf8(utf32Str);
    assert(utf8Str == utf8Result);
    Logger::log("testUtfConversion round trip passed", LogLevel::INFO, __FILE__, __LINE__);

    // Test empty string
    utf8Str = "";
    utf32Str = normaliser.utf8ToUtf32(utf8Str);
    utf8Result = normaliser.utf32ToUtf8(utf32Str);
    assert(utf8Str == utf8Result);
    Logger::log("testUtfConversion empty string passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testPerformance() {
    std::string input(1000000, 'a'); // 1 million 'a' characters
    input += " \t\n\r\x01\x02"; // add some whitespace and control chars at the end

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        std::string result = normaliser.bertCleaning(input);
        (void)result; // suppress unused variable warning
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    Logger::log("Performance test completed in " + std::to_string(elapsed.count()) + " seconds", LogLevel::INFO, __FILE__, __LINE__);
}

int main() {
    try {
        testWhitespaceHandling();
        testControlCharacterRemoval();
        testChineseCharacterHandling();
        testIsFunctions();
        testUtfConversion();
        testPerformance();

        std::string input = "Café résumé naïve";
        std::string result = normaliser.stripAccents(input);
        assert(result == "Cafe resume naive");
        Logger::log("Accent stripping test passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in tests: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
        return 1;
    } catch (...) {
        Logger::log("Unknown exception in tests", LogLevel::ERROR, __FILE__, __LINE__);
        return 1;
    }

    return 0;
}