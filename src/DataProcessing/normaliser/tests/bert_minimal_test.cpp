#include <iostream>
#include <cassert>
#include <chrono>
#include "bert.hpp"

BertNormaliser normaliser;

void testWhitespaceHandling() {
    // Test trimming and collapsing whitespace
    std::string input = "  Hello   \t\n  World!  ";
    std::string expected = "Hello World!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testWhitespaceHandling passed" << std::endl;

    // Test multiple consecutive spaces
    input = "Hello     World";
    expected = "Hello World";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testWhitespaceHandling multiple spaces passed" << std::endl;

    // Test leading and trailing spaces only
    input = "    Hello World    ";
    expected = "Hello World";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testWhitespaceHandling leading/trailing spaces passed" << std::endl;
}

void testControlCharacterRemoval() {
    // Test removal of control characters
    std::string input = "Hello\x01World\x02!";
    std::string expected = "HelloWorld!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testControlCharacterRemoval passed" << std::endl;

    // Test control characters mixed with whitespace
    input = "Hello\x01 \x02World!";
    expected = "Hello World!";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testControlCharacterRemoval mixed passed" << std::endl;
}

void testChineseCharacterHandling() {
    // Test Chinese characters are preserved
    std::string input = "Hello 你好 World!";
    std::string expected = "Hello 你好 World!";
    std::string result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testChineseCharacterHandling passed" << std::endl;

    // Edge case: Chinese characters only
    input = "你好世界";
    expected = "你好世界";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testChineseCharacterHandling only Chinese passed" << std::endl;

    // Edge case: Chinese characters with control characters
    input = "你\x01好\x02世\x03界";
    expected = "你好世界";
    result = normaliser.bertCleaning(input);
    assert(result == expected);
    std::cout << "testChineseCharacterHandling with control chars passed" << std::endl;
}

void testIsFunctions() {
    // Test isWhitespace
    assert(normaliser.isWhitespace(U' ') == true);
    assert(normaliser.isWhitespace(U'\t') == true);
    assert(normaliser.isWhitespace(U'A') == false);
    std::cout << "testIsWhitespace passed" << std::endl;

    // Test isControl
    assert(normaliser.isControl(U'\x01') == true);
    assert(normaliser.isControl(U'A') == false);
    std::cout << "testIsControl passed" << std::endl;

    // Test isChineseChar
    assert(normaliser.isChineseChar(U'你') == true);
    assert(normaliser.isChineseChar(U'A') == false);
    std::cout << "testIsChineseChar passed" << std::endl;
}

void testUtfConversion() {
    // Test utf8ToUtf32 and utf32ToUtf8 round trip
    std::string utf8Str = u8"Hello 你好";
    std::u32string utf32Str = normaliser.utf8ToUtf32(utf8Str);
    std::string utf8Result = normaliser.utf32ToUtf8(utf32Str);
    assert(utf8Str == utf8Result);
    std::cout << "testUtfConversion round trip passed" << std::endl;

    // Test empty string
    utf8Str = "";
    utf32Str = normaliser.utf8ToUtf32(utf8Str);
    utf8Result = normaliser.utf32ToUtf8(utf32Str);
    assert(utf8Str == utf8Result);
    std::cout << "testUtfConversion empty string passed" << std::endl;
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

    std::cout << "Performance test: 100 runs took " << elapsed.count() << " seconds." << std::endl;
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
        std::cout << "stripAccents output: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in tests: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception in tests" << std::endl;
        return 1;
    }

    return 0;
}
