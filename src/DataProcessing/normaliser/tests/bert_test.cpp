#include <iostream>
#include <cassert>
#include "../bert.hpp"
#include "utils/logger.hpp"

BertNormaliser BertNormaliser;

void testWhitespaceNormlisation() {
    std::string input = "   Hello\t\n World!   ";
    std::string expected = "Hello World!";
    std::string result = BertNormaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Whitespace normalisation test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testControlCharacterRemoval() {
    std::string input = "Hello\x01World\x02!";
    std::string expected = "HelloWorld!";
    std::string result = BertNormaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Control character removal test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testChineseCharacterHandling() {
    std::string input = "Hello 你好 World!";
    std::string expected = "Hello 你好 World!";
    std::string result = BertNormaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Chinese character handling test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testAccentStripping() {
    std::string input = "Café résumé naïve";
    std::string expected = "Cafe resume naive";
    std::string result = BertNormaliser.stripAccents(input);
    assert(result == expected);
    Logger::log("Accent stripping test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testMixedInput() {
    std::string input = "  Hello\x01 World! Café résumé 你好  ";
    std::string expected = "Hello World! Cafe resume 你好";
    std::string result = BertNormaliser.bertCleaning(input);
    assert(result == expected);
    Logger::log("Mixed input test passed", LogLevel::INFO, __FILE__, __LINE__);
}

int main() {
    try {
        testWhitespaceNormlisation();
        Logger::log("testWhitespaceNormlisation passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in testWhitespaceNormlisation: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
    }
    try {
        testControlCharacterRemoval();
        Logger::log("testControlCharacterRemoval passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in testControlCharacterRemoval: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
    }
    try {
        testChineseCharacterHandling();
        Logger::log("testChineseCharacterHandling passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in testChineseCharacterHandling: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
    }
    try {
        testAccentStripping();
        Logger::log("testAccentStripping passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in testAccentStripping: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
    }
    try {
        testMixedInput();
        Logger::log("testMixedInput passed", LogLevel::INFO, __FILE__, __LINE__);
    } catch (const std::exception& e) {
        Logger::log("Exception in testMixedInput: " + std::string(e.what()), LogLevel::ERROR, __FILE__, __LINE__);
    }
    Logger::log("All tests completed", LogLevel::INFO, __FILE__, __LINE__);
    return 0;
}