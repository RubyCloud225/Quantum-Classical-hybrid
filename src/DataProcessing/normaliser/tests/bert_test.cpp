#include <iostream>
#include <cassert>
#include "bert.hpp"

void testWhitespaceNormlisation() {
    std::string input = "   Hello\t\n World!   ";
    std::string expected = "Hello World!";
    std::string result = BertNormaliser::bertCleaning(input);
    assert(result == expected);
}

void testControlCharacterRemoval() {
    std::string input = "Hello\x01World\x02!";
    std::string expected = "HelloWorld!";
    std::string result = BertNormaliser::bertCleaning(input);
    assert(result == expected);
}

void testChineseCharacterHandling() {
    std::string input = "Hello 你好 World!";
    std::string expected = "Hello 你好 World!";
    std::string result = BertNormaliser::bertCleaning(input);
    assert(result == expected);
}

void testAccentStripping() {
    std::string input = "Café résumé naïve";
    std::string expected = "Cafe resume naive";
    std::string result = BertNormaliser::stripAccents(input);
    assert(result == expected);
}

void testMixedInput() {
    std::string input = "  Hello\x01 World! Café résumé 你好  ";
    std::string expected = "Hello World! Cafe resume 你好";
    std::string result = BertNormaliser::bertCleaning(input);
    assert(result == expected);
}

int main() {
    try {
        testWhitespaceNormlisation();
        std::cout << "testWhitespaceNormlisation passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testWhitespaceNormlisation: " << e.what() << std::endl;
    }
    try {
        testControlCharacterRemoval();
        std::cout << "testControlCharacterRemoval passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testControlCharacterRemoval: " << e.what() << std::endl;
    }
    try {
        testChineseCharacterHandling();
        std::cout << "testChineseCharacterHandling passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testChineseCharacterHandling: " << e.what() << std::endl;
    }
    try {
        testAccentStripping();
        std::cout << "testAccentStripping passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testAccentStripping: " << e.what() << std::endl;
    }
    try {
        testMixedInput();
        std::cout << "testMixedInput passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testMixedInput: " << e.what() << std::endl;
    }
    std::cout << "All tests completed" << std::endl;
    return 0;
}
