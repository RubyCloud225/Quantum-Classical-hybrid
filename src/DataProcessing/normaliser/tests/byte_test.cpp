#include "../byte_level.hpp"
#include <cassert>
#include <iostream>
#include <chrono>
#include "utils/logger.hpp"

ByteNormalizer ByteNormalizer;

void testEmptyInput() {
    auto result = ByteNormalizer.ByteNormalise("", true);
    assert(result.empty());
    Logger::log("Empty input test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testSpacesOnly () {
    auto result = ByteNormalizer.ByteNormalise("   \t\n\r", true);
    assert(result.empty());
    Logger::log("Spaces only test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testMixedWhitespaceAndSymbols() {
    std::string input = " \t\n\r!@#";
    auto result = ByteNormalizer.ByteNormalise(input, true);
    assert(result[0] == "Ġ");
    assert(result[4] == "!");
    assert(result[5] == "@");
    assert(result[6] == "#");
    Logger::log("Mixed whitespace and symbols test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testUTF8Characters() {
    std::string input = u8"é漢字";
    auto result = ByteNormalizer.ByteNormalise(input, true);
    assert(!result.empty()); // at least multibyte encoded output
    Logger::log("UTF8 characters test passed", LogLevel::INFO, __FILE__, __LINE__);
}

void testPerformance() {
    std::string input(1000000, 'a');
    auto start = std::chrono::high_resolution_clock::now();
    auto result = ByteNormalizer.ByteNormalise(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Performance test: " << duration.count() << " seconds\n";
    assert(result.size() == input.size());
    Logger::log("Performance test passed", LogLevel::INFO, __FILE__, __LINE__);
}

int main() {
    testEmptyInput();
    testSpacesOnly();
    testMixedWhitespaceAndSymbols();
    testUTF8Characters();
    testPerformance();
    Logger::log("All ByteNormalizer tests passed", LogLevel::INFO, __FILE__, __LINE__);
    return 0;
}
