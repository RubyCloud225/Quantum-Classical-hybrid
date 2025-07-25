#include "byte_level.hpp"
#include <cassert>
#include <iostream>
#include <chrono>

void testEmptyInput() {
    auto result = ByteNormalizer::ByteNormaliser("", true);
    assert(result.empty());
}

void testSpacesOnly () {
    
}

void testMixedWhitespaceAndSymbols() {
    std::string input = " \t\n\r!@#";
    auto result = ByteNormalizer::ByteNormaliser(input, true);
    assert(result[0] == "Ġ");
    assert(result[4] == "!");
}

void testUTF8Characters() {
    std::string input = u8"é漢字";
    auto result = ByteNormalizer::ByteNormaliser(input, true);
    assert(!result.empty()); // at least multibyte encoded output
}

void testPerformance() {
    std::string input(1000000, 'a');
    auto start = std::chrono::high_resolution_clock::now();
    auto result = ByteNormalizer::ByteNormaliser(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Performance test: " << duration.count() << " seconds\n";
    assert(result.size() == input.size());
}

int main() {
    testEmptyInput();
    testSpacesOnly();
    testMixedWhitespaceAndSymbols();
    testUTF8Characters();
    testPerformance();
    std::cout << "All byte-level normaliser tests passed.\n";
    return 0;
}
