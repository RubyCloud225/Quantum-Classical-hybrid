#include <benchmark/benchmark.h>
#include "prepend_benchmark.cpp"
#include "prepend.hpp"
#include <random>
#include <sstream>

// empty input
static void BM_Prepend_Empty(benchmark::State& state) {
    std::string input = "";
    auto result = extract_normalised(input);
    assert(result.empty() && "Expected empty result for empty input");
}
// only non numeric text
// only numbers
// numbers in complex text
// malformed Floating points
// duplicate numbers
// numbers with decimal points
// numbers with decimal points and negative numbers
// non numeric text with numbers
// invalid data points
// benchmark performance
// performance test
