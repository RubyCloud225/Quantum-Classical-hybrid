#include <benchmark/benchmark.h>
#include "prepend_benchmark.cpp"
#include "prepend.hpp"
#include <random>
#include <sstream>
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
static void BM_Prepend_NonNumeric(benchmark::State& state) {
    std::string input = "This is a test string with no numbers.";
    auto result = extract_normalised(input);
    assert(result.empty() && "Expected empty result for non-numeric input");
}
static void BM_Prepend_SingleNumber(benchmark::State& state) {
    std::string input = "42";
    auto result = extract_normalised(input);
    assert(result.size() == 1 && "Expected one number in the result");
    assert(result[0] == 42.0 && "Expected the number to be 42.0");
}
static void BM_Prepend_SingleNumberDecimal(benchmark::State& state) {
    std::string input = "3.14";
    auto result = extract_normalised(input);
    assert(result.size() == 1 && "Expected one number in the result");
    assert(result[0] == 3.14 && "Expected the number to be 3.14");
}
static void BM_Prepend_SingleNumberDecimalNegative(benchmark::State& state) {
    std::string input = "-2.718";
    auto result = extract_normalised(input);
    assert(result.size() == 1 && "Expected one number in the result");
    assert(result[0] == -2.718 && "Expected the number to be -2.718");
}
static void BM_Prepend_MultipleNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2.5, and -3.14 in this text.";
    auto result = extract_normalised(input);
    assert(result.size() == 3 && "Expected three numbers in the result");
    assert(result[0] == 1.0 && "Expected the first number to be 1.0");
    assert(result[1] == 2.5 && "Expected the second number to be 2.5");
    assert(result[2] == -3.14 && "Expected the third number to be -3.14");
}
static void BM_Prepend_MalformedFloatingPoints(benchmark::State& state) {
    std::string input = "This is a test with malformed numbers like 1.2.3 and 4.5.6.";
    auto result = extract_normalised(input);
    assert(result.empty() && "Expected empty result for malformed floating points");
}
static void BM_Prepend_DuplicateNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2, 2, and 3 in this text.";
    auto result = extract_normalised(input);
    assert(result.size() == 3 && "Expected three unique numbers in the result");
    assert(result[0] == 1.0 && "Expected the first number to be 1.0");
    assert(result[1] == 2.0 && "Expected the second number to be 2.0");
    assert(result[2] == 3.0 && "Expected the third number to be 3.0");
}
static void BM_Prepend_NegativeNumbers(benchmark::State& state) {
    std::string input = "The numbers are -1, -2.5, and 3.14 in this text.";
    auto result = extract_normalised(input);
    assert(result.size() == 3 && "Expected three numbers in the result");
    assert(result[0] == -1.0 && "Expected the first number to be -1.0");
    assert(result[1] == -2.5 && "Expected the second number to be -2.5");
    assert(result[2] == 3.14 && "Expected the third number to be 3.14");
}
static void BM_Prepend_LeadingAndTrailingSpaces(benchmark::State& state) {
    std::string input = "   42   ";
    auto result = extract_normalised(input);
    assert(result.size() == 1 && "Expected one number in the result");
    assert(result[0] == 42.0 && "Expected the number to be 42.0");
}
static void BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers(benchmark::State& state) {
    std::string input = "   1, 2.5, -3.14   ";
    auto result = extract_normalised(input);
    assert(result.size() == 3 && "Expected three numbers in the result");
    assert(result[0] == 1.0 && "Expected the first number to be 1.0");
    assert(result[1] == 2.5 && "Expected the second number to be 2.5");
    assert(result[2] == -3.14 && "Expected the third number to be -3.14");
}
static void BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints(benchmark::State& state) {
    std::string input = "   1.2.3, 4.5.6   ";
    auto result = extract_normalised(input);
    assert(result.empty() && "Expected empty result for malformed floating points with spaces");
}
=======

#include <benchmark/benchmark.h>
#include "prepend.hpp"
#include <random>
#include <sstream>

// empty input
static void BM_Prepend_Empty(benchmark::State& state) {
    std::string input = "";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.empty() && "Expected empty result for empty input");
    }
}
// only non numeric text
static void BM_Prepend_NonNumeric(benchmark::State& state) {
    std::string input = "This is a test string with no numbers.";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.empty() && "Expected empty result for non-numeric input");
    }
}
static void BM_Prepend_SingleNumber(benchmark::State& state) {
    std::string input = "42";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 1 && "Expected one number in the result");
        assert(result[0] == 42.0 && "Expected the number to be 42.0");
    }
}
static void BM_Prepend_SingleNumberDecimal(benchmark::State& state) {
    std::string input = "3.14";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 1 && "Expected one number in the result");
        assert(result[0] == 3.14 && "Expected the number to be 3.14");
    }
}
static void BM_Prepend_SingleNumberDecimalNegative(benchmark::State& state) {
    std::string input = "-2.718";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 1 && "Expected one number in the result");
        assert(result[0] == -2.718 && "Expected the number to be -2.718");
    }
}
static void BM_Prepend_MultipleNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2.5, and -3.14 in this text.";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 3 && "Expected three numbers in the result");
        assert(result[0] == 1.0 && "Expected the first number to be 1.0");
        assert(result[1] == 2.5 && "Expected the second number to be 2.5");
        assert(result[2] == -3.14 && "Expected the third number to be -3.14");
    }
}
static void BM_Prepend_MalformedFloatingPoints(benchmark::State& state) {
    std::string input = "This is a test with malformed numbers like 1.2.3 and 4.5.6.";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.empty() && "Expected empty result for malformed floating points");
    }
}
static void BM_Prepend_DuplicateNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2, 2, and 3 in this text.";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 3 && "Expected three unique numbers in the result");
        assert(result[0] == 1.0 && "Expected the first number to be 1.0");
        assert(result[1] == 2.0 && "Expected the second number to be 2.0");
        assert(result[2] == 3.0 && "Expected the third number to be 3.0");
    }
}
static void BM_Prepend_NegativeNumbers(benchmark::State& state) {
    std::string input = "The numbers are -1, -2.5, and 3.14 in this text.";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 3 && "Expected three numbers in the result");
        assert(result[0] == -1.0 && "Expected the first number to be -1.0");
        assert(result[1] == -2.5 && "Expected the second number to be -2.5");
        assert(result[2] == 3.14 && "Expected the third number to be 3.14");
    }
}
static void BM_Prepend_LeadingAndTrailingSpaces(benchmark::State& state) {
    std::string input = "   42   ";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 1 && "Expected one number in the result");
        assert(result[0] == 42.0 && "Expected the number to be 42.0");
    }
}
static void BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers(benchmark::State& state) {
    std::string input = "   1, 2.5, -3.14   ";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.size() == 3 && "Expected three numbers in the result");
        assert(result[0] == 1.0 && "Expected the first number to be 1.0");
        assert(result[1] == 2.5 && "Expected the second number to be 2.5");
        assert(result[2] == -3.14 && "Expected the third number to be -3.14");
    }
}
static void BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints(benchmark::State& state) {
    std::string input = "   1.2.3, 4.5.6   ";
    for (auto _ : state) {
        auto result = extract_normalised(input);
        assert(result.empty() && "Expected empty result for malformed floating points with spaces");
    }
}

// New performance test function
static void BM_Prepend_PerformanceTest(benchmark::State& state) {
    // Generate a large input string with many numbers
    std::stringstream ss;
    for (int i = 0; i < 10000; ++i) {
        ss << i << ", ";
    }
    std::string input = ss.str();

    for (auto _ : state) {
        auto result = extract_normalised(input);
        // We expect 10000 numbers
        assert(result.size() == 10000 && "Expected 10000 numbers in the result");
    }
}

// Register benchmarks
BENCHMARK(BM_Prepend_Empty);
BENCHMARK(BM_Prepend_NonNumeric);
BENCHMARK(BM_Prepend_SingleNumber);
BENCHMARK(BM_Prepend_SingleNumberDecimal);
BENCHMARK(BM_Prepend_SingleNumberDecimalNegative);
BENCHMARK(BM_Prepend_MultipleNumbers);
BENCHMARK(BM_Prepend_MalformedFloatingPoints);
BENCHMARK(BM_Prepend_DuplicateNumbers);
BENCHMARK(BM_Prepend_NegativeNumbers);
BENCHMARK(BM_Prepend_LeadingAndTrailingSpaces);
BENCHMARK(BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers);
BENCHMARK(BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints);
BENCHMARK(BM_Prepend_PerformanceTest);

BENCHMARK_MAIN();