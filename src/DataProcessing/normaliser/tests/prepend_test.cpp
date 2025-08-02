#include "utils/benchmark.hpp"
#include "../prepend.hpp"
#include <random>
#include <sstream>
#include <cassert>
#include "utils/logger.hpp"

Prepend prepend("test.txt", "This is a test string with numbers 1, 2.5, and -3.14.");

void BM_Prepend_Empty(benchmark::State& state) {
    std::string input = "";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.empty() && "Expected empty result for empty input");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_Empty benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_NonNumeric(benchmark::State& state) {
    std::string input = "This is a test string with no numbers.";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.empty() && "Expected empty result for non-numeric input");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_NonNumeric benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_SingleNumber(benchmark::State& state) {
    std::string input = "42";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 1 && "Expected single number result for single number input");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_SingleNumber benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_SingleNumberDecimal(benchmark::State& state) {
    std::string input = "3.14";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 1 && "Expected single number result for single decimal number input");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_SingleNumberDecimal benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_SingleNumberDecimalNegative(benchmark::State& state) {
    std::string input = "-2.718";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 1 && "Expected one number in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_SingleNumberDecimalNegative benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_MultipleNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2.5, and -3.14 in this text.";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 3 && "Expected three numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_MultipleNumbers benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_MalformedFloatingPoints(benchmark::State& state) {
    std::string input = "This is a test with malformed numbers like 1.2.3 and 4.5.6.";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.empty() && "Expected empty result for malformed floating points");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_MalformedFloatingPoints benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_DuplicateNumbers(benchmark::State& state) {
    std::string input = "The numbers are 1, 2, 2, and 3 in this text.";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 3 && "Expected three unique numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_DuplicateNumbers benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_NegativeNumbers(benchmark::State& state) {
    std::string input = "The numbers are -1, -2.5, and 3.14 in this text.";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 3 && "Expected three numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_NegativeNumbers benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_LeadingAndTrailingSpaces(benchmark::State& state) {
    std::string input = "   42   ";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 1 && "Expected one number in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_LeadingAndTrailingSpaces benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers(benchmark::State& state) {
    std::string input = "   1, 2.5, -3.14   ";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 3 && "Expected three numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

void BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints(benchmark::State& state) {
    std::string input = "   1.2.3, 4.5.6   ";
    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.empty() && "Expected empty result for malformed floating points with spaces");
        assert(result.size() == 0 && "Expected no numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

// New performance test function
void BM_Prepend_PerformanceTest(benchmark::State& state) {
    // Generate a large input string with many numbers
    std::stringstream ss;
    for (int i = 0; i < 10000; ++i) {
        ss << i << ", ";
    }
    std::string input = ss.str();

    for (auto _ : state) {
        auto result = prepend.extract_normalised(input);
        benchmark::do_not_optimize(result);
        assert(result.size() == 10000 && "Expected 10000 numbers in the result");
    }
    if (state.is_measurement()) {
        Logger::log("BM_Prepend_PerformanceTest benchmark passed", LogLevel::INFO, __FILE__, __LINE__);
    }
}

int main() {
    benchmark::State state;
    state.iterations = 10000;
    state.initialize_as_measurement();

    BM_Prepend_Empty(state);
    BM_Prepend_NonNumeric(state);
    BM_Prepend_SingleNumber(state);
    BM_Prepend_SingleNumberDecimal(state);
    BM_Prepend_SingleNumberDecimalNegative(state);
    BM_Prepend_MultipleNumbers(state);
    BM_Prepend_MalformedFloatingPoints(state);
    BM_Prepend_DuplicateNumbers(state);
    BM_Prepend_NegativeNumbers(state);
    BM_Prepend_LeadingAndTrailingSpaces(state);
    BM_Prepend_LeadingAndTrailingSpacesMultipleNumbers(state);
    BM_Prepend_LeadingAndTrailingSpacesMalformedFloatingPoints(state);
    BM_Prepend_PerformanceTest(state);

    return 0;
}