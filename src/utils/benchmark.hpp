#ifndef BENCHMARK_IS_MEASUREMENT_HPP
#define BENCHMARK_IS_MEASUREMENT_HPP

#include <iostream>
#include <chrono>
#include "utils/logger.hpp"

struct MeasurementFlag {
    bool measurement;

    MeasurementFlag() : measurement(false) {}

    void initialize_as_measurement() {
        measurement = true; // Simulate some setup logic
    }

    bool is_measurement() const {
        return measurement;
    }
};

namespace benchmark {

// Utility to prevent compiler optimizations on a value
template<typename T>
struct DoNotOptimize {
    DoNotOptimize(const T& value) {
        asm volatile("" : : "g"(value) : "memory");
    }
};

template<typename T>
inline void do_not_optimize(const T& value) {
    asm volatile("" : : "g"(value) : "memory");
}

struct BenchmarkResult {
    size_t iterations;
    double average_time_us;
};

class State {
public:
    MeasurementFlag flag;
    size_t iterations = 0;
    size_t index = 0;

    void initialize_as_measurement() {
        flag.initialize_as_measurement();
    }

    bool is_measurement() const {
        return flag.is_measurement();
    }

    int thread_index() const {
        return 0; // Single-threaded by default
    }

    // Iterator for range-based for loops
    struct Iterator {
        size_t idx;
        size_t max;

        Iterator(size_t start, size_t max) : idx(start), max(max) {}
        bool operator!=(const Iterator& other) const { return idx < other.max; }
        void operator++() { ++idx; }
        size_t operator*() const { return idx; }
    };

    Iterator begin() { return Iterator(0, iterations); }
    Iterator end() { return Iterator(iterations, iterations); }
};

inline BenchmarkResult benchmark_is_measurement(size_t iterations = 1000000) {
    State state;
    state.iterations = iterations;
    state.initialize_as_measurement();  // Replace with actual setup logic

    volatile bool warmup = state.is_measurement();

    auto start = std::chrono::high_resolution_clock::now();

    for (auto _ : state) {
        volatile bool result = state.is_measurement();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;
    double average_time = duration.count() / iterations;

    Logger::log("Benchmark is_measurement completed", LogLevel::INFO, __FILE__, __LINE__);
    Logger::log("Iterations: {}", LogLevel::ITERATIONS);
    Logger::log("Result: {}", LogLevel::RESULT);
    return {iterations, average_time};
}

} // namespace benchmark

#endif // BENCHMARK_IS_MEASUREMENT_HPP