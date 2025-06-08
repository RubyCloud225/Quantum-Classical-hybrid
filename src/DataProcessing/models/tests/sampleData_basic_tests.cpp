#include "../sampleData.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio> // for remove()

bool compareSamples(const SampleData& a, const SampleData& b) {
    if (a.token_embedding != b.token_embedding) return false;
    if (a.noise != b.noise) return false;
    if (a.target_value != b.target_value) return false;
    if (a.normalized_noise != b.normalized_noise) return false;
    if (a.density != b.density) return false;
    if (a.nll != b.nll) return false;
    if (a.entopy != b.entopy) return false;
    return true;
}

void testSaveLoadSamples() {
    std::vector<SampleData> samples;
    SampleData s1;
    s1.token_embedding = {1.0, 2.0, 3.0};
    s1.noise = {0.1, 0.2};
    s1.target_value = 5.0;
    s1.normalized_noise = {0.01, 0.02};
    s1.density = 0.5;
    s1.nll = 0.1;
    s1.entopy = 0.05;
    samples.push_back(s1);

    SampleData s2;
    s2.token_embedding = {4.0, 5.0};
    s2.noise = {0.3, 0.4, 0.5};
    s2.target_value = 10.0;
    s2.normalized_noise = {0.03, 0.04, 0.05};
    s2.density = 0.7;
    s2.nll = 0.2;
    s2.entopy = 0.1;
    samples.push_back(s2);

    std::string filename = "test_samples.bin";
    saveSamples(samples, filename);

    std::vector<SampleData> loadedSamples = loadSamples(filename);

    assert(samples.size() == loadedSamples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        assert(compareSamples(samples[i], loadedSamples[i]));
    }

    std::cout << "SampleData save/load test passed." << std::endl;

    // Clean up
    std::remove(filename.c_str());
}

int main() {
    testSaveLoadSamples();
    return 0;
}
