#ifndef HYPRID_ENCODER_HPP
#define HYPRID_ENCODER_HPP
#include "AngleEncoder.hpp"
#include <vector>

strict HybridGate {
    RotationGate angleGate;
    double amplitude; // Amplitude for the hybrid gate
};

class HybridEncoder {
    public:
    HybridEncoder(std::vector<double>& input);
    // get both angle + amplitude data
    std::vector<HybridGate> getHybridEncodedGates() const;
    void printHybridGates() const;
    private:
    std::vector<HybridGate> hybridEncodedGates;
    void normaliseAndEncode(const std::vector<double>& input);
};
#endif // HYPRID_ENCODER_HPP