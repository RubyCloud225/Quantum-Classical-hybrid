#ifndef AngleEncoder_hpp
#define AngleEncoder_hpp
#include <vector>
#include <string>

enum class RotationAxis {
    X,
    Y,
    Z
};

struct RotationGate {
    RotationAxis type;
    double angle; // in radians
};

class AngleEncoder {
public:
    // accepts normalized real input in [0, 1]
    AngleEncoder(const std::vector<double>& input, RotationAxis axis = RotationAxis::Y);
    // list of rotation gates
    std::vector<RotationGate> get_gates() const;
    // readable gate 
    void printGates() const;
private:
    std::vector<double> data;
    RotationAxis axis;
    std::vector<RotationGate> encoded;
    void encode();
};

#endif // AngleEncoder_hpp