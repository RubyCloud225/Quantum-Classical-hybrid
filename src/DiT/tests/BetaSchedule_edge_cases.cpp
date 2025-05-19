#include "../BetaSchedule.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

void testNegativeEpochs() {
    BetaSchedule beta_schedule(-1, 0.1);
    std::vector<double> nll_losses = {0.1};
    std::vector<double> entropy_losses = {0.1};
    double beta = beta_schedule.update(nll_losses, entropy_losses, 0);
    std::cout << "Beta for negative epochs: " << beta << std::endl;
    // Expect beta to be initial_beta since epoch is out of range
    assert(beta == beta_schedule.getCurrentBeta());
    std::cout << "testNegativeEpochs passed\n";
}

void testEpochOutOfRange() {
    BetaSchedule beta_schedule(10, 0.1);
    std::vector<double> nll_losses = {0.1};
    std::vector<double> entropy_losses = {0.1};
    double beta = beta_schedule.update(nll_losses, entropy_losses, 11);
    std::cout << "Beta for out of range epoch: " << beta << std::endl;
    assert(beta == beta_schedule.getCurrentBeta());
    std::cout << "testEpochOutOfRange passed\n";
}

int main() {
    testNegativeEpochs();
    testEpochOutOfRange();
    std::cout << "BetaSchedule edge case tests completed.\n";
    return 0;
}
