#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <vector>
#include <string>

class Experiment {
public:
    static void runAll(unsigned seed = 42);

private:
    static void compareSingleSolve(const std::vector<size_t>& sizes, unsigned seed);
    static void multipleRhsEfficiency(size_t n, const std::vector<size_t>& ks, unsigned seed);
    static void hilbertAccuracy(const std::vector<size_t>& sizes);
    static void printTable(const std::vector<std::vector<std::string>>& table);
};

#endif // EXPERIMENT_H