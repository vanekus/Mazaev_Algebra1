#include "experiment.h"
#include "matrix.h"
#include "solver.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

void Experiment::runAll(unsigned seed) {
    std::cout << "=== Comparison of solving time for a single system ===\n";
    std::vector<size_t> sizes = {100, 200, 500, 1000};
    compareSingleSolve(sizes, seed);

    std::cout << "\n=== Efficiency with multiple right-hand sides (n=500) ===\n";
    std::vector<size_t> ks = {1, 10, 100};
    multipleRhsEfficiency(500, ks, seed);

    std::cout << "\n=== Accuracy on Hilbert matrix ===\n";
    std::vector<size_t> hilbertSizes = {5, 10, 15};
    hilbertAccuracy(hilbertSizes);
}

void Experiment::compareSingleSolve(const std::vector<size_t>& sizes, unsigned seed) {
    std::vector<std::vector<std::string>> table;
    table.push_back({"n", "Gauss(no pivot)", "Gauss(partial)", "LU (decomp)", "LU (solve)", "LU total"});

    for (size_t n : sizes) {
        Matrix A = Matrix::random(n, n, -1.0, 1.0, seed);
        std::vector<double> b = Matrix::randomVector(n, -1.0, 1.0, seed + 1);

        Timer t1;
        auto x1 = Solver::gaussNoPivot(A, b);
        double timeNoPivot = t1.elapsed();

        Timer t2;
        auto x2 = Solver::gaussPartialPivot(A, b);
        double timePartial = t2.elapsed();

        Timer t3;
        Matrix LU = Solver::luDecompose(A);
        double timeDecomp = t3.elapsed();

        Timer t4;
        auto x3 = Solver::solveLU(LU, b);
        double timeSolve = t4.elapsed();

        double luTotal = timeDecomp + timeSolve;

        table.push_back({
            std::to_string(n),
            std::to_string(timeNoPivot),
            std::to_string(timePartial),
            std::to_string(timeDecomp),
            std::to_string(timeSolve),
            std::to_string(luTotal)
        });
    }

    printTable(table);
}

void Experiment::multipleRhsEfficiency(size_t n, const std::vector<size_t>& ks, unsigned seed) {
    Matrix A = Matrix::random(n, n, -1.0, 1.0, seed);

    std::vector<std::vector<std::string>> table;
    table.push_back({"k", "Gauss(partial) total", "LU total", "LU ratio"});

    for (size_t k : ks) {
        std::vector<std::vector<double>> rhs;
        for (size_t i = 0; i < k; ++i)
            rhs.push_back(Matrix::randomVector(n, -1.0, 1.0, seed + 100 + static_cast<unsigned>(i)));

        Timer tGauss;
        for (size_t i = 0; i < k; ++i) {
            auto x = Solver::gaussPartialPivot(A, rhs[i]);
        }
        double gaussTime = tGauss.elapsed();

        Timer tLU;
        Matrix LU = Solver::luDecompose(A);
        double decompTime = tLU.elapsed();
        Timer tSolve;
        for (size_t i = 0; i < k; ++i) {
            auto x = Solver::solveLU(LU, rhs[i]);
        }
        double solveTime = tSolve.elapsed();
        double luTotal = decompTime + solveTime;

        double ratio = gaussTime / luTotal;

        table.push_back({
            std::to_string(k),
            std::to_string(gaussTime),
            std::to_string(luTotal),
            std::to_string(ratio)
        });
    }

    printTable(table);
}

void Experiment::hilbertAccuracy(const std::vector<size_t>& sizes) {
    std::vector<std::vector<std::string>> table;
    table.push_back({"n", "Method", "Relative Error", "Residual"});

    for (size_t n : sizes) {
        Matrix H = Matrix::hilbert(n);
        std::vector<double> exact(n, 1.0);
        std::vector<double> b = H.multiply(exact);

        // Gauss no pivot
        try {
            auto x = Solver::gaussNoPivot(H, b);
            double relErr = Matrix::norm2(Matrix::subtract(x, exact)) / Matrix::norm2(exact);
            auto residVec = Matrix::subtract(H.multiply(x), b);
            double resid = Matrix::norm2(residVec);
            table.push_back({std::to_string(n), "Gauss(no pivot)", std::to_string(relErr), std::to_string(resid)});
        } catch (const std::exception& e) {
            table.push_back({std::to_string(n), "Gauss(no pivot)", "FAIL", "FAIL"});
        }

        // Gauss partial pivot
        try {
            auto x = Solver::gaussPartialPivot(H, b);
            double relErr = Matrix::norm2(Matrix::subtract(x, exact)) / Matrix::norm2(exact);
            auto residVec = Matrix::subtract(H.multiply(x), b);
            double resid = Matrix::norm2(residVec);
            table.push_back({std::to_string(n), "Gauss(partial)", std::to_string(relErr), std::to_string(resid)});
        } catch (const std::exception& e) {
            table.push_back({std::to_string(n), "Gauss(partial)", "FAIL", "FAIL"});
        }

        // LU
        try {
            Matrix LU = Solver::luDecompose(H);
            auto x = Solver::solveLU(LU, b);
            double relErr = Matrix::norm2(Matrix::subtract(x, exact)) / Matrix::norm2(exact);
            auto residVec = Matrix::subtract(H.multiply(x), b);
            double resid = Matrix::norm2(residVec);
            table.push_back({std::to_string(n), "LU", std::to_string(relErr), std::to_string(resid)});
        } catch (const std::exception& e) {
            table.push_back({std::to_string(n), "LU", "FAIL", "FAIL"});
        }
    }

    printTable(table);
}

void Experiment::printTable(const std::vector<std::vector<std::string>>& table) {
    if (table.empty()) return;

    // Определяем максимальную ширину каждого столбца
    std::vector<size_t> widths(table[0].size(), 0);
    for (const auto& row : table) {
        for (size_t j = 0; j < row.size(); ++j) {
            if (row[j].size() > widths[j]) {
                widths[j] = row[j].size();
            }
        }
    }

    // Печатаем верхнюю границу
    std::cout << "+";
    for (size_t w : widths) {
        std::cout << std::string(w + 2, '-') << "+";
    }
    std::cout << "\n";

    // Печатаем строки таблицы
    for (size_t i = 0; i < table.size(); ++i) {
        std::cout << "|";
        for (size_t j = 0; j < table[i].size(); ++j) {
            std::cout << " " 
                      << std::setw(static_cast<int>(widths[j])) 
                      << std::left 
                      << table[i][j] 
                      << " |";
        }
        std::cout << "\n";

        // После заголовка (первая строка) печатаем разделитель
        if (i == 0) {
            std::cout << "+";
            for (size_t w : widths) {
                std::cout << std::string(w + 2, '-') << "+";
            }
            std::cout << "\n";
        }
    }

    // Печатаем нижнюю границу
    std::cout << "+";
    for (size_t w : widths) {
        std::cout << std::string(w + 2, '-') << "+";
    }
    std::cout << "\n";
}