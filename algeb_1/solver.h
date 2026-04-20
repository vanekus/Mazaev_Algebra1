#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include "matrix.h"

class Solver {
public:
    // Метод Гаусса без выбора ведущего элемента
    static std::vector<double> gaussNoPivot(Matrix A, std::vector<double> b);

    // Метод Гаусса с частичным выбором ведущего элемента по столбцу
    static std::vector<double> gaussPartialPivot(Matrix A, std::vector<double> b);

    // LU-разложение (возвращает матрицу, содержащую L и U)
    static Matrix luDecompose(Matrix A);

    // Прямая подстановка: L * y = b
    static std::vector<double> forwardSubstitution(const Matrix& LU, const std::vector<double>& b);

    // Обратная подстановка: U * x = y
    static std::vector<double> backwardSubstitution(const Matrix& LU, const std::vector<double>& y);

    // Решение с использованием LU-разложения
    static std::vector<double> solveLU(const Matrix& LU, const std::vector<double>& b);
};

#endif // SOLVER_H