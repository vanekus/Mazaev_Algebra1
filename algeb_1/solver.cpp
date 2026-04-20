#include "solver.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

static void swapRows(Matrix& A, std::vector<double>& b, size_t r1, size_t r2) {
    if (r1 == r2) return;
    size_t cols = A.cols();
    for (size_t j = 0; j < cols; ++j)
        std::swap(A(r1, j), A(r2, j));
    std::swap(b[r1], b[r2]);
}

std::vector<double> Solver::gaussNoPivot(Matrix A, std::vector<double> b) {
    size_t n = A.rows();
    if (n != A.cols() || n != b.size())
        throw std::invalid_argument("GaussNoPivot: dimension mismatch");

    for (size_t k = 0; k < n; ++k) {
        if (std::abs(A(k, k)) < 1e-12)
            throw std::runtime_error("GaussNoPivot: zero pivot encountered");
        for (size_t i = k + 1; i < n; ++i) {
            double factor = A(i, k) / A(k, k);
            A(i, k) = 0.0;
            for (size_t j = k + 1; j < n; ++j)
                A(i, j) -= factor * A(k, j);
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[i];
        for (size_t j = i + 1; j < n; ++j)
            sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }
    return x;
}

std::vector<double> Solver::gaussPartialPivot(Matrix A, std::vector<double> b) {
    size_t n = A.rows();
    if (n != A.cols() || n != b.size())
        throw std::invalid_argument("GaussPartialPivot: dimension mismatch");

    for (size_t k = 0; k < n; ++k) {
        size_t pivotRow = k;
        double maxVal = std::abs(A(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > maxVal) {
                maxVal = std::abs(A(i, k));
                pivotRow = i;
            }
        }
        if (maxVal < 1e-12)
            throw std::runtime_error("GaussPartialPivot: singular matrix");

        if (pivotRow != k)
            swapRows(A, b, k, pivotRow);

        for (size_t i = k + 1; i < n; ++i) {
            double factor = A(i, k) / A(k, k);
            A(i, k) = 0.0;
            for (size_t j = k + 1; j < n; ++j)
                A(i, j) -= factor * A(k, j);
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[i];
        for (size_t j = i + 1; j < n; ++j)
            sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }
    return x;
}

Matrix Solver::luDecompose(Matrix A) {
    size_t n = A.rows();
    if (n != A.cols())
        throw std::invalid_argument("LU: matrix must be square");

    for (size_t k = 0; k < n; ++k) {
        if (std::abs(A(k, k)) < 1e-12)
            throw std::runtime_error("LU: zero pivot encountered");
        for (size_t i = k + 1; i < n; ++i) {
            double factor = A(i, k) / A(k, k);
            A(i, k) = factor;
            for (size_t j = k + 1; j < n; ++j)
                A(i, j) -= factor * A(k, j);
        }
    }
    return A;
}

std::vector<double> Solver::forwardSubstitution(const Matrix& LU, const std::vector<double>& b) {
    size_t n = LU.rows();
    if (n != b.size())
        throw std::invalid_argument("ForwardSubstitution: dimension mismatch");

    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = b[i];
        for (size_t j = 0; j < i; ++j)
            sum -= LU(i, j) * y[j];
        y[i] = sum;
    }
    return y;
}

std::vector<double> Solver::backwardSubstitution(const Matrix& LU, const std::vector<double>& y) {
    size_t n = LU.rows();
    if (n != y.size())
        throw std::invalid_argument("BackwardSubstitution: dimension mismatch");

    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = y[i];
        for (size_t j = i + 1; j < n; ++j)
            sum -= LU(i, j) * x[j];
        x[i] = sum / LU(i, i);
    }
    return x;
}

std::vector<double> Solver::solveLU(const Matrix& LU, const std::vector<double>& b) {
    std::vector<double> y = forwardSubstitution(LU, b);
    return backwardSubstitution(LU, y);
}