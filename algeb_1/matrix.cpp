#include "matrix.h"
#include <stdexcept>
#include <cmath>

Matrix::Matrix() : m_rows(0), m_cols(0) {}

Matrix::Matrix(size_t rows, size_t cols, double init)
    : m_rows(rows), m_cols(cols), m_data(rows * cols, init) {}

Matrix::Matrix(const Matrix& other)
    : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;
    }
    return *this;
}

double& Matrix::operator()(size_t i, size_t j) {
    if (i >= m_rows || j >= m_cols)
        throw std::out_of_range("Matrix: index out of range");
    return m_data[i * m_cols + j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
    if (i >= m_rows || j >= m_cols)
        throw std::out_of_range("Matrix: index out of range");
    return m_data[i * m_cols + j];
}

Matrix Matrix::random(size_t rows, size_t cols, double minVal, double maxVal, unsigned seed) {
    Matrix mat(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(minVal, maxVal);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = dist(gen);
    return mat;
}

Matrix Matrix::hilbert(size_t n) {
    Matrix mat(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            mat(i, j) = 1.0 / static_cast<double>(i + j + 1);
    return mat;
}

std::vector<double> Matrix::randomVector(size_t size, double minVal, double maxVal, unsigned seed) {
    std::vector<double> vec(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(minVal, maxVal);
    for (size_t i = 0; i < size; ++i)
        vec[i] = dist(gen);
    return vec;
}

std::vector<double> Matrix::multiply(const std::vector<double>& vec) const {
    if (vec.size() != m_cols)
        throw std::invalid_argument("Matrix::multiply: vector size mismatch");
    std::vector<double> result(m_rows, 0.0);
    for (size_t i = 0; i < m_rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < m_cols; ++j)
            sum += (*this)(i, j) * vec[j];
        result[i] = sum;
    }
    return result;
}

double Matrix::norm2(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec)
        sum += v * v;
    return std::sqrt(sum);
}

std::vector<double> Matrix::subtract(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Matrix::subtract: vectors have different sizes");
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] - b[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(m_cols, m_rows);
    for (size_t i = 0; i < m_rows; ++i)
        for (size_t j = 0; j < m_cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}