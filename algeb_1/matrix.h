#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>

class Matrix {
public:
    // Конструкторы
    Matrix();
    Matrix(size_t rows, size_t cols, double init = 0.0);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix() = default;

    // Доступ к элементам
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

    // Генерация матриц
    static Matrix random(size_t rows, size_t cols, double minVal = -1.0, double maxVal = 1.0, unsigned seed = 42);
    static Matrix hilbert(size_t n);
    static std::vector<double> randomVector(size_t size, double minVal = -1.0, double maxVal = 1.0, unsigned seed = 42);

    // Операции с векторами
    std::vector<double> multiply(const std::vector<double>& vec) const;
    static double norm2(const std::vector<double>& vec);
    static std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b);

    // Вспомогательные функции
    Matrix transpose() const;

private:
    size_t m_rows, m_cols;
    std::vector<double> m_data; // хранение по строкам
};

#endif // MATRIX_H