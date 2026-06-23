#include "Matrix.h"
#include <random>
#include <stdexcept>
#include <cmath>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int r, int c, double fill_value) : rows(r), cols(c), data(r * c, fill_value) {}

double& Matrix::operator()(int r, int c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(int r, int c) const {
    return data[r * cols + c];
}

void Matrix::randomize(double min_val, double max_val) {
    static std::mt19937 generator(1337); // Fixed seed for reproducible training
    std::uniform_real_distribution<double> distribution(min_val, max_val);
    for (auto& val : data) {
        val = distribution(generator);
    }
}

void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix multiply dimension mismatch: A.cols != B.rows");
    }
    Matrix result(A.rows, B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            double a_val = A(i, k);
            for (int j = 0; j < B.cols; ++j) {
                result(i, j) += a_val * B(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("Matrix add dimension mismatch.");
    }
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i) {
        result.data[i] = A.data[i] + B.data[i];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("Matrix sub dimension mismatch.");
    }
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i) {
        result.data[i] = A.data[i] - B.data[i];
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("Matrix Hadamard dimension mismatch.");
    }
    Matrix result(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i) {
        result.data[i] = A.data[i] * B.data[i];
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]);
    }
    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double activated_val) {
    return activated_val * (1.0 - activated_val);
}

double tanh_derivative(double activated_val) {
    return 1.0 - (activated_val * activated_val);
}