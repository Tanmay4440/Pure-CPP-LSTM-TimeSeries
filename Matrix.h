#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;

    Matrix();
    Matrix(int r, int c, double fill_value = 0.0);

    double& operator()(int r, int c);
    const double& operator()(int r, int c) const;

    void randomize(double min_val = -0.1, double max_val = 0.1);
    void fill(double value);

    static Matrix multiply(const Matrix& A, const Matrix& B);
    static Matrix add(const Matrix& A, const Matrix& B);
    static Matrix subtract(const Matrix& A, const Matrix& B);
    static Matrix hadamard(const Matrix& A, const Matrix& B);
    
    Matrix transpose() const;
    Matrix apply(std::function<double(double)> func) const;

    void print() const;
};

// Common Activation Utilities
double sigmoid(double x);
double sigmoid_derivative(double activated_val); 
double tanh_derivative(double activated_val);   

#endif