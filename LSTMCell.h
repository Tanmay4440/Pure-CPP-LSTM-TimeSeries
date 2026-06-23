#ifndef LSTM_CELL_H
#define LSTM_CELL_H

#include "Matrix.h"

// Explicit structure tracking state configurations across sequence time boundaries
struct LSTMCellCache {
    Matrix x;
    Matrix h_prev;
    Matrix c_prev;
    Matrix f;
    Matrix i;
    Matrix c_tilde;
    Matrix c;
    Matrix o;
    Matrix tanh_c;
};

class LSTMCell {
public:
    int input_dim;
    int hidden_dim;

    // Weight and bias parameter arrays
    Matrix Wf, Wi, Wc, Wo;
    Matrix Uf, Ui, Uc, Uo;
    Matrix bf, bi, bc, bo;

    // Gradients accumulated over BPTT passes
    Matrix dWf, dWi, dWc, dWo;
    Matrix dUf, dUi, dUc, dUo;
    Matrix dbf, dbi, dbc, dbo;

    LSTMCell();
    LSTMCell(int input_dim, int hidden_dim);

    LSTMCellCache forward(const Matrix& x, const Matrix& h_prev, const Matrix& c_prev);
    
    void backward(const LSTMCellCache& cache, 
                  const Matrix& dh_next, 
                  const Matrix& dc_next, 
                  Matrix& dh_prev_out, 
                  Matrix& dc_prev_out);

    void clear_gradients();
    void clip_gradients(double threshold);
    void update_parameters(double learning_rate);
};

#endif