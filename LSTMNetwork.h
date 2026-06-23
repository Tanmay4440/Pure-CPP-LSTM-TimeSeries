#ifndef LSTM_NETWORK_H
#define LSTM_NETWORK_H

#include "LSTMCell.h"
#include <string>

class LSTMNetwork {
public:
    LSTMCell cell;
    Matrix W_output;
    Matrix b_output;

    Matrix dW_output;
    Matrix db_output;

    LSTMNetwork(int input_dim, int hidden_dim, int output_dim);

    std::vector<LSTMCellCache> forward_sequence(const std::vector<Matrix>& inputs, Matrix& h_final, Matrix& c_final);
    
    double train_step(const std::vector<Matrix>& inputs, const Matrix& target, double learning_rate);
    Matrix predict(const std::vector<Matrix>& inputs);

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

#endif