#include "LSTMNetwork.h"
#include <fstream>
#include <iostream>

LSTMNetwork::LSTMNetwork(int input_dim, int hidden_dim, int output_dim) {
    cell = LSTMCell(input_dim, hidden_dim);
    W_output = Matrix(output_dim, hidden_dim); W_output.randomize();
    b_output = Matrix(output_dim, 1, 0.0);
}

std::vector<LSTMCellCache> LSTMNetwork::forward_sequence(const std::vector<Matrix>& inputs, Matrix& h_final, Matrix& c_final) {
    std::vector<LSTMCellCache> caches;
    Matrix h(cell.hidden_dim, 1, 0.0);
    Matrix c(cell.hidden_dim, 1, 0.0);

    for (const auto& x : inputs) {
        LSTMCellCache step_cache = cell.forward(x, h, c);
        h = Matrix::hadamard(step_cache.o, step_cache.tanh_c);
        c = step_cache.c;
        caches.push_back(step_cache);
    }
    h_final = h;
    c_final = c;
    return caches;
}

double LSTMNetwork::train_step(const std::vector<Matrix>& inputs, const Matrix& target, double learning_rate) {
    Matrix h_final, c_final;
    std::vector<LSTMCellCache> caches = forward_sequence(inputs, h_final, c_final);

    // Compute explicit Linear Output Target Prediction
    Matrix prediction = Matrix::add(Matrix::multiply(W_output, h_final), b_output);

    // Compute Mean Squared Error (MSE) Loss
    double loss = 0.0;
    Matrix loss_grad(prediction.rows, 1);
    for (int i = 0; i < prediction.rows; ++i) {
        double diff = prediction(i, 0) - target(i, 0);
        loss += 0.5 * diff * diff;
        loss_grad(i, 0) = diff; // dLoss/dPrediction
    }

    // Dense Layer Backpropagation
    dW_output = Matrix::multiply(loss_grad, h_final.transpose());
    db_output = loss_grad;

    Matrix dh_next = Matrix::multiply(W_output.transpose(), loss_grad);
    Matrix dc_next(cell.hidden_dim, 1, 0.0);

    cell.clear_gradients();

    // Execute Backpropagation Through Time (BPTT) loop
    for (int t = static_cast<int>(inputs.size()) - 1; t >= 0; --t) {
        Matrix dh_prev(cell.hidden_dim, 1, 0.0);
        Matrix dc_prev(cell.hidden_dim, 1, 0.0);

        cell.backward(caches[t], dh_next, dc_next, dh_prev, dc_prev);

        dh_next = dh_prev;
        dc_next = dc_prev;
    }

    // Stabilize training with explicit gradient clipping
    cell.clip_gradients(1.0);
    auto clip_lambda = [](double val) { return std::max(std::min(val, 1.0), -1.0); };
    dW_output = dW_output.apply(clip_lambda);
    db_output = db_output.apply(clip_lambda);

    // SGD Parameter Step
    cell.update_parameters(learning_rate);
    for (size_t i = 0; i < W_output.data.size(); ++i) W_output.data[i] -= learning_rate * dW_output.data[i];
    for (size_t i = 0; i < b_output.data.size(); ++i) b_output.data[i] -= learning_rate * db_output.data[i];

    return loss;
}

Matrix LSTMNetwork::predict(const std::vector<Matrix>& inputs) {
    Matrix h_final, c_final;
    forward_sequence(inputs, h_final, c_final);
    return Matrix::add(Matrix::multiply(W_output, h_final), b_output);
}

void LSTMNetwork::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) return;

    auto serialize = [&out](const Matrix& M) {
        out << M.rows << " " << M.cols << "\n";
        for (double v : M.data) out << v << " ";
        out << "\n";
    };

    serialize(cell.Wf); serialize(cell.Wi); serialize(cell.Wc); serialize(cell.Wo);
    serialize(cell.Uf); serialize(cell.Ui); serialize(cell.Uc); serialize(cell.Uo);
    serialize(cell.bf); serialize(cell.bi); serialize(cell.bc); serialize(cell.bo);
    serialize(W_output); serialize(b_output);
}

void LSTMNetwork::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) return;

    auto deserialize = [&in](Matrix& M) {
        int r, c;
        in >> r >> c;
        M = Matrix(r, c);
        for (int i = 0; i < r * c; ++i) in >> M.data[i];
    };

    deserialize(cell.Wf); deserialize(cell.Wi); deserialize(cell.Wc); deserialize(cell.Wo);
    deserialize(cell.Uf); deserialize(cell.Ui); deserialize(cell.Uc); deserialize(cell.Uo);
    deserialize(cell.bf); deserialize(cell.bi); deserialize(cell.bc); deserialize(cell.bo);
    deserialize(W_output); deserialize(b_output);
}