#include "LSTMCell.h"
#include <cmath>

LSTMCell::LSTMCell() : input_dim(0), hidden_dim(0) {}

LSTMCell::LSTMCell(int input_dim, int hidden_dim) : input_dim(input_dim), hidden_dim(hidden_dim) {
    // Initialize standard gate projections
    Wf = Matrix(hidden_dim, input_dim); Wf.randomize();
    Wi = Matrix(hidden_dim, input_dim); Wi.randomize();
    Wc = Matrix(hidden_dim, input_dim); Wc.randomize();
    Wo = Matrix(hidden_dim, input_dim); Wo.randomize();

    Uf = Matrix(hidden_dim, hidden_dim); Uf.randomize();
    Ui = Matrix(hidden_dim, hidden_dim); Ui.randomize();
    Uc = Matrix(hidden_dim, hidden_dim); Uc.randomize();
    Uo = Matrix(hidden_dim, hidden_dim); Uo.randomize();

    bf = Matrix(hidden_dim, 1, 0.0);
    bi = Matrix(hidden_dim, 1, 0.0);
    bc = Matrix(hidden_dim, 1, 0.0);
    bo = Matrix(hidden_dim, 1, 0.0);

    clear_gradients();
}

void LSTMCell::clear_gradients() {
    dWf = Matrix(hidden_dim, input_dim, 0.0);
    dWi = Matrix(hidden_dim, input_dim, 0.0);
    dWc = Matrix(hidden_dim, input_dim, 0.0);
    dWo = Matrix(hidden_dim, input_dim, 0.0);

    dUf = Matrix(hidden_dim, hidden_dim, 0.0);
    dUi = Matrix(hidden_dim, hidden_dim, 0.0);
    dUc = Matrix(hidden_dim, hidden_dim, 0.0);
    dUo = Matrix(hidden_dim, hidden_dim, 0.0);

    dbf = Matrix(hidden_dim, 1, 0.0);
    dbi = Matrix(hidden_dim, 1, 0.0);
    dbc = Matrix(hidden_dim, 1, 0.0);
    dbo = Matrix(hidden_dim, 1, 0.0);
}

LSTMCellCache LSTMCell::forward(const Matrix& x, const Matrix& h_prev, const Matrix& c_prev) {
    LSTMCellCache cache;
    cache.x = x;
    cache.h_prev = h_prev;
    cache.c_prev = c_prev;

    // Gate evaluation paths (Wrapped std::tanh in a explicit lambda to fix overload resolution)
    cache.f = Matrix::add(Matrix::add(Matrix::multiply(Wf, x), Matrix::multiply(Uf, h_prev)), bf).apply(sigmoid);
    cache.i = Matrix::add(Matrix::add(Matrix::multiply(Wi, x), Matrix::multiply(Ui, h_prev)), bi).apply(sigmoid);
    cache.c_tilde = Matrix::add(Matrix::add(Matrix::multiply(Wc, x), Matrix::multiply(Uc, h_prev)), bc).apply([](double val) { return std::tanh(val); });
    cache.o = Matrix::add(Matrix::add(Matrix::multiply(Wo, x), Matrix::multiply(Uo, h_prev)), bo).apply(sigmoid);

    // State accumulation operations
    cache.c = Matrix::add(Matrix::hadamard(cache.f, c_prev), Matrix::hadamard(cache.i, cache.c_tilde));
    cache.tanh_c = cache.c.apply([](double val) { return std::tanh(val); });

    return cache;
}

void LSTMCell::backward(const LSTMCellCache& cache, 
                        const Matrix& dh_next, 
                        const Matrix& dc_next, 
                        Matrix& dh_prev_out, 
                        Matrix& dc_prev_out) {

    // 1. Calculate the error gradient for the internal cell state
    Matrix dc = Matrix::add(
        Matrix::hadamard(Matrix::hadamard(dh_next, cache.o), cache.tanh_c.apply(tanh_derivative)),
        dc_next
    );

    // 2. Compute raw pre-activation gate errors
    Matrix df_raw(hidden_dim, 1), di_raw(hidden_dim, 1), dc_tilde_raw(hidden_dim, 1), do_raw(hidden_dim, 1);

    for (int d = 0; d < hidden_dim; ++d) {
        df_raw(d, 0)       = dc(d, 0) * cache.c_prev(d, 0) * sigmoid_derivative(cache.f(d, 0));
        di_raw(d, 0)       = dc(d, 0) * cache.c_tilde(d, 0) * sigmoid_derivative(cache.i(d, 0));
        dc_tilde_raw(d, 0) = dc(d, 0) * cache.i(d, 0) * tanh_derivative(cache.c_tilde(d, 0));
        do_raw(d, 0)       = dh_next(d, 0) * cache.tanh_c(d, 0) * sigmoid_derivative(cache.o(d, 0));
    }

    // 3. Accumulate weight and bias gradients
    Matrix x_T = cache.x.transpose();
    Matrix h_prev_T = cache.h_prev.transpose();

    dWf = Matrix::add(dWf, Matrix::multiply(df_raw, x_T));
    dWi = Matrix::add(dWi, Matrix::multiply(di_raw, x_T));
    dWc = Matrix::add(dWc, Matrix::multiply(dc_tilde_raw, x_T));
    dWo = Matrix::add(dWo, Matrix::multiply(do_raw, x_T));

    dUf = Matrix::add(dUf, Matrix::multiply(df_raw, h_prev_T));
    dUi = Matrix::add(dUi, Matrix::multiply(di_raw, h_prev_T));
    dUc = Matrix::add(dUc, Matrix::multiply(dc_tilde_raw, h_prev_T));
    dUo = Matrix::add(dUo, Matrix::multiply(do_raw, h_prev_T));

    dbf = Matrix::add(dbf, df_raw);
    dbi = Matrix::add(dbi, di_raw);
    dbc = Matrix::add(dbc, dc_tilde_raw);
    dbo = Matrix::add(dbo, do_raw);

    // 4. Propagate error to previous states
    dh_prev_out = Matrix::add(
        Matrix::add(Matrix::multiply(Uf.transpose(), df_raw), Matrix::multiply(Ui.transpose(), di_raw)),
        Matrix::add(Matrix::multiply(Uc.transpose(), dc_tilde_raw), Matrix::multiply(Uo.transpose(), do_raw))
    );

    dc_prev_out = Matrix::hadamard(dc, cache.f);
}

void LSTMCell::clip_gradients(double threshold) {
    auto clip_lambda = [threshold](double val) {
        if (val > threshold) return threshold;
        if (val < -threshold) return -threshold;
        return val;
    };
    dWf = dWf.apply(clip_lambda); dWi = dWi.apply(clip_lambda); dWc = dWc.apply(clip_lambda); dWo = dWo.apply(clip_lambda);
    dUf = dUf.apply(clip_lambda); dUi = dUi.apply(clip_lambda); dUc = dUc.apply(clip_lambda); dUo = dUo.apply(clip_lambda);
    dbf = dbf.apply(clip_lambda); dbi = dbi.apply(clip_lambda); dbc = dbc.apply(clip_lambda); dbo = dbo.apply(clip_lambda);
}

void LSTMCell::update_parameters(double learning_rate) {
    auto update = [learning_rate](const Matrix& param, const Matrix& grad) {
        Matrix out(param.rows, param.cols);
        for(size_t i = 0; i < param.data.size(); ++i) {
            out.data[i] = param.data[i] - learning_rate * grad.data[i];
        }
        return out;
    };

    Wf = update(Wf, dWf); Wi = update(Wi, dWi); Wc = update(Wc, dWc); Wo = update(Wo, dWo);
    Uf = update(Uf, dUf); Ui = update(Ui, dUi); Uc = update(Uc, dUc); Uo = update(Uo, dUo);
    bf = update(bf, dbf); bi = update(bi, dbi); bc = update(bc, dbc); bo = update(bo, dbo);
}