#  LSTM From Scratch in C++

A pure C++ implementation of an **LSTM (Long Short-Term Memory) neural network** built entirely from scratch using only the STL.

No TensorFlow. No PyTorch. No Eigen.

---

## ✨ Features

- ✅ Custom Matrix class
- ✅ LSTM forward propagation
- ✅ Backpropagation Through Time (BPTT)
- ✅ Gradient clipping
- ✅ SGD optimizer
- ✅ Dense output layer
- ✅ Model save/load
- ✅ Sine-wave forecasting demo
- ✅ CSV generation for visualization
- ✅ Loss vs Epoch plot
- ✅ True vs Predicted plot

---





# Equations

Forget Gate

```math
f_t=\sigma(W_fx_t+U_fh_{t-1}+b_f)
```

Input Gate

```math
i_t=\sigma(W_ix_t+U_ih_{t-1}+b_i)
```

Candidate State

```math
\tilde c_t=\tanh(W_cx_t+U_ch_{t-1}+b_c)
```

Cell State

```math
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
```

Output Gate

```math
o_t=\sigma(W_ox_t+U_oh_{t-1}+b_o)
```

Hidden State

```math
h_t=o_t\odot\tanh(c_t)
```

---

# Training Pipeline

```text
Sequence
   │
   ▼
Forward Pass
   │
   ▼
Prediction
   │
   ▼
MSE Loss
   │
   ▼
BPTT
   │
   ▼
Gradient Accumulation
   │
   ▼
Gradient Clipping
   │
   ▼
SGD Update
   │
   ▼
Next Epoch
```

---

# Project Structure

```text
LSTM-From-Scratch-CPP
│
├── Matrix.h
├── Matrix.cpp
│
├── LSTMCell.h
├── LSTMCell.cpp
│
├── LSTMNetwork.h
├── LSTMNetwork.cpp
│
├── sine_wave_pred.cpp
│
├── loss.csv
├── prediction.csv
│
├── plot_loss.py
├── plot_prediction.py
│
└── README.md
```

---



# Training Example

Input

```text
0.00
0.10
0.20
0.30
...
```

Target

```text
0.91
```


---



Gradients flow backwards through time and update

```text
Wf Wi Wc Wo

Uf Ui Uc Uo

bf bi bc bo
```

---




# Build

```bash
g++ sine_wave_pred.cpp Matrix.cpp LSTMCell.cpp LSTMNetwork.cpp -o sine_wave_pred.exe
```

---

# Run

Windows

```powershell
.\sine_wave_pred.exe
```

Linux

```bash
./sine_wave_pred.exe
```

---

# Example Output

```text
Epoch 50  Loss = 0.00302

Epoch 100 Loss = 0.000416

Epoch 200 Loss = 0.000143

Epoch 500 Loss = 0.0000849
```

Testing

```text
True next value = 0.956376

Predicted value = 0.959757

Absolute error = 0.003381
```

---

# Future Improvements

- Adam optimizer
- Xavier initialization
- Mini-batches
- Gradient checking
- Stock price prediction
- Multi-layer LSTM
- Sequence-to-sequence learning
- Character-level language model
- CSV data loader
- GPU acceleration

---

# Author 
Tanmay Gaurav

Built from scratch in C++ for learning sequence models and time-series forecasting.
