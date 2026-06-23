#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "Matrix.h"
#include "LSTMNetwork.h"

using namespace std;

int main()
{
    cout << "===== Sine Wave Prediction Test =====\n\n";

    // Network Parameters
    int input_dim = 1;
    int hidden_dim = 16;
    int output_dim = 1;
    int sequence_length = 20;

    double learning_rate = 0.01;
    int epochs = 500;
    double step = 0.1;

    LSTMNetwork net(input_dim, hidden_dim, output_dim);

    vector<pair<vector<Matrix>, Matrix>> dataset;

    for(double start = 0.0; start < 20.0; start += 0.1)
    {
        vector<Matrix> sequence;

        for(int i = 0; i < sequence_length; i++)
        {
            double x = start + i * step;

            Matrix input(1,1);
            input(0,0) = sin(x);

            sequence.push_back(input);
        }

        double target_x = start + sequence_length * step;

        Matrix target(1,1);
        target(0,0) = sin(target_x);

        dataset.push_back({sequence, target});
    }


    // Loss CSV

    ofstream loss_file("loss.csv");
    loss_file << "epoch,loss\n";


    // Training

    cout << "Training...\n";

    for(int epoch = 1; epoch <= epochs; epoch++)
    {
        double total_loss = 0.0;

        for(auto &sample : dataset)
        {
            total_loss += net.train_step(
                sample.first,
                sample.second,
                learning_rate
            );
        }

        double avg_loss = total_loss / dataset.size();

        loss_file << epoch << "," << avg_loss << "\n";

        if(epoch % 50 == 0)
        {
            cout << "Epoch "
                 << epoch
                 << " Loss = "
                 << avg_loss
                 << endl;
        }
    }

    loss_file.close();


    // Prediction CSV

    cout << "\n===== Testing =====\n";

    ofstream pred_file("prediction.csv");

    pred_file << "x,true,predicted\n";

    double total_error = 0.0;
    int count = 0;

    for(double start = 25.0; start < 35.0; start += 0.1)
    {
        vector<Matrix> test_sequence;

        for(int i = 0; i < sequence_length; i++)
        {
            Matrix x(1,1);
            x(0,0) = sin(start + i * step);

            test_sequence.push_back(x);
        }

        double true_value =
            sin(start + sequence_length * step);

        Matrix prediction =
            net.predict(test_sequence);

        double pred_value = prediction(0,0);

        pred_file
            << fixed
            << setprecision(8)
            << start + sequence_length * step
            << ","
            << true_value
            << ","
            << pred_value
            << "\n";

        total_error += abs(pred_value - true_value);
        count++;
    }

    pred_file.close();


    // Single Example

    double test_start = 25.0;

    vector<Matrix> example_sequence;

    for(int i = 0; i < sequence_length; i++)
    {
        Matrix x(1,1);
        x(0,0) = sin(test_start + i * step);

        example_sequence.push_back(x);
    }

    double true_value =
        sin(test_start + sequence_length * step);

    Matrix prediction =
        net.predict(example_sequence);

    cout << "\nTrue next value = "
         << true_value
         << endl;

    cout << "Predicted value = "
         << prediction(0,0)
         << endl;

    cout << "Absolute error = "
         << abs(prediction(0,0) - true_value)
         << endl;

    cout << "\nAverage test error = "
         << total_error / count
         << endl;

    cout << "\nGenerated:\n";
    cout << "loss.csv\n";
    cout << "prediction.csv\n";

    return 0;
}