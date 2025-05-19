#include <bits/stdc++.h>
using namespace std;

#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2
#define ACTIVATION_RELU 3

int ACTIVATION = ACTIVATION_SIGMOID;

double activation(double x) {
  if (ACTIVATION == ACTIVATION_SIGMOID)
    return 1.0 / (1.0 + exp(-x));
  else if (ACTIVATION == ACTIVATION_TANH)
    return tanh(x);
  else if (ACTIVATION == ACTIVATION_RELU)
    return x > 0 ? x : 0;
  return x;
}

double activationDerivative(double x) {
  if (ACTIVATION == ACTIVATION_SIGMOID) {
    double sig = activation(x);
    return sig * (1 - sig);
  } else if (ACTIVATION == ACTIVATION_TANH)
    return 1 - pow(tanh(x), 2);
  else if (ACTIVATION == ACTIVATION_RELU)
    return x > 0 ? 1 : 0;
  return 1;
}
struct Neuron {
  vector<double> weights;
  double bias, output, inputSum; // para derivada
  Neuron(int inputSize) {
    for (int i = 0; i < inputSize; ++i)
      weights.push_back((rand() / double(RAND_MAX)) * 2 - 1);
    bias = (rand() / double(RAND_MAX)) * 2 - 1;
  }

  double feedForward(const vector<double> &inputs) {
    inputSum = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputSum += weights[i] * inputs[i];
    }
    inputSum += bias;
    output = activation(inputSum);
    return output;
  }
};
class MLP {
  vector<Neuron> hiddenLayer;
  Neuron outputNeuron;
  double learningRate;

public:
  MLP() : outputNeuron(2), learningRate(0.1) {
    hiddenLayer.push_back(Neuron(2));
    hiddenLayer.push_back(Neuron(2));
  }

  double train(vector<double> inputs, double target) {
    // Forward pass
    vector<double> hiddenLayerOutputs;
    for (auto &h : hiddenLayer)
      hiddenLayerOutputs.push_back(h.feedForward(inputs));
    double finalOutput = outputNeuron.feedForward(hiddenLayerOutputs);
    double error = target - finalOutput;
    double derivativeOutput =
        error * activationDerivative(outputNeuron.inputSum);
    // Backpropagation - Output
    for (int i = 0; i < outputNeuron.weights.size(); ++i) {
      double delta = learningRate * derivativeOutput * hiddenLayer[i].output;
      outputNeuron.weights[i] += delta;
    }
    outputNeuron.bias += learningRate * derivativeOutput;
    // Backpropagation - Hidden Layer
    for (int i = 0; i < hiddenLayer.size(); ++i) {
      double derivativeHiddenLayer =
          derivativeOutput * outputNeuron.weights[i] *
          activationDerivative(hiddenLayer[i].inputSum);
      for (int j = 0; j < hiddenLayer[i].weights.size(); ++j)
        hiddenLayer[i].weights[j] +=
            learningRate * derivativeHiddenLayer * inputs[j];
      hiddenLayer[i].bias += learningRate * derivativeHiddenLayer;
    }
    return error * error;
  }

  double predict(const vector<double> &inputs) {
    vector<double> hiddenLayerOutputs;
    for (auto &h : hiddenLayer)
      hiddenLayerOutputs.push_back(h.feedForward(inputs));
    return outputNeuron.feedForward(hiddenLayerOutputs);
  }
};
int main() {
  srand(time(0));
  MLP mlp;
  vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<double> targets = {0, 1, 1, 0}; // XOR
  // vector<double> targets = {0, 0, 0, 1}; // AND
  // vector<double> targets = {0, 1, 1, 1}; // OR
  for (int epoch = 0; epoch <= 50000; ++epoch) {
    double loss = 0.0;
    for (int i = 0; i < 4; ++i)
      loss += mlp.train(inputs[i], targets[i]);
    if (epoch % 10000 == 0)
      cout << "Epoch " << epoch << ", Loss: " << loss << endl;
  }
  cout << "Resultados:\n";
  for (int i = 0; i < 4; ++i) {
    double result = mlp.predict(inputs[i]);
    cout << inputs[i][0] << " , " << inputs[i][1] << " = " << result << endl;
  }
  return 0;
}
