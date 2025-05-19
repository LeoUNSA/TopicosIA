#include <bits/stdc++.h>
using namespace std;

class Perceptron {
private:
  double bias, learningRate;
  vector<double> weights;
  int actFun(double x) { return x >= 0 ? 1 : 0; }

public:
  Perceptron(int inputSize, double lr = 0.5) {
    learningRate = lr;
    bias = rand() % 5;
    weights.resize(inputSize, 0.0);
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = (rand() % 10);
    }
  }
  double wSum(vector<int> inputs) {
    double sum = bias;
    for (int i = 0; i < inputs.size(); i++) {
      sum += weights[i] * inputs[i];
    }
    return sum;
  }
  int predict(vector<int> input) { return actFun(wSum(input)); }
  void train(vector<vector<int>> input, vector<int> target) {
    for (int i = 0; i < input.size(); i++) {
      int prediction = predict(input[i]);
      int error = target[i] - prediction;

      cout << "Input: " << input[i][0] << " , " << input[i][1] << "\n";
      cout << "Valor predecido: " << prediction
           << "\tValor esperado: " << target[i];
      cout << "\tError: " << error << ":\n";

      for (int j = 0; j < weights.size(); j++) {
        weights[j] += learningRate * error * input[i][j];
      }
      bias += learningRate * error;
      cout << "Pesos actualizados: ";
      for (auto i : weights) {
        cout << i << " ";
      }
      cout << "\nBias: " << bias << endl;
    }
  }
  void fit(vector<vector<int>> input, vector<int> target, int epochs) {
    cout << "Pesos iniciales: ";
    for (auto i : weights) {
      cout << i << " ";
    }
    cout << "\t Bias inicial: " << bias << endl;
    for (int i = 0; i < epochs; i++) {
      cout << "\nEpoch - " << i + 1 << ":\n";
      train(input, target);
      cout << endl;
    }
  }
};

int main() {
  srand(time(0));
  vector<vector<int>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<int> andTargets = {0, 0, 0, 1};
  vector<int> orTargets = {1, 1, 1, 0};

  cout << "AND Perceptron:\n";
  Perceptron andPerceptron(2);
  andPerceptron.fit(inputs, andTargets, 10);

  cout << "\n----------------------------------------\n";

  cout << "OR Perceptron:\n";
  Perceptron orPerceptron(2);
  orPerceptron.fit(inputs, orTargets, 10);
}
