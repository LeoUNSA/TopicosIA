#include <bits/stdc++.h>

using namespace std;
using namespace chrono;
vector<vector<double>> readMnistImg(const string &path) {
  ifstream file(path, ios::binary);
  if (!file)
    throw runtime_error("Cannot open file: " + path);

  int magic, numImgs, rows, cols;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&numImgs), 4);
  file.read(reinterpret_cast<char *>(&rows), 4);
  file.read(reinterpret_cast<char *>(&cols), 4);

  // Big Endian -> Little Endian
  magic = __builtin_bswap32(magic);
  numImgs = __builtin_bswap32(numImgs);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  int imgSize = rows * cols;
  vector<vector<double>> images(numImgs, vector<double>(imgSize));

  for (int i = 0; i < numImgs; ++i) {
    vector<unsigned char> buffer(imgSize);
    file.read(reinterpret_cast<char *>(buffer.data()), imgSize);
    for (int j = 0; j < imgSize; ++j) {
      images[i][j] = buffer[j] / 255.0;
    }
  }

  return images;
}

vector<int> readMnistLabels(const string &path) {
  ifstream file(path, ios::binary);
  if (!file)
    throw runtime_error("Cannot open file: " + path);

  int magic, numLabels;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&numLabels), 4);
  magic = __builtin_bswap32(magic);
  numLabels = __builtin_bswap32(numLabels);

  vector<unsigned char> labels(numLabels);
  file.read(reinterpret_cast<char *>(labels.data()), numLabels);

  return vector<int>(labels.begin(), labels.end());
}

class MLP {
public:
  enum Activation { RELU, SIGMOID, TANH, SOFTMAX };

  MLP(vector<int> layers, Activation hidden = RELU, Activation output = SOFTMAX,
      double lr = 0.05, int batch = 32)
      : layer_sizes(layers), hidden_act(hidden), output_act(output),
        learning_rate(lr), batch_size(batch) {

    // Inicializar pesos y biases
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 0.1);

    for (size_t i = 1; i < layers.size(); ++i) {
      vector<vector<double>> w(layers[i], vector<double>(layers[i - 1]));
      vector<double> b(layers[i]);

      for (int j = 0; j < layers[i]; ++j) {
        for (int k = 0; k < layers[i - 1]; ++k) {
          w[j][k] = dist(gen);
        }
        b[j] = dist(gen);
      }

      weights.push_back(w);
      biases.push_back(b);
      outputs.emplace_back(layers[i]);
      errors.emplace_back(layers[i]);
    }
  }

  void train(const vector<vector<double>> &X, const vector<int> &y, int epochs,
             double targetAcc = 1.0,
             const string &logFile = "trainingLog.csv") {

    vector<vector<double>> Y = oneHotEncode(y, 10);
    ofstream logStream(logFile);

    logStream << "epoch,loss,accuracy,time_elapsed\n";

    auto start_time = high_resolution_clock::now();
    bool target_reached = false;

    for (int epoch = 0; epoch < epochs && !target_reached; ++epoch) {
      auto epoch_start = high_resolution_clock::now();
      double loss = 0.0;
      int correct = 0;

      // Entrenamiento por lotes
      for (size_t i = 0; i < X.size(); i += batch_size) {
        size_t end = min(i + batch_size, X.size());

        // Procesar lote
        for (size_t j = i; j < end; ++j) {
          forward(X[j]);
          loss += crossEntropy(Y[j], outputs.back());

          int pred = max_element(outputs.back().begin(), outputs.back().end()) -
                     outputs.back().begin();
          if (pred == y[j])
            correct++;

          backward(Y[j]);
        }

        // Actualizar pesos
        updateWeights();
      }

      // Calcular métricas
      double avg_loss = loss / X.size();
      double accuracy = static_cast<double>(correct) / X.size();
      auto epoch_end = high_resolution_clock::now();
      auto elapsed =
          duration_cast<milliseconds>(epoch_end - epoch_start)
              .count();

      // Guardar logs
      logStream << epoch + 1 << "," << fixed << setprecision(6) << avg_loss
                << "," << accuracy << "," << elapsed << "\n";

      // Mostrar progreso
      cout << "Epoch " << setw(3) << epoch + 1 << " - Loss: " << setw(10)
           << avg_loss << " - Accuracy: " << setw(6) << fixed << setprecision(2)
           << accuracy * 100 << "%"
           << " - Time: " << elapsed << "ms" << endl;

      // Verificar condición de parada
      if (accuracy >= targetAcc) {
        cout << "\n¡Objetivo de precisión alcanzado! (" << accuracy * 100 <<
              "% >= " << targetAcc * 100 << "%)\n"
             << endl;
        target_reached = true;
      }
    }

    auto total_time = duration_cast<seconds>(
                          high_resolution_clock::now() - start_time)
                          .count();
    cout << "Entrenamiento completado. Tiempo total: " << total_time << "s"
         << endl;
  }

  double evaluate(const vector<vector<double>> &X, const vector<int> &y) {
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
      forward(X[i]);
      int pred = max_element(outputs.back().begin(), outputs.back().end()) -
                 outputs.back().begin();
      if (pred == y[i])
        correct++;
    }
    return static_cast<double>(correct) / X.size();
  }

  void save_weights(const string &filename) {
    ofstream file(filename);
    if (!file)
      throw runtime_error("No se pudo abrir el archivo para guardar pesos");

    file << "LayerSizes:";
    for (int size : layer_sizes)
      file << " " << size;
    file << "\n";

    file << fixed << setprecision(15);

    // Guardar pesos y biases por capa
    for (size_t i = 0; i < weights.size(); ++i) {
      file << "Layer " << i + 1 << " Weights:\n";
      for (const auto &neuron_weights : weights[i]) {
        for (double w : neuron_weights)
          file << w << " ";
        file << "\n";
      }

      file << "Layer " << i + 1 << " Biases:\n";
      for (double b : biases[i])
        file << b << " ";
      file << "\n\n";
    }
  }

  void loadWeights(const string &filename) {
    ifstream file(filename);
    if (!file)
      throw runtime_error("No se pudo abrir el archivo de pesos");

    string line;
    // Layer size
    getline(file, line);
    istringstream iss(line);
    string dummy;
    iss >> dummy;

    vector<int> new_sizes;
    int size;
    while (iss >> size)
      new_sizes.push_back(size);

    if (new_sizes != layer_sizes) {
      layer_sizes = new_sizes;
      weights.clear();
      biases.clear();
      outputs.clear();
      errors.clear();

      for (size_t i = 1; i < layer_sizes.size(); ++i) {
        weights.emplace_back(layer_sizes[i],
                             vector<double>(layer_sizes[i - 1]));
        biases.emplace_back(layer_sizes[i]);
        outputs.emplace_back(layer_sizes[i]);
        errors.emplace_back(layer_sizes[i]);
      }
    }

    // Load weights/bias
    for (size_t i = 0; i < weights.size(); ++i) {
      while (getline(file, line) && line.find("Weights:") == string::npos)
        ;

      // Leer pesos
      for (auto &neuron_weights : weights[i]) {
        getline(file, line);
        istringstream weightStream(line);
        for (double &w : neuron_weights)
          weightStream >> w;
      }

      // Saltar línea de encabezado de biases
      while (getline(file, line) && line.find("Biases:") == string::npos)
        ;

      // Leer biases
      getline(file, line);
      istringstream biasStream(line);
      for (double &b : biases[i])
        biasStream >> b;
    }
  }

private:
  vector<int> layer_sizes;
  vector<vector<vector<double>>> weights;
  vector<vector<double>> biases, outputs, errors;
  Activation hidden_act, output_act;
  double learning_rate;
  int batch_size;

  double relu(double x) { return max(0.0, x); }
  double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
  double tanh(double x) { return std::tanh(x); }

  double activate(double x, Activation act) {
    switch (act) {
    case RELU:
      return relu(x);
    case SIGMOID:
      return sigmoid(x);
    case TANH:
      return tanh(x);
    default:
      return relu(x);
    }
  }

  double activateDeriv(double x, Activation act) {
    switch (act) {
    case RELU:
      return x > 0 ? 1 : 0;
    case SIGMOID: {
      double s = sigmoid(x);
      return s * (1 - s);
    }
    case TANH: {
      double t = tanh(x);
      return 1 - t * t;
    }
    default:
      return x > 0 ? 1 : 0;
    }
  }

  vector<vector<double>> oneHotEncode(const vector<int> &labels, int classes) {
    vector<vector<double>> encoded(labels.size(), vector<double>(classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
      encoded[i][labels[i]] = 1.0;
    }
    return encoded;
  }

  void forward(const vector<double> &input) {
    const vector<double> *current = &input;

    for (size_t i = 0; i < weights.size(); ++i) {
      Activation act = (i == weights.size() - 1) ? output_act : hidden_act;

      for (int j = 0; j < layer_sizes[i + 1]; ++j) {
        double sum = biases[i][j];
        for (int k = 0; k < layer_sizes[i]; ++k) {
          sum += weights[i][j][k] * (*current)[k];
        }
        outputs[i][j] = (act == SOFTMAX) ? sum : activate(sum, act);
      }

      if (act == SOFTMAX) {
        double max_val = *max_element(outputs[i].begin(), outputs[i].end());
        double sum = 0.0;
        for (double &val : outputs[i]) {
          val = exp(val - max_val);
          sum += val;
        }
        for (double &val : outputs[i])
          val /= sum;
      }

      current = &outputs[i];
    }
  }

  double crossEntropy(const vector<double> &target,
                      const vector<double> &output) {
    double loss = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
      loss += -target[i] * log(output[i] + 1e-15);
    }
    return loss;
  }

  void backward(const vector<double> &target) {
    // Capa de salida
    for (int i = weights.size() - 1; i >= 0; --i) {
      Activation act =
          (i == static_cast<int>(weights.size()) - 1) ? output_act : hidden_act;

      for (int j = 0; j < layer_sizes[i + 1]; ++j) {
        if (i == static_cast<int>(weights.size()) - 1) {
          errors[i][j] = outputs[i][j] - target[j];
        } else {
          double error = 0.0;
          for (int k = 0; k < layer_sizes[i + 2]; ++k) {
            error += weights[i + 1][k][j] * errors[i + 1][k];
          }
          errors[i][j] = error * activateDeriv(outputs[i][j], act);
        }
      }
    }
  }

  void updateWeights() {
    for (size_t i = 0; i < weights.size(); ++i) {
      for (int j = 0; j < layer_sizes[i + 1]; ++j) {
        for (int k = 0; k < layer_sizes[i]; ++k) {
          double gradient =
              errors[i][j] * ((i == 0) ? outputs[i][j] : outputs[i - 1][k]);
          weights[i][j][k] -= learning_rate * gradient / batch_size;
        }
        biases[i][j] -= learning_rate * errors[i][j] / batch_size;
      }
    }
  }
};

int main() {
  try {
    cout << "Cargando MNIST..." << endl;
    auto train_X = readMnistImg("files/archive/train-images.idx3-ubyte");
    auto train_y = readMnistLabels("files/archive/train-labels.idx1-ubyte");
    auto test_X = readMnistImg("files/archive/t10k-images.idx3-ubyte");
    auto test_y = readMnistLabels("files/archive/t10k-labels.idx1-ubyte");

    cout << "Entrenando MLP" << endl;
    MLP mlp({784, 128, 64, 10}, MLP::RELU, MLP::SOFTMAX, 0.1, 64);

    // Epocas - TargetAcc
    mlp.train(train_X, train_y, 200, 0.95, "trainingLog.csv");

    double accuracy = mlp.evaluate(test_X, test_y);
    cout << "\nPrecisión en test: " << accuracy * 100 << "%" << endl;

    mlp.save_weights("modelWeights.txt");

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
