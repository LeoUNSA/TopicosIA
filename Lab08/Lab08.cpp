#include <bits/stdc++.h>
using namespace std;
using Tensor3D = vector<vector<vector<double>>>;
using Kernel4D = vector<vector<vector<vector<double>>>>;

class ActivationLayer {
private:
  string activation_type;

public:
  ActivationLayer(const string &type = "ReLU") : activation_type(type) {}

  double applyActivation(double x) {
    if (activation_type == "ReLU") {
      return max(0.0, x);
    } else if (activation_type == "Sigmoid") {
      return 1.0 / (1.0 + exp(-x));
    } else if (activation_type == "Tanh") {
      return tanh(x);
    }
    return x;
  }

  Tensor3D forward(const Tensor3D &input) {
    int height = input.size();
    if (height == 0)
      return {};
    int width = input[0].size();
    if (width == 0)
      return {};
    int channels = input[0][0].size();

    Tensor3D output(height,
                    vector<vector<double>>(width, vector<double>(channels)));

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          output[h][w][c] = applyActivation(input[h][w][c]);
        }
      }
    }

    return output;
  }

  void printInfo() {
    cout << "Activation Layer (" << activation_type << ")" << endl;
  }
};

class PoolingLayer {
private:
  int pool_width, pool_height;
  int stride;
  int padding;
  string pooling_type;

public:
  PoolingLayer(int pool_w, int pool_h, int stride = 1, int padding = 0,
               const string &type = "MaxPooling")
      : pool_width(pool_w), pool_height(pool_h), stride(stride),
        padding(padding), pooling_type(type) {}

  Tensor3D applyPadding3D(const Tensor3D &input) {
    if (padding == 0)
      return input;

    int height = input.size();
    if (height == 0)
      return {};
    int width = input[0].size();
    if (width == 0)
      return {};
    int channels = input[0][0].size();

    Tensor3D padded(height + 2 * padding,
                    vector<vector<double>>(width + 2 * padding,
                                           vector<double>(channels, 0.0)));

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          padded[h + padding][w + padding][c] = input[h][w][c];
        }
      }
    }

    return padded;
  }

  Tensor3D forward(const Tensor3D &input) {
    auto padded_input = applyPadding3D(input);

    int input_height = padded_input.size();
    if (input_height == 0)
      return {};
    int input_width = padded_input[0].size();
    if (input_width == 0)
      return {};
    int channels = padded_input[0][0].size();

    int output_height = (input_height - pool_height) / stride + 1;
    int output_width = (input_width - pool_width) / stride + 1;

    Tensor3D output(output_height, vector<vector<double>>(
                                       output_width, vector<double>(channels)));

    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
          int h_start = h * stride;
          int w_start = w * stride;
          int h_end = h_start + pool_height;
          int w_end = w_start + pool_width;

          if (pooling_type == "MaxPooling") {
            double max_val = -numeric_limits<double>::infinity();
            for (int i = h_start; i < h_end; i++) {
              for (int j = w_start; j < w_end; j++) {
                max_val = max(max_val, padded_input[i][j][c]);
              }
            }
            output[h][w][c] = max_val;
          } else if (pooling_type == "AveragePooling") {
            double sum = 0.0;
            int count = 0;
            for (int i = h_start; i < h_end; i++) {
              for (int j = w_start; j < w_end; j++) {
                sum += padded_input[i][j][c];
                count++;
              }
            }
            output[h][w][c] = sum / count;
          }
        }
      }
    }

    return output;
  }

  void printInfo() {
    cout << "Pooling Layer (" << pooling_type << ")" << endl;
    cout << "Pool size: " << pool_width << "×" << pool_height << endl;
    cout << "Stride: " << stride << ", Padding: " << padding << endl;
  }
};

class FlattenLayer {
public:
  vector<double> forward(const Tensor3D &input) {
    vector<double> output;

    int height = input.size();
    if (height == 0)
      return output;
    int width = input[0].size();
    if (width == 0)
      return output;
    int channels = input[0][0].size();

    output.reserve(height * width * channels);

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          output.push_back(input[h][w][c]);
        }
      }
    }

    return output;
  }

  void printInfo() { cout << "Flatten Layer" << endl; }
};

class ConvolutionLayer3D {
private:
  int input_width, input_height, input_channels;
  int filter_width, filter_height;
  int num_filters, padding, stride;
  // Kernels 4D: [num_filtros][filter_height][filter_width][input_channels]
  Kernel4D filters;
  mt19937 generator;
  uniform_real_distribution<double> distribution;

public:
  ConvolutionLayer3D(int input_w, int input_h, int input_c, int filter_w,
                     int filter_h, int num_filters, int padding = 0,
                     int stride = 1)
      : input_width(input_w), input_height(input_h), input_channels(input_c),
        filter_width(filter_w), filter_height(filter_h),
        num_filters(num_filters), padding(padding), stride(stride),
        generator(random_device{}()), distribution(-1.0, 1.0) {

    initializeFilters();
  }

  void initializeFilters() {
    filters.resize(num_filters);

    for (int f = 0; f < num_filters; f++) {
      filters[f].resize(filter_height);
      for (int fh = 0; fh < filter_height; fh++) {
        filters[f][fh].resize(filter_width);
        for (int fw = 0; fw < filter_width; fw++) {
          filters[f][fh][fw].resize(input_channels);
          for (int c = 0; c < input_channels; c++) {
            filters[f][fh][fw][c] = distribution(generator);
          }
        }
      }
    }
  }

  Tensor3D createTensor3D(int width, int height, int channels,
                          double value = 0.0) {
    return Tensor3D(
        height, vector<vector<double>>(width, vector<double>(channels, value)));
  }

  Tensor3D applyPadding3D(const Tensor3D &input) {
    if (padding == 0)
      return input;

    int padded_height = input.size() + 2 * padding;
    int padded_width = input[0].size() + 2 * padding;
    int channels = input[0][0].size();

    auto padded = createTensor3D(padded_width, padded_height, channels, 0.0);

    for (int h = 0; h < input.size(); h++) {
      for (int w = 0; w < input[0].size(); w++) {
        for (int c = 0; c < channels; c++) {
          padded[h + padding][w + padding][c] = input[h][w][c];
        }
      }
    }

    return padded;
  }

  tuple<int, int, int> getOutputDimensions() {
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;

    int output_height = (padded_height - filter_height) / stride + 1;
    int output_width = (padded_width - filter_width) / stride + 1;
    int output_channels = num_filters;

    return {output_width, output_height, output_channels};
  }

  vector<vector<double>> convolveWithFilter3D(const Tensor3D &input,
                                              int filter_index) {

    auto padded_input = applyPadding3D(input);
    auto [output_width, output_height, output_channels] = getOutputDimensions();

    vector<vector<double>> output(output_height,
                                  vector<double>(output_width, 0.0));

    for (int out_h = 0; out_h < output_height; out_h++) {
      for (int out_w = 0; out_w < output_width; out_w++) {
        double sum = 0.0;

        for (int fh = 0; fh < filter_height; fh++) {
          for (int fw = 0; fw < filter_width; fw++) {
            for (int c = 0; c < input_channels; c++) {
              int input_h = out_h * stride + fh;
              int input_w = out_w * stride + fw;

              sum += padded_input[input_h][input_w][c] *
                     filters[filter_index][fh][fw][c];
            }
          }
        }

        output[out_h][out_w] = sum;
      }
    }

    return output;
  }

  Tensor3D convolve3D(const Tensor3D &input) {
    auto [output_width, output_height, output_channels] = getOutputDimensions();
    auto output = createTensor3D(output_width, output_height, output_channels);

    for (int f = 0; f < num_filters; f++) {
      auto feature_map = convolveWithFilter3D(input, f);

      for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
          output[h][w][f] = feature_map[h][w];
        }
      }
    }

    return output;
  }

  void printLayerInfo() {
    cout << "Entrada: " << input_width << "×" << input_height << "×"
         << input_channels << endl;
    cout << "Filtros: " << num_filters << " filtros de " << filter_width << "×"
         << filter_height << "×" << input_channels << endl;
    cout << "Padding: " << padding << ", Stride: " << stride << endl;

    auto [out_w, out_h, out_c] = getOutputDimensions();
    cout << "Salida: " << out_w << "×" << out_h << "×" << out_c << endl;
  }

  void printFilterChannel(int filter_index, int channel) {
    if (filter_index >= num_filters || channel >= input_channels) {
      cout << "Not found" << endl;
      return;
    }

    cout << "Filtro " << filter_index << ", Canal " << channel << ":" << endl;
    for (int h = 0; h < filter_height; h++) {
      for (int w = 0; w < filter_width; w++) {
        cout << setw(8) << fixed << setprecision(3)
             << filters[filter_index][h][w][channel] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  void printFilter3D(int filter_index) {
    if (filter_index >= num_filters) {
      cout << "Not foudnd" << endl;
      return;
    }

    cout << " Filtro " << filter_index << " Completo " << endl;
    for (int c = 0; c < input_channels; c++) {
      printFilterChannel(filter_index, c);
    }
  }

  void printAllFilters3D() {
    for (int f = 0; f < num_filters; f++) {
      printFilter3D(f);
    }
  }
};

void printTensor3D(const Tensor3D &tensor, const string &title = "Tensor 3D") {
  if (tensor.empty())
    return;

  int height = tensor.size();
  int width = tensor[0].size();
  int channels = tensor[0][0].size();

  cout << title << " (" << width << "×" << height << "×" << channels
       << "):" << endl;

  for (int c = 0; c < channels; c++) {
    cout << "Canal " << c << ":" << endl;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        cout << setw(8) << fixed << setprecision(3) << tensor[h][w][c] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
}

Tensor3D createExampleTensor3D(int width, int height, int channels) {
  Tensor3D tensor(height,
                  vector<vector<double>>(width, vector<double>(channels)));

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channels; c++) {
        tensor[h][w][c] = (h * width + w + 1) * (c + 1);
      }
    }
  }

  return tensor;
}

Tensor3D createRGBExample(int width, int height) {
  Tensor3D rgb(height, vector<vector<double>>(width, vector<double>(3)));

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      rgb[h][w][0] = (h + w) % 256;
      rgb[h][w][1] = (h * 2 + w) % 256;
      rgb[h][w][2] = (h + w * 2) % 256;
    }
  }

  return rgb;
}

int main() {
  Tensor3D input = createRGBExample(4, 4);
  printTensor3D(input, "Input Tensor");

  ConvolutionLayer3D conv(4, 4, 3, 3, 3, 2, 1, 1);
  conv.printLayerInfo();
  auto conv_output = conv.convolve3D(input);

  printTensor3D(conv_output, "Convolution Output");

  ActivationLayer relu("ReLU");
  auto relu_output = relu.forward(conv_output);
  printTensor3D(relu_output, "ReLU Output");

  PoolingLayer maxpool(2, 2, 2, 0, "MaxPooling");
  auto pool_output = maxpool.forward(relu_output);
  printTensor3D(pool_output, "MaxPooling Output");

  FlattenLayer flatten;
  auto flat_output = flatten.forward(pool_output);

  cout << "Flatten Output (" << flat_output.size() << " elements):" << endl;
  for (size_t i = 0; i < flat_output.size(); i++) {
    cout << flat_output[i] << " ";
    if ((i + 1) % 10 == 0)
      cout << endl;
  }
  cout << endl;

  return 0;
}
