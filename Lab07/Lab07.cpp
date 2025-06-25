#include <bits/stdc++.h>
using namespace std;
using Tensor3D = vector<vector<vector<double>>>;
using Kernel4D = vector<vector<vector<vector<double>>>>;

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

    // Copiar al centro
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

    // Convolusion
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

  // Mostrar información de la capa
  //
  void printLayerInfo() {
    cout << "Entrada: " << input_width << "×" << input_height << "×"<<
          input_channels << endl;
    cout << "Filtros: " << num_filters << " filtros de " << filter_width
         << "×"  <<
        filter_height << "×" << input_channels << endl;
    cout << "Padding: " << padding << ", Stride: " << stride << endl;

    auto [out_w, out_h, out_c] = getOutputDimensions();
    cout << "Salida: " << out_w << "×" << out_h << "×" << out_c << endl;
  }

  // Mostrar un canal específico de un filtro
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

  // Mostrar un filtro completo (todos los canales)
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

  cout << title << " (" << width << "×" << height << "×" << channels <<
      "):" << endl;

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
        // Patrón diferente para cada canal
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
      rgb[h][w][0] = (h + w) % 256;     // Canal R
      rgb[h][w][1] = (h * 2 + w) % 256; // Canal G
      rgb[h][w][2] = (h + w * 2) % 256; // Canal B
    }
  }

  return rgb;
}

int main() {
  cout << "Ejemplo - 1: Convolución 3D Básica " << endl;
  int width = 5, height = 5, channels = 3;
  auto input_tensor = createExampleTensor3D(width, height, channels);

  printTensor3D(input_tensor, "Tensor de Entrada");

  ConvolutionLayer3D conv1(width, height, channels, 3, 3, 2);
  conv1.printLayerInfo();

  cout << "Ejemplo de filtro 0:" << endl;
  conv1.printFilter3D(0);

  auto result1 = conv1.convolve3D(input_tensor);
  printTensor3D(result1, "Resultado Convolución 3D");

  cout << "\nEjemplo - 2: Imagen RGB (6×6×3) " << endl;
  auto rgb_image = createRGBExample(6, 6);
  printTensor3D(rgb_image, "Imagen RGB");

  ConvolutionLayer3D conv2(6, 6, 3, 3, 3, 4, 1); // padding=1, 4 filtros
  conv2.printLayerInfo();

  auto result2 = conv2.convolve3D(rgb_image);
  printTensor3D(result2, "Mapas de Características RGB");

  cout << "\nEjemplo - 3: Con Stride = 2 " << endl;
  ConvolutionLayer3D conv3(6, 6, 3, 3, 3, 2, 0, 2); // stride=2
  conv3.printLayerInfo();

  auto result3 = conv3.convolve3D(rgb_image);
  printTensor3D(result3, "Resultado con Stride=2");

  cout << "\nEjemplo - 4: Filtros 1×1 " << endl;
  ConvolutionLayer3D conv4(6, 6, 3, 1, 1, 5); // filtros 1×1
  conv4.printLayerInfo();
  cout << "Filtro 1×1 (canal 0):" << endl;
  conv4.printFilterChannel(0, 0);
  conv4.printFilterChannel(0, 1);
  conv4.printFilterChannel(0, 2);
  auto result4 = conv4.convolve3D(rgb_image);
  printTensor3D(result4, "Resultado filtros 1×1"); 
  return 0;
}
