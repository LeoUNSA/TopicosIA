// Elaborar un programa que utilice la GPU para la suma de dos vectores de
// valores flotantes. Mida el tiempo de ejecucion para valores mayores a 1000
// elementos.
#include <chrono>
#include <iostream>
using namespace std;
using namespace chrono;

__global__ void vAddGPU(const float *a, const float *b, float *c, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
void randVec(float *vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = static_cast<float>(rand() % 1000);
  }
}
int main() {
  int n = 100000;
  float *vec = new float[n], *vec2 = new float[n], *ans = new float[n];
  randVec(vec, n);
  randVec(vec2, n);
  srand(time(0));
  randVec(vec, n);
  srand(time(0) + 1);
  randVec(vec2, n);
  float *dvec, *dvec2, *dans;
  cudaMalloc(&dvec, n * sizeof(float));
  cudaMalloc(&dvec2, n * sizeof(float));
  cudaMalloc(&dans, n * sizeof(float));
  cudaMemcpy(dvec, vec, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvec2, vec2, n * sizeof(float), cudaMemcpyHostToDevice);
  auto t1 = high_resolution_clock::now();
  vAddGPU<<<40, 256>>>(dvec, dvec2, dans, n);
  cudaDeviceSynchronize();
  auto t2 = high_resolution_clock::now();
  cudaMemcpy(ans, dans, n * sizeof(float), cudaMemcpyDeviceToHost);
  duration<double, milli> ms_double = t2 - t1;
  cout << "Tiempo: " << ms_double.count() << " ms\n";
  cudaFree(dvec);
  cudaFree(dvec2);
  cudaFree(dans);
  delete[] vec;
  delete[] vec2;
  delete[] ans;
  return 0;
}
