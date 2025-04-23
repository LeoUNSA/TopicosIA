// Compilar con:
// nvcc mMultiplication.cu -o mMultiplication
// ./mMultiplication
#include <cuda.h>
#include <stdio.h>

#define N 4

typedef struct {
  int width;
  int height;
  float *elements;
} Matrix;

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  int row = threadIdx.y;
  int col = threadIdx.x;
  float value = 0;

  for (int k = 0; k < A.width; ++k) {
    float a = A.elements[row * A.width + k];
    float b = B.elements[k * B.width + col];
    value += a * b;
  }

  C.elements[row * C.width + col] = value;
}

int main() {
  float h_A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float h_B[N * N] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  float h_C[N * N];

  float *d_A, *d_B, *d_C;
  size_t size = N * N * sizeof(float);
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  Matrix A = {N, N, d_A};
  Matrix B = {N, N, d_B};
  Matrix C = {N, N, d_C};

  dim3 dimBlock(N, N);
  MatMulKernel<<<1, dimBlock>>>(A, B, C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("\nMatriz Resultante C:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%6.1f ", h_C[i * N + j]);
    }
    printf("\n");
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
