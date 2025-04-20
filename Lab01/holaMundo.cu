// Implementar el Hola Mundo desde un kernel de CUDA.
#include <stdio.h>
__global__ void holaMundo() {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hola mundo. Hilo numero: %d\n", i);
}
int main() {
  holaMundo<<<5, 2>>>();
  cudaDeviceSynchronize();
  return 0;
}
