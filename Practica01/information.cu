// Compilar con:
// nvcc -o information.cu information
// ./information
#include <stdio.h>
int main() {
  int deviceId = 0;
  cudaGetDeviceCount(&deviceId);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);
  printf("Name: %s\n", prop.name);
  printf("Total global memory: %ld\n", prop.totalGlobalMem);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("Shared Memory / SM: %ld\n", prop.sharedMemPerBlock);
  printf("Registers / SM: %d\n", prop.regsPerBlock);
  printf("Warp Size: %d\n", prop.warpSize);
  return 0;
}
