#include<stdio.h>
#include<cuda_runtime.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// kernel without atomics (take it as an example of incorrect kernel)
__global__ void incrementCounterNonAtomic(int* counter){
  // not locked
  int old = *counter;
  int new_val = old + 1;
  // not unlocked
  *counter = new_val;
}

// kernel with atomics
__global__ void incrementCounterAtomic(int* counter){
  int a = atomicAdd(counter, 1);
}

int main(){
  int h_counterNA = 0; // host counter non atomicAdd
  int h_counterA = 0; // host counter atomicAdd
  int *d_counterNA, *d_counterA;

  // allocate memory on device
  cudaMalloc(&d_counterNA, sizeof(int));
  cudaMalloc(&d_counterA, sizeof(int));

  // copy intial counter values to device
  cudaMemcpy(d_counterNA, &h_counterNA, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_counterA, &h_counterA, sizeof(int), cudaMemcpyHostToDevice);

  // launch the kernel
  incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNA);
  incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterA);

  // copy the final counter values from device to host
  cudaMemcpy(&h_counterNA, d_counterNA, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_counterA, d_counterA, sizeof(int), cudaMemcpyDeviceToHost);

  // print the final counter values
  printf("Non-atomic counter value: %d\n", h_counterNA);
  printf("Atomic counter value: %d\n", h_counterA);

  // free memory on device because its not needed anymore and its a nice practice
  cudaFree(d_counterNA);
  cudaFree(d_counterA);

  return 0;
}