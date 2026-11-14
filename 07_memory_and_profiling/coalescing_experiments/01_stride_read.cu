#include<stdio.h>
#include<cuda_runtime.h>

#define N 4194304     // 4M threads, 4M floats written per run
#define MAX_STRIDE 32 
#define ITERS 50

__global__ void strided_read(const float* in, float* out, int stride){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in[(size_t)idx * stride];
}

int main(){
  float *d_in, *d_out;

  cudaMalloc(&d_in, (size_t)N * MAX_STRIDE * sizeof(float)); // 512MB
  cudaMalloc(&d_out, (size_t)N * sizeof(float));
  cudaMemset(d_in, 0, (size_t)N * MAX_STRIDE * sizeof(float));

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  strided_read<<<blocks, threads>>>(d_in, d_out, 1); // warmup
  cudaDeviceSynchronize();

  printf("stride | ms per run | useful GB/s\n");
  for(int stride = 1; stride <= MAX_STRIDE; stride*=2){
    cudaEventRecord(t0);
    for(int i = 0; i < ITERS; i++){
      strided_read<<<blocks, threads>>>(d_in, d_out, stride);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms; cudaEventElapsedTime(&ms, t0, t1);
    ms /= ITERS;
    double useful_gb = 2.0 * N * sizeof(float) / 1e9;   // N floats read + N floats written
    printf("%6d | %10.4f | %10.1f\n", stride, ms, useful_gb / (ms / 1000.0));
  }

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
  
}