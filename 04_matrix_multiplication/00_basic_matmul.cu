#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include <math.h>

// lets do a simple matrix multiplication
/*
* consider two matrix A and B of order 2x3 and 3x2 respectively.
                                     (m x k) (k x n)
  A = [[ 1, 2, 3],
       [ 4, 5, 6]]

  B = [[ 10, 20],
       [ 30, 40],
       [ 50, 60]]

  Let C = A @ B
  Then C = [[ 220, 280],
            [ 490, 640]]

  Order of C = (2x3) @ (3x2) = (2x2)

  We have to replicate this undergoing process so that it can run on GPU
  C = [[(1 * 10) + (2 * 30) + (3 * 50), 1 * 20 + 2 * 40 + 3 * 60],
       [4 * 10 + 5 * 30 + 6 * 50, 4 * 20 + 5 * 40 + 6 * 60]]

  We'll use a bigger matrix    
*/

#define M 128
#define K 256
#define N 512
#define BLOCK_SIZE 32

void cpu_matmul(float* A, float* B, float* C, int m, int k, int n)
{
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      float sum = 0.0f;
      for(int k_x = 0; k_x < k; k_x++){
        sum += A[i * k + k_x] * B[k_x * n + j]; // row major
      }
      C[i * n + j] = sum;
    }
  }
}

__global__ void gpu_matmul(float* A, float* B, float* C, int m, int k, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < m && col < n){
    float sum = 0.0f;
    for(int k_x = 0; k_x < k; k_x++){
      sum += A[row * k + k_x] * B[k_x * n + col];
    }
    C[row * n + col] = sum;
  }
}

void init_mat(float* mat, int n)
{
  for(int i = 0; i < n; i++){
    mat[i] = (float)rand()/RAND_MAX;
  }
}

double get_time()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(){
  float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
  float *d_A, *d_B, *d_C;
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  h_A = (float*)malloc(size_A);
  h_B = (float*)malloc(size_B);
  h_C_cpu = (float*)malloc(size_C);
  h_C_gpu = (float*)malloc(size_C);

  srand(time(NULL));
  init_mat(h_A, M * K);
  init_mat(h_B, K * N);

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  dim3 dimGrid((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // ( 32, 32, 1)

  printf("Performing warmup run.....\n");
  for(int i = 0; i < 3; i++){
    cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);
    gpu_matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
  }
  printf("Starting CPU benchmark.....\n");
  double total_cpu_time = 0.0;
  for(int i = 0; i < 20; i++){
    double start = get_time();
    cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);
    double end = get_time();
    total_cpu_time += end - start;
  }
  double avg_cpu_time = total_cpu_time / 20.0;
  printf("Average CPU time: %f milliseconds\n\n", avg_cpu_time * 1000);

  printf("Starting GPU benchmark.....\n");
  double total_gpu_time = 0.0;
  for(int i = 0; i < 20; i++){
    double start = get_time();
    gpu_matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    double end = get_time();
    total_gpu_time += end - start;
  }
  double avg_gpu_time = total_gpu_time / 20.0;
  printf("Average GPU time: %f milliseconds\n\n", avg_gpu_time * 1000);
  printf("Speedup %fx\n", avg_cpu_time / avg_gpu_time);

  cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++) {
        // fabs() works on doubles, so the floats are converted up first (harmless, just slower)
        // 1e-5 is an ABSOLUTE tolerance: "the two answers may differ by at most 0.00001"
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
        // Problem 1: with values around 64, 1e-5 is too tight (explained below),
        // so this can fire even when the GPU is right.
            printf("Error at index %d: %f vs %f\n", i, h_C_cpu[i], h_C_gpu[i]);
            break;
        }
    }

  int ok = 1;
  for (int i = 0; i < M * N; i++) {
    float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
    // RELATIVE tolerance: the allowed error grows with the size of the value.
    //   value ~ 64  ->  allowed = 1e-3 * 64 = 0.064
    // fmaxf(1.0f, ...) is a floor: for values near 0, a purely relative limit
    // would shrink toward 0 and reject harmless noise, so it falls back to 1e-3.
    if (diff > 1e-3f * fmaxf(1.0f, fabsf(h_C_cpu[i]))) {
      printf("Error at index %d: %f vs %f\n", i, h_C_cpu[i], h_C_gpu[i]);
      ok = 0;
      break;
    } 
  }
  if (ok) printf("Results match\n");

  free(h_A);
  free(h_B);
  free(h_C_cpu);
  free(h_C_gpu);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}