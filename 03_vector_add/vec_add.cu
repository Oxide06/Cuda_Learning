#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

// CPU vector addition
void add_vec_cpu(float *a, float *b, float *c, int n){
  for(int i = 0; i<n; i++){
    c[i] = a[i] + b[i];
  }
}

// GPU kernel for vector addition
__global__ void add_vec_gpu(float *a, float *b, float *c, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    c[i] = a[i] + b[i]; 
  }

}

// to intialize the vectors
void init_vec(float *vec, int n){
  for(int i = 0; i<n; i++){
    vec[i] = (float)rand() / RAND_MAX;
  }
}

double get_time(){
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv){
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu; //host vectors
  float *d_a, *d_b, *d_c; //device vectors
  size_t size = N * sizeof(float);

  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c_cpu = (float*)malloc(size);
  h_c_gpu = (float*)malloc(size);

  // intializing the vectors
  srand(time(NULL));
  init_vec(h_a, N);
  init_vec(h_b, N);

  // allocate memory on device
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // copy host vectors to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  
  // define grid and block dimensions 
  int num_blocks = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

  printf("Performing warmup run.....\n");
  for(int i = 0; i < 3; i++){
    add_vec_cpu(h_a, h_b, h_c_cpu, N);
    add_vec_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  printf("Starting CPU benchmark.....\n");
  double total_cpu_time = 0.0;
  for(int i = 0; i < 20; i++){
    double start = get_time();
    add_vec_cpu(h_a, h_b, h_c_cpu, N);
    double end = get_time();
    total_cpu_time += end - start;
  }
  double avg_cpu_time = total_cpu_time / 20.0;
  printf("Average CPU time: %f milliseconds\n", avg_cpu_time * 1000);

  printf("Starting GPU benchmark.....\n");
  double total_gpu_time = 0.0;
  for(int i = 0; i < 20; i++){
    double start = get_time();
    add_vec_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end = get_time();
    total_gpu_time += end - start;
  }
  double avg_gpu_time = total_gpu_time / 20.0;
  printf("Average GPU time: %f milliseconds\n", avg_gpu_time * 1000);

  printf("Speedup %fx\n", avg_cpu_time / avg_gpu_time);

  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
  bool out = false;
  for(int i = 0; i < N; i++){
    if(fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
      printf("Error at index %d: %f vs %f\n", i, h_c_cpu[i], h_c_gpu[i]);
      out = true;
      break;
    }
  }
  if(!out){
    printf("Results are correct\n");
  }

  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
