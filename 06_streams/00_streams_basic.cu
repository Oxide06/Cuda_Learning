#include<stdio.h>
#include<cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
    exit(EXIT_FAILURE);
  }
}

__global__ void vec_add(float *a, float *b, float *c, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
    c[idx] = a[idx] + b[idx];
  }
}

void init_vec(float *a, int n){
  for(int i = 0; i < n; i++){
    a[i] = (float)rand() / RAND_MAX;
  }
}

int main(){
  int num_elements = 50000;
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  size_t size = num_elements * sizeof(float);
  cudaStream_t stream1, stream2;

  // allocate memory on host
  h_A = (float*)malloc(size);
  h_B = (float*)malloc(size);
  h_C = (float*)malloc(size);

  // initialize the arrays
  init_vec(h_A, num_elements);
  init_vec(h_B, num_elements);

  // allocate memory on device
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

  // create streams
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

  // copy the arrays from host to device
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

  // The kernel (in stream1) needs B, which is in a different stream, so wait for it
  // A needs no wait: stream order already puts the kernel after A's copy
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2)); // cpu waits until d_B is copied to device

  // launch the kernel
  int num_threads = 256;
  int num_blocks = (num_elements + num_threads - 1)/num_threads;
  vec_add<<<num_blocks, num_threads, 0, stream1>>>(d_A, d_B, d_C, num_elements);

  // copy the result from device to host
  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

  // wait for all streams to complete
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

  // Verify result
  for (int i = 0; i < num_elements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Clean up
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}