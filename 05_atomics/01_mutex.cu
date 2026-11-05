#include<stdio.h>
#include<cuda_runtime.h>

#define NUM_THREADS 600
#define NUM_BLOCKS 2

// Our mutex structure
struct Mutex{
  int *lock;
};

// intialize the mutex
__host__ void initMutex(Mutex *m){
  cudaMalloc((void**)&(m->lock), sizeof(int));
  int intial_val = 0;
  cudaMemcpy(m->lock, &intial_val, sizeof(int), cudaMemcpyHostToDevice);
}

// hold the mutex
__device__ void lock(Mutex *m){
  while(atomicCAS(m->lock, 0, 1) != 0){
    __threadfence();
    //Spin-wait
  }
}

// release the mutex
__device__ void unlock(Mutex *m){
  __threadfence();
  atomicExch(m->lock, 0);
}


// kernel to demonstrate mutex
__global__ void incrementCounter(int *counter, Mutex m){
  lock(&m);
  // critical section
  int old = *counter;
  *counter = old + 1;
  unlock(&m);
}

int main(){
  int h_counter = 0; // host counter
  int *d_counter;
  Mutex m;
  initMutex(&m);
  // allocate memory on device
  cudaMalloc((void**)&d_counter, sizeof(int));

  // copy intial counter values to device
  cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

  // launch the kernel
  incrementCounter<<<2, 600>>>(d_counter, m);

  cudaError_t err = cudaDeviceSynchronize();
  printf("Kernel: %s\n", cudaGetErrorString(err));

  int result;
  err = cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Memcpy: %s\n", cudaGetErrorString(err));
    
  printf("Counter value: %d\n", result);

  // free memory on device because its not needed anymore and its a nice practice
  cudaFree(d_counter);
  cudaFree(m.lock);

  return 0;
}