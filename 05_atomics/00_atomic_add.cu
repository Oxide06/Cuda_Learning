#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// kernel without atomics (incorrect)
__global__ void incrementCounter_non_atomic(int* counter) {
    // not locked
    int old = *counter;
    int new_val = old + 1;
    // not unlocked
    *counter = new_val;
}

// Kernel with atomics (correct)
__global__ void incrementCounter_atomic(int* counter) {
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counter_non_atomic = 0;
    int h_counter_atomic = 0;
    int *d_counter_non_atomic, *d_counter_atomic;

    // allocate device memory
    cudaMalloc((void**)&d_counter_non_atomic, sizeof(int));
    cudaMalloc((void**)&d_counter_atomic, sizeof(int));

    // copy initial counter values to device
    cudaMemcpy(d_counter_non_atomic, &h_counter_non_atomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter_atomic, &h_counter_atomic, sizeof(int), cudaMemcpyHostToDevice);

    // launching kernels
    incrementCounter_non_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counter_non_atomic);
    incrementCounter_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // copying results back to host
    cudaMemcpy(&h_counter_non_atomic, d_counter_non_atomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counter_atomic, d_counter_atomic, sizeof(int), cudaMemcpyDeviceToHost);

    
    printf("Non-atomic counter value: %d\n", h_counter_non_atomic);
    printf("Atomic counter value: %d\n", h_counter_atomic);

    // free device memory
    cudaFree(d_counter_non_atomic);
    cudaFree(d_counter_atomic);

    return 0;
}