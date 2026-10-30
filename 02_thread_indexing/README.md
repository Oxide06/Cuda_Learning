# 02 - Thread indexing

Launch: grid of 2 x 3 x 4 = **24 blocks**, each block 4 x 4 x 4 = **64 threads**, so **1536 threads**.

```
block_id      = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y
block_offset  = block_id * (blockDim.x * blockDim.y * blockDim.z)
thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
id            = block_offset + thread_offset
```

This is the same row-major formula as a 2D array index (`col + row * width`), extended to 3D.

Worked example, from the program output:

```
0630 | Block(1 1 1) =   9 | Thread(2 1 3) =  54
block_id = 1 + 1*2 + 1*6 = 9      thread_offset = 2 + 1*4 + 3*16 = 54      id = 9*64 + 54 = 630
```

## What to check in the output

- Every id from 0000 to 1535 appears exactly once.
- Lines are shuffled, but runs of 32 consecutive thread offsets stay together. Each run is one **warp**.
- 64 threads per block = 2 warps (offsets 0-31 and 32-63). 24 blocks x 2 = 48 warps in total.

```bash
nvcc 01_idxing.cu -o 01_idxing && ./01_idxing
```

Open `../visualizations/cuda_loop_vs_parallel.html` in a browser to see why there is no loop.
