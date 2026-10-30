# 01 - Hello, CUDA

```cuda
hello<<<2, 4>>>();   // 2 blocks, 4 threads each = 8 threads
```

Each thread prints its `threadIdx.x` and `blockIdx.x`.

```bash
nvcc -arch=sm_75 hello.cu -o hello && ./hello
```

The lines do **not** print in order (block 1 can appear before block 0). All 8 threads run in
parallel, so whichever finishes `printf` first shows up first. `cudaDeviceSynchronize()` makes the
host wait so the buffered output actually appears.
