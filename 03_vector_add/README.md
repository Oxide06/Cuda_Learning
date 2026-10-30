# 03 - Vector addition (CPU vs GPU)

Adds two vectors of 10,000,000 floats. The GPU kernel gives each thread one element:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

- `BLOCK_SIZE = 256`, so `num_blocks = (N + 255) / 256 = 39,063`. The rounded-up launch has 128 extra
  threads, and the `if (i < n)` check keeps them from writing past the end.
- 3 warmup runs, then 20 timed runs averaged. `cudaDeviceSynchronize()` is called before the end timestamp.
- The result is copied back and compared with the CPU result (tolerance 1e-5).

## Results (Tesla T4, Colab)

| | Time |
|---|---|
| CPU, average of 20 runs | 33.19 ms |
| GPU kernel only, average of 20 runs | 0.469 ms |
| Speedup (kernel only) | about 70.8x |
| Correctness check | passed |

## How to read this honestly

- The GPU time covers **only the kernel**. The host-to-device copies (two vectors in) and the copy
  back are not timed. For a simple operation like addition, copy time is often larger than compute
  time, so the end-to-end speedup is much smaller than 70x.
- The CPU loop is single-threaded. Compiling with `-O2` or using multiple cores would narrow the gap.
- Numbers are from one machine and one session. Colab GPUs vary run to run.

## Next steps

- [ ] Time the full pipeline (copies included)
- [ ] Add a `CUDA_CHECK` error macro
- [ ] Try different `BLOCK_SIZE` values (128, 256, 512, 1024)
- [ ] Compare against a multi-threaded CPU version

```bash
nvcc -arch=sm_75 -O2 vec_add.cu -o vec_add && ./vec_add
```
