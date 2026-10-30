# CUDA Learning Journal

My notes and experiments while learning CUDA and GPU programming, starting from C pointers and
working up to a first vector-addition benchmark.

> **Credit:** my learning has been heavily based on
> [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course/tree/master).
> The curriculum belongs to its author. This repo is my own practice, notes and experiments while
> following it. If you want to learn CUDA, start there.

## Contents

| Folder | What it covers |
|---|---|
| [`00_c_pointers`](00_c_pointers) | Pointer refresher in plain C: multi-level pointers, `void*`, `NULL` and `malloc`, pointer arithmetic, `sizeof` and `size_t` |
| [`01_hello_cuda`](01_hello_cuda) | The first kernel: `hello<<<2, 4>>>()` |
| [`02_thread_indexing`](02_thread_indexing) | Flattening 3D `blockIdx` / `threadIdx` into one global id, and why 64 threads per block means 2 warps |
| [`03_vector_add`](03_vector_add) | CPU vs GPU vector addition on 10M floats, with a correctness check |
| [`visualizations`](visualizations) | Interactive HTML page: a CPU `for` loop vs a kernel launch |
| [`notebooks`](notebooks) | The Colab notebook these files came from |

## Setup

- GPU: Tesla T4 (Google Colab free tier), compute capability 7.5 (`sm_75`)
- CUDA toolkit: 13.0 (check yours with `nvcc --version`)
- No GPU of your own? Open the notebook in Colab and pick **Runtime > Change runtime type > T4 GPU**.

Compile and run any CUDA file:

```bash
nvcc -arch=sm_75 -O2 file.cu -o file
./file
```

Compile and run any C file:

```bash
gcc file.c -o file && ./file
```

## Key takeaways

1. **There is no loop in a kernel.** The launch line starts all the threads, and each one computes
   its own global id from `blockIdx` and `threadIdx`.
2. **3D indexing is just row-major flattening**, once for the block in the grid and once for the
   thread in the block.
3. **Warps are 32 threads**, cut from the flat thread id. A block of 64 threads is 2 warps.
4. **Output order from parallel threads is not guaranteed.** Warps start in whatever order the
   scheduler picks.
5. **Benchmark carefully.** Launches are asynchronous, so synchronize before stopping the timer.

## Testing checklist

Things I check before trusting a result:

- [ ] Compiles with no warnings (`nvcc -Wall`-style flags where available)
- [ ] The GPU result matches the CPU result (tolerance, not exact equality, for floats)
- [ ] The kernel has a bounds check (`if (i < n)`) when the thread count is rounded up
- [ ] A warmup run happens before timing
- [ ] `cudaDeviceSynchronize()` is called before the end timestamp
- [ ] CUDA calls are checked for errors (planned: a `CUDA_CHECK` macro)
- [ ] Timings are averaged over several runs and the GPU model is written down

## Results so far

See [`03_vector_add`](03_vector_add) for the numbers and what they do and do not mean.
