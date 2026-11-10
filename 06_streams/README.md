# 06 - Understanding Streams

> Work in progress. I'm adding to this as I learn more.

## What I've learned so far

- **A stream is an in-order queue of GPU work** (copies and kernels).
- **Within one stream, operations run in order. Between streams, there is no guaranteed order.**
  If one stream needs another's result, I have to wait for it myself.
- Async calls return to the CPU immediately. The CPU only blocks at a sync call.

```cuda
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
kernel<<<blocks, threads, 0, stream1>>>(...);   // 3rd = dynamic shared memory bytes, 4th = stream
```

- **Sync calls:** `cudaStreamSynchronize(s)` waits for one stream, `cudaDeviceSynchronize()` waits
  for everything. Prefer the narrowest wait that works.
- **Pinned memory** (`cudaMallocHost`) is needed for async copies to really overlap with other work.
  Free it with `cudaFreeHost`, not `free`.
- **Events** can mark a point in a stream, for timing or for making one stream wait on another
  (`cudaStreamWaitEvent`) without blocking the CPU.

## Files

| File | What it does |
|---|---|
| `00_streams_basic.cu` | Vector add with A copied in stream1 and B in stream2, then a kernel and copy back |

## Bugs I hit

- Copied `h_A` into `d_B` by mistake. The kernel computed `A + A` and the verify loop caught it.
- Used a variable name that no longer existed in the verify loop (compile error).
- Passed no arguments to a `fprintf` with format placeholders, so the error message was garbage.

## Takeaways

- With 50,000 elements the data is tiny, so streams give no speedup here. This program is for
  learning the API.
- Streams help when data is large, split into chunks, and copy time and compute time are similar.
- Don't trust a result without checking it. Always verify against the CPU.

## Next

- [ ] Chunked version across several streams, and time it against a single stream
- [ ] Events: replace `cudaStreamSynchronize` with `cudaStreamWaitEvent`
- [ ] Profile with Nsight Systems and look at the real timeline