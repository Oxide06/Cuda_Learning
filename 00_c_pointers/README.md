# 00 - C pointers refresher

Small programs I wrote to make sure the pointer basics are solid before writing kernels,
because `cudaMalloc`, `cudaMemcpy` and every kernel argument are pointers.

| File | What it tests | What to look for |
|---|---|---|
| `01_pointer_levels.c` | `int*`, `int**`, `int***` pointing at one `int` | `*ptr`, `**ptr2`, `***ptr3` all print the same value; each `%p` is the address one level down |
| `02_void_pointer.c` | One `void*` pointing at an `int`, then a `float` | A `void*` cannot be dereferenced directly, so cast first: `*(int*)ptr` |
| `03_null_and_malloc.c` | `NULL` check, then `malloc` | Print `ptr` before and after allocation. Always test the `malloc` result |
| `04_pointer_arithmetic.c` | Walking an array with `ptr++` | Addresses go up by 4 bytes, because `ptr++` moves by `sizeof(int)`, not by 1 byte |
| `05_sizeof_and_size_t.c` | `sizeof(arr) / sizeof(arr[0])` | Gives the element count (5). `size_t` is 8 bytes on a 64-bit system, `int` is 4 |

Run one:

```bash
gcc 04_pointer_arithmetic.c -o ptr4 && ./ptr4
```

Sample output of `04_pointer_arithmetic.c` (addresses differ on every run):

```
Value and Address: 1 0x7ffeb04fbbc0
Value and Address: 2 0x7ffeb04fbbc4
Value and Address: 3 0x7ffeb04fbbc8
```

## Notes and things to improve

- `03_null_and_malloc.c` never calls `free(ptr)`. It is fine for a tiny demo, but it is a leak.
- Printing a `NULL` pointer with `%p` shows `(nil)` on glibc. Other platforms print `0x0`.
- Cast to `(void*)` when passing a pointer to `%p` to stay strictly standard-conforming.
