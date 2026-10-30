# Visualizations

`cuda_loop_vs_parallel.html` - open it in a browser (double-click). Each square is one of the 1536
threads. **Run CPU loop** lights them one at a time; **Launch GPU kernel** lights them in batches of
warps. Click any square to see how that thread works out its own id from `blockIdx` and `threadIdx`.

If GitHub Pages is enabled for this repo, it can be served as a live page.
