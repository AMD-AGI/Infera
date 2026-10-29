# Example: elementwise square plus one

This is a template, not a measured optimization. All sources use ordinary Python
and run on CPU. Replace them with your original operator, naive reference and
optimized implementation. For tensor kernels, a PyTorch reference is appropriate.

The input is a sequence of numbers; output element i is input[i] squared plus one.
There are no in-place writes. Integration is explicit via integration.select;
passing false restores the original. No imports are changed by the catalog tool.
