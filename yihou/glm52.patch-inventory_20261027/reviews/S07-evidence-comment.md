Additional independent AMD reproduction for the attention-DP validation fix already included here; we will not open a competing PR for it.

On GLM-5.2-MXFP4, MI355X, sglang `0.5.21.dev20261008+g11b1fb21de`, same-node 1P1D with TP4/attention-DP4 on each leg:

```text
--disaggregation-decode-allocation-policy prefill_complete
--tp-size 4 --dp-size 4 --enable-dp-attention
```

Run `pc48` exited during prefill startup with `ValueError: ... does not yet support attention tensor parallelism greater than one`. `handle_deprecated_dp_attention` had already resolved the topology to `attn_dp_size=4`, `dp_size=1`, and `enable_dp_attention=False`, so the old validator incorrectly derived attention DP as 1.

With our local validation-only fix, run `pc48b` started both roles with `prefill_complete`, and completed two 3,600-second C48 windows: A1 had 3,773 successful requests / 1 error; A2 had 4,027 successful requests / 3 errors. Prefill resolved `optimistic_prefill_attempts=1` and HiCache write-back. This supports the startup-validation diagnosis; it is not an error-free serving claim.

I rechecked current main `e4d4a9a7de`: the old-field expression is still present. This PR's use of canonical `attn_dp_size` and its regression covering the deprecated rewrite and both parameter spellings address the same defect. Our data is for PP1, so it does not validate this PR's separate PP scheduling changes.

The historical workload used simulated MTP acceptance (`3.61`). It does not establish generated-text correctness, and we make no throughput claim for this one-line fix. Source record: `yihou/glm52.v0521-prefill-complete.packup_20261014/README.md` and `results/rounds/pc48/prefill-0.startup-ValueError.log` in the Infera experiment archive.
