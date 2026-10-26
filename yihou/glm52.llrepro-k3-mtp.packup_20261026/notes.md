# Notes (what / why / how)

- **K3 beats K5 here only under the simulated acceptance pair 3.61 / 2.99.** Decode step p50 fell 55.3 -> 44.6 ms
  (4 vs 6 verify rows, 3 vs 5 draft steps). Real acceptance on AgentX-style requests is lower than both simulated values
  (K5 2.67-2.81, K3 2.32-2.61); the K3/K5 ratio is similar in both (INFERENCE from the table, not an e2e run with real
  acceptance).
- **Graph cap 80 (llr25) OOMed at K5**: 5.85 GB free after capture at mem 0.93; HSA out of resources 28 min in.
  K3 leaves 13.5 GB free, hence the retry llr31.
- **MLA bmm configs (llr28)**: microbench faster at most M, e2e neutral (step p50 unchanged). Not carried.
- **Draft FP8 (llr26)**: PTPC log lines confirmed; shared-expert fusion is off for the draft. InferenceX rules on a
  quantized draft: unknown.
- **aiter config merge**: every `model_configs/*bf16_tuned_gemm*.csv` is merged with the default table; a duplicate
  key raises at start-up after rewriting files. Do not set `AITER_CONFIG_GEMM_BF16` (it disables the scan).
- **Pool node 190**: a second 4-GPU RCCL server on GPU4-7 failed with `hipIpcGetMemHandle failed: invalid argument`
  while GPU0-3 was in use (4 attempts); cause not investigated.
- **Disk**: runs write to the local `/mnt/m2m_nobackup/yihou_llrepro/work` on 135 and are rsynced to NFS
  (`/shared_nfs` hit its quota 10-25).
