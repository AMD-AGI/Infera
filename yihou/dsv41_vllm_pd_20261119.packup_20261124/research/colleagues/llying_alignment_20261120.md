# Alignment with llying (liyingli) DSv4.1 vLLM P/D work — read 2026-11-20

Source: commit 16a377402 (record later reverted from dev/yihou/dsv41 by 9a0d05eb5; copy in llying_20261120/),
live workdir /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119 (RO for us). All numbers single runs, his.

## His stack (differs from ours)
- Image vllm nightly-9c1d45c2 (vllm 0.34.0rc1.dev897) — ours is 60bd6dd2 (dev571, the CI image).
- Router: Infera 0.3.3 + overlay (Rust/Python), not vllm-router. Mooncake, per-GPU NIC pinning (MC_TOPO=pergpu), peer-mem or GEM MR.
- Model /perf_apps/data/models/DeepSeek-V4.1-Flash rev dba1be0a. KV dtype: vLLM default fp8_ds_mla.
- Shape: 2P1D on 8 GPUs = prefill DP2(TP1) x2 + decode DP4; also 1P1D (6 GPUs).

## His findings (second-hand for us)
1. vLLM bug: DSv4.1 prefill leg ending FINISHED_STOPPED never sends KV via MooncakeConnector -> decode waits to timeout.
   Fix: treat STOPPED like LENGTH_CAPPED in request_finished (fork PR llying-001/vllm#3), or router min_tokens=1+ignore_eos.
   Our E7 kit does NOT carry this fix -> must add before E7.
2. Decode DP4 + golden synthetic acceptance + CUDA graphs stalls in AgentX warmup (7 configs). Eager decode works.
3. decode TP4: KV replicated per TP rank -> 36M tokens total, saturates at ~250 reqs; ZMQ 1023-socket limit (overlay v2).
4. Best formal: 43x 2P1D decode DP4 eager golden c768 = 291,400 tok/s/GPU (TTFT p50 2.28 s, interactivity 33.6);
   43t TP4 = 141,900. B300 c768 = 420,200 (12 GPUs). Ours colocated R2 c384 = 407,049.
5. B300 prefill settings (mem 0.93, long-prefill 8192, max-num-seqs 64) pass gate; prefix pool 26.3M -> 41M tok/rank.
6. 43n (DP4 PIECEWISE graphs) 03:18Z: AgentX rc=1, 0/825 ok. 43o (1P1D eager) 04:48Z rc=137 (killed). No newer entries.
7. Nothing about Triton "no kernel image" (our E7 crusoe crash) in his record.

## Overlap / proposed split (needs user + llying agreement)
- llying: PD bring-up on Infera router, decode DP4 CUDA-graph stall.
- us: (a) restore AITER sparse-MLA fast path under a PD connector (measured -13.5% colocated, E4b) — research/g1_fastpath_override.md;
  (b) E12/E13 per-GPU prefill/decode rate + capacity model -> is 641k reachable with PD at all;
  (c) colocated reference (isolated R2) as the bar PD must beat at equal GPUs.
- Reuse from him instead of duplicating: STOPPED fix, socket-limit overlay, per-GPU NIC map, B300 prefill settings.
