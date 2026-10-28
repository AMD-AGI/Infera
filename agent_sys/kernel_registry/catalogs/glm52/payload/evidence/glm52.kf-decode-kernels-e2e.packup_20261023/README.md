> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# GLM-5.2-MXFP4 decode: KernelForge kernels in the sglang 0.5.22 pure DPA+DPA PD stack — TPOT A/B (2026-10-23)

**What.** The three KernelForge (KF) decode kernels (sparse MLA, DSA indexer logits + top-k, MXFP4 MoE) were wired into
sglang through a drop-in python overlay (`overlay/`, decode leg only, no sglang source edits), and A/B-tested end to end
on the latest pure DPA+DPA configuration (dpax1 of `glm52.dcpx-pack-fix-best.packup_20261023`) placed on ONE cluster
cluster node (prefill GPU0-3, decode GPU4-7). Goal (user): observe **TPOT** at a non-absurd throughput to validate the
decode-side optimizations.

**Result** (AgentX C56, 3600 s, simulated acceptance 3.61 — speed only, **no correctness claim**; crossed over two nodes):

| | rounds | score tok/s/chip | TPOT mean ms | TPOT p90 ms | TTFT p50 s |
|---|---|---|---|---|---|
| baseline (KF off) | kfa1 A1/A2 (compute-node-06), kfa2 A1/A2 (compute-node-05) | 24,190 / 25,336 / 24,267 / 24,991 | 15.31 / 14.69 / 15.10 / 15.05 | 19.45 / 18.79 / 19.02 / 18.91 | 7.19 / 6.85 / 7.63 / 6.99 |
| KF on | kfb1 A1/A2 (compute-node-05), kfb2 A1/A2 (compute-node-06) | 25,529 / 25,763 / 25,850 / 26,180 | 11.29 / 11.19 / 11.33 / 11.34 | 12.84 / 12.82 / 12.95 / 13.02 | 8.65 / 8.37 / 8.53 / 8.20 |

- TPOT mean -24 % (14.7-15.3 -> 11.2-11.3 ms), p90 -32 %; the same on both nodes, so not a node effect.
- Score +1..+8 %. TTFT p50 higher with KF (+1.0..1.8 s) — **cause not established**.
- All 8 benches complete (kfa2 A2 appended 12:44Z).
- Kernel level in the same stack (`results/kbench.v0522.compute-node-06.log`, CUDA-graph replay, workset shapes T = 6·bs,
  bs 8-14): sparse MLA 3.13-3.48x, indexer 2.77-3.65x (identical top-k sets), MoE 1.09-1.13x (46-52 dB vs prod).

**Which kernels actually ran** (`results/kf_engagement.kfb2.txt`, `results/decode_bs_distribution.txt`):
per-rank decode batch in the A2 windows is mostly small — mean 4.6-5.4, 88-93 % of samples at bs <= 8.
- sparse MLA: KF on every TARGET_VERIFY row count seen (6..144 rows).
- MoE (gathered M = 24·bs_graph): KF for M 120..432; M <= 96 (draft / tiny) and M >= 480 stay on the original.
- indexer: KF only for 60..120 rows (bs 10-20, ~7-12 % of samples). For <= 48 rows the TARGET_VERIFY call that reaches
  `_get_topk_paged` never shows a KF decision in the log (only `-> orig`); why those rows bypass the gate is **not
  established** (candidate: the gfx950 fused decode indexer path for <= 48 rows). > 128 rows exceed the KF limit.
So most of the measured TPOT gain must come from sparse MLA + MoE at small bs — shapes the KF campaigns did NOT tune
(they tuned bs 8-14). Per-kernel attribution in-model was not measured.

## Navigate
- `REPRODUCE.md` — commands (kernel A/B, harness rounds, crossover), the one-variable dryrun diff.
[Private access details omitted.]
- `notes.md` — integration method, traps hit (what / why / how), open questions.
- `overlay/` — the integration (`sitecustomize.py`, `infera_kernels/__init__.py` + the three pinned candidates), `MD5SUMS`.
- `scripts/` — kernel A/B (`kernel/kbench.registry.py`), compute-cluster v0.5.21 single-server scripts, harness snapshot.
- `cluster/` — what ran on cluster: private harness copy + env files + dryruns (`kf_e2e_20261023/`), per-round status,
  audit, census, AgentX JSON + aiperf summaries (`v0521dbg_20261010/runs/kf*`). Full logs stay on cluster NFS (sizes).
- `results/` — `summary.tsv`, kernel A/B log, decode batch distribution, KF engagement per shape.

**DCP4 (ppdc1 same-node, C56):** KF off 27,062 / 27,051 tok/s/chip, TPOT 19.44 / 19.26 ms; KF on 27,077 / 27,033,
TPOT 19.38 / 19.26 ms — no effect: the overlay's kernels are not on the DCP decode path (notes 9-10).
