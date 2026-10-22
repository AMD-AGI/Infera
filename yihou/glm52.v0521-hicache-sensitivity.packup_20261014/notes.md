# Notes — gotchas, caveats, open questions (what / why / how / context)

## 1. The comparison is not single-variable (node + image build + cache location + foreign container)

- **What:** hc50 ran on 136 (image 9872bbda, its own build; caches on node-local `/mnt/m2m_nobackup`; NFS read-only),
  e2g on 102 (image 9d755dd6; caches on NFS), two days apart. The dryrun token diff
  (`results/dryrun/tokdiff.e2g-102_vs_hc50-136.txt`) shows only `--hicache-size 450 -> 50` plus IPs and mounts, so the
  *engine arguments* are single-variable; the *machine* is not.
- **Why it matters:** no e2g-recipe run exists on 136 and no hc50 run exists on 102, so node/build effects cannot be
  subtracted. The 102 fresh-C40 replicates spanned 19,335-20,371 (decode-transfer packup); node-to-node spread on
  this cluster at this recipe is unmeasured.
- **How to close:** run `env.e2g` (450 GB) on 136 with 9872bbda, or hc50 on a 9d755dd6 node (085/274/089 via
  `config.yihou.mc4340n4.sh`); ideally both.
- **Context:** 136 was free at 15:55 (ledger: limou's vLLM gone; 135/137 = llying, untouched); 102 had ended and the
  oai holds were running the x8pc/pc48e/tb56n/x128p rounds (poll.md). The user directive asked for the e2g config "everything else unchanged" — the config is unchanged, the node
  could not be.

## 2. Foreign container dev_xb on 136

- **What:** `dev_xb` (lmsysorg/sglang:v0.5.20-rocm724-mi35x) was up for ~5 days at env capture (2026-10-14 01:57),
  i.e. during hc50. `run.v3.yihou.sh:137-138` already describes it as "a foreign looping job (136 dev_xb)" with short
  idle gaps — which is why the GPU-idle gate needs 2 consecutive idle reads >= 60 s apart.
- **Measured:** at chain start 0 KFD processes (chain_cap2l check) and max VRAM 283 MiB twice; nothing else.
- **Not measured:** whether dev_xb launched GPU or CPU work during the 3 h of hc50. Not ours — not touched.

## 3. Host pool was already full when the scored A2 started

- **What (measured, `analysis/hc50_A2_metrics_summary.txt`):** first A2 sample after the two `flush_cache` calls:
  prefill `hicache_host_used_tokens` 1,111,680-1,112,896 of 1,112,896 on all 4 ranks; `load_back_tokens_total` stayed at
  8,192 (rank 0) for the whole A2; host-backup and evicted-token counters kept rising (rank 0: +5.57 M backed up,
  +9.15 M evicted during A2).
- **Context:** AgentX JSON `cpu_cache_hit_rate` = 0.000 for A2; A1's JSON has no `cpu_cache_hit_rate` key (shown as
  absent in the table). Whether `/flush_cache` clears the host tier at all was not checked here (e2g has no /metrics TSV;
  the sampler came later). Do not read "full at A2 start" as "flush does not clear host"; it may have refilled
  between the flush and the first sample — unknown.

## 4. HiCache size vs ratio

- `--hicache-ratio 1.5` stays on the command line in every e2g-family round; `--hicache-size N` (GB per rank) overrides it.
  Live census of hc50: `Allocating kv hierarchical KV host pool: 1112896 tokens, 50.00 GB host memory` x4. Without
  `--hicache-size` the ratio gives 153.81 GB / 3,423,488 tok per rank (e1c) — that is the "154 GB" class.
- Per-rank capacities seen in this session: 450 GB = 10,016,064 tok, 370 GB = 8,235,456, 360 GB = 8,012,864 (k1 JSON),
  ratio 1.5 = 3,423,488 (v0.5.21 e1c) / 3,649,664 (v0.5.21 tb56 JSON, same 154 GB class), 50 GB = 1,112,896.

## 5. Cited HiCache points are mostly not like-for-like

- k1 (360 GB) is C44; compare it to C44 runs at 450 GB (X1 21,773, q1 A2 21,862), not to e2g/hc50.
- e1c (154 GB) differs in decode mem (0.85 vs 0.95) and is a fresh run.
- The 8-expert C56 arms (154 / 274 / 450 GB: 23,892 / 26,462 / 26,071) are v0.5.19 and a synthetic model; they are
  like-for-like with each other only. In that sweep 274 > 450 GB, i.e. score was not monotonic in HiCache size.
- Ordering of v0.5.21 real-model points by HiCache at their own CONC is therefore not a clean curve; the only
  same-recipe, same-CONC pair is e2g (450) vs hc50 (50), with the caveats in §1-2.

## 6. Login-host driver for plain-ssh nodes (chain_cap2l)

- 136's `/shared_nfs` is read-only, so the node cannot write `$R`. `chain_cap2l.yihou.sh` runs on `spur006`, drives
  the node over plain ssh, and runs the `/metrics` sampler on the login host. Server logs are written on the node under
  `NODE_RW_ROOT` and only **snapshot-copied** to `$R/hc50/server-logs/` at bring-up (16:28); the full logs (to 19:22)
  are on 136 only (`logs/LOG_INDEX.md`).
- If the login host dies, a login-host driver dies with it (the 2026-10-12 session lost a driver that way). Spur nodes
  avoid this by running `chain_cap2` on the node via `spur exec` + the local ssh shim.

## 7. Stack left running

- The chain ends with "chain DONE (stack left running)". At 2026-10-14 01:57 the four hc50 containers were still up on
  136 holding ~283-300 GB VRAM per GPU. Stop with `run.v3.yihou.sh stop` when 136 is needed (not done by this pack-up:
  read-only).

## 8. Score anatomy (measured fields, no mechanism claimed)

score ~ (input + output tok/s) / 8; A2 input tok/s 159,721 (hc50) vs output 1,176 — the score is >99 % input
tokens, so it moves with qps x mean ISL. hc50 A2 had both lower qps (-5.2 %) and lower mean ISL (-4.1 %) than e2g A2.
Mean ISL depends on which trajectories complete in the window (same dataset/seed/CONC); why ISL is lower here is not
established.

## 9. Cold-read verification (2026-10-14, separate pass)

- Numbers: hc50 A1/A2 and e2g A2 rows of `results/results.csv` match `tools_summ.yihou.py` output (score, qps, ISL,
  TTFT mean/p95, GPU/CPU/overall hit, ok/err, cpu pool). Packed JSONs of hc50 A1/A2, e2g A1/A2, k1 A2, e1c r2 are
  byte-identical (sha256) to `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/...` originals.
- Scripts: `chain_cap2l`, `run.v3`, `pipeline.v3`, `tools/audit_live.v3`, `config.yihou.E2-target`, `env.hc50-136`,
  `mounts.yihou.v2-regbase.tsv`, `bench-harness/engine.sh`, `shimbin/ssh` are byte-identical to the originals.
- Added: `scripts/harness/tools/pidcensus.yihou.sh` (called by `run.v3.yihou.sh:306`, was missing). Not copied (not on
  the hc50 path): `tools/audit_live.v2.yihou.py` (run.v2 only), `mounts.yihou.v2.tsv` (base default, overridden by
  mc4340n), `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin/ssh` (off-node spur pipeline only).
- `scripts/harness/bench-harness/tools/__pycache__/agentx_env.cpython-314.pyc` is a verbatim byproduct of the harness
  tree, not needed; left in place.
- Secrets: regex scan for token/key/password values (hf_/ghp_/AKIA/sk-/private keys, `*TOKEN=`/`*KEY=` env lines,
  inside .gz too) found none; only null-valued server args.
