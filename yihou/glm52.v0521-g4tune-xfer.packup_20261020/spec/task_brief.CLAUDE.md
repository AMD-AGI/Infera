# TASK — tune GLM-5.2 v0.5.21 same-node P4D4 with OffloaderV2 G4 at high concurrency (2026-10-19)

Base config = the best point so far, **ofg4b A2 23,771 tok/s/chip** (C48, prefill OffloaderV2 G4K1 step 1,
pf90 recipe: prefill mem 0.90, HiCache 450 GB write_through, decode mem 0.95, decode max_running 96,
chunked-prefill 32768, MTP sim 3.61). Goal: more throughput at **C56** (debug/analysis first), then **C52**.
Decode always runs `--tokenizer-worker-num 4 --detokenizer-worker-num 4` (user rule 10-15).

User's work items (2026-10-19):
1. **Explain the prefill "inflight" stage** (last chunk sent -> KV transfer Success; 5 s at C48, 10-12 s at C56,
   15-19 s at C64; per request ~7.5-8 GB transferred = whole context). Suspects: the machine; mooncake —
   turn on mooncake logging via env vars and debug.
2. **Larger chunk size** — user's hypothesis: small chunks cannot hide the group-MoE weight load of OffloaderV2.
3. Debug at C56; C52 when there is time.
4. Any other point the analysis needs verified.

## Where things are

- spur cluster, login `ssh spur006` (fallback spur007 / spur005). Login hosts evict a user above ~4 GB.
- Harness `H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`, runs `R=…/v0521dbg_20261010/runs`,
  session tools `N=/shared_nfs/yihou_glm52/v0521cap2_20261013`; analysis so far in `$N/analysis_conc_20261019/`
  (scripts `$N/{conc_anal2,kv_ts,isl_turn,perdp,dec_q,stage,hist}.yihou.py`).
- **New workspace for this task: `/shared_nfs/yihou_glm52/g4tune_20261019/`** (NFS) and `/tmp/yihou/g4tune/` (local).
- sglang source of the image (upstream commit `11b1fb21de1c48acee33c6f23a6fc57b1c322341`) fetched to
  `/tmp/yihou/sgl_11b1/` via `gh api`. Local checkouts under `~/dev/git.16-19/sglang` are a DIFFERENT commit.
- Ledger: `/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug/poll.md` (on spur).
- Latest packup: `~/dev/git/infera.glm52.pd/yihou/glm52.v0521-offloaderv2-burst-c56c64.packup_20261019` (on spur).

## Nodes

- OAI is allowed again (user 10-19): `-A amd-oai -q amd-oai-qos -p amd-spur`. Holds 224693/224694 submitted.
- Fallback: amd-spur-v2 burst (preemptible, often 20 min – hours). amd-spur burst holds die at 20:00 — do not use.
- Job 224677 `yihou-dsp-batch` belongs to ANOTHER session — never touch.
- 136 is down (~85 h); user will say when it is back. Teammates: burstkeeper2 (holds + prep), watch136.

## Core principles

1. Suspend, don't conclude: report measurements; no mechanism without a code read or an experiment.
2. Never delete outside a `yihou` dir; no recursive delete with a variable target; `docker stop`, never `rm`.
3. One variable at a time vs the G4 baseline; dryrun + tokdiff before every launch; record every change in the ledger.
4. Correctness is waived by construction (sim 3.61) — never present acceptance as correctness.
5. English in files, Chinese to the user. Commits need `git commit -s` (none planned).
