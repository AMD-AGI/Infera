# Notes — wrong turns, blind spots, corrections, open points

## 1. Bisect before the debug (10-20 02:00-08:00Z)

- **What**: with the new rule (PR #42553 + index_share true + decode custom AR on) decode died with a MAF in g4n2 (chunk 65536, 250).
  Suspects in turn: chunk 65536, node 250, custom AR, index_share.
- **How it was settled**: g4n2b (= g4n2, 250) faulted → reproducible; g4n1 (chunk 32768, 097) faulted → not chunk, not node; g4n4 (custom AR
  off, 250) faulted → not custom AR; g4n3/g4n3b (index_share false) ran A1+A2 clean on both nodes → the trigger is index_share. One knob per
  round, tokdiff-checked (`results/dryruns/`).
- **Earlier history**: c44c (10-09, index_share true) also faulted; its host stack at the fault pointed at `fused_dsa_draft_extend_metadata`
  — HIP errors are asynchronous, so the host stack does not name the faulting kernel. That is why the debug agent was used.

## 2. Naming the kernel (dbge1a/1b)

ROCm debug agent (decode only; `--disable-linux-signals`; ~1.00x overhead per the 10-09 R009 validation) stopped 16 waves with
MEMORY_VIOLATION at the same pc in both runs: `sgl::dsa_topk::coop_topk_kernel<2048,12,4096,1024>(CoopParams<2048>)`, pc +700 = `s_waitcnt`
after `global_load_dwordx4` in the round-0 histogram row scan. SGPR map from the disassembly of the image's code object
(`results/debug_agent/coop.dis.yihou.txt`): s[40:55] = CoopParams (s40:41 input, s42:43 out, s44:45 row_starts, s46:47 lengths, ...),
s2 = row, s76 = lengths[row], s3 = max(len, 0), s58:59 = row base.
- dbge1a: row 0, row_starts null, len 60,314 (s3 0xeb9a), input 0x7cc61abf4600, fault page 0x7cc61ac00000 = input + 47,616 B.
- dbge1b: row 0, len 99,462 (s3 0x18486), input 0x73914c7f5000, fault page = input + 45,056 B.
So the score buffer ended < ~46 KB after its start while the kernel scanned 60k-99k floats of a row. The AOT host entry
(`kernels/aot/csrc/elementwise/topk.hip`) does not check `row_start + lengths[row] <= score.size(1)` (device values).
The seed guard (count) reported `bad_entries=0` → the 10-09 "stale/out-of-pool DSA seed" suspicion did not hold here.

## 3. Guard blind spots (why dbge2a-dbge3b found nothing)

| guard | round(s) | blind spot (observed) |
|---|---|---|
| indexer guard v1 | dbge2a | logged eager calls only; serving top-k calls are CUDA-graph replays → one warm-up line in 17 min |
| indexer guard v2 | dbge2b | daemon-thread logger covers replays, but the device counter was allocated lazily at the first eager call, i.e. after the serving graphs were captured → captured graphs carry no guard, counters frozen |
| indexer guard v3 | dbge2c | counter allocated in `Indexer.__init__`; instrumented the paged path only — the faulting k-only path was not instrumented |
| indexer guard v4 | dbge2d / 2e | adds k-only counters; counters still frozen during serving (~200 requests). Why was not established; both rounds ran only ~20 min of bench, and BAD calls later appeared ~20 min into load |
| SKGUARD v1 (sgl_kernel call site) | dbge3a / 3b | keyed by Python call site, graph replays counted (unit test 5/5) — but graphs captured before the first eager top-k call carry no guard, and a faulting eager call kills the process before the 60 s logger flushes |
| SKGUARD v2 + indexer v5 | dbge4a / 4b | armed at `Indexer.__init__` (before any capture); eager calls checked synchronously before launch → caught the BAD call with its stack |
| SKGUARD v3 (mask + dump) | dbge5a / 5b | dumps ForwardBatch + attention metadata found up the stack; clamps → 12 / 24 BAD samples, 0 MAF |

Lesson: a guard must exist before graph capture and must report synchronously, or it is blind to exactly the call that kills the process.

## 4. Reasoning step that was later confirmed

10-20 12:0xZ (before any BAD record): a paged-path score row is 1,048,704 floats (4 MiB), so a fault ~46 KB after row 0 cannot leave it; the fault
is consistent only with a narrow score buffer → the k-only path (2048-wide dummy logits). Confirmed 12:17Z by dbge4a's BAD record from
`Indexer._forward_cuda_k_only` (score (2, 2048), len 19,238), MAF 1 s later with s3 = 0x4b26 = 19,238.

## 5. Correction to the 10-20 report

The report (`analysis/REPORT.idxshare-maf.20261020.md`) and ledger say "dbge4b: 4 BAD, no MAF (silent garbage)". The decode log shows the 4 BAD
calls at 12:13:51-52Z without a fault, **and** 2 more BAD calls + a MAF at 12:21:12Z — one second after the operator stop terminated the bench
(`bench FAIL rc=143` 12:21:11Z, `results/rounds/dbge4b/status.tsv`). So "a BAD call does not always fault" still holds (the 12:13 calls), but
dbge4b as a whole did fault (`results/maf/maf_lines.txt`).

## 6. The 10-19 Mooncake microbench claim is unrelated

The 10-19 mc_bw2 result (Mooncake HIP transfer 0.35-0.4 GB/s with a GEMM burn on the sender GPU) concerns KV-transfer speed, not this fault; no
link to the MAF chain was found or claimed. It was also not reproduced on 10-20 (xgmi_load v1-v6 on 097: 28-63 GB/s; the 10-19 burn ran as a
thread in the sender process).

## 7. Measurement caveats

- Scores are single runs (n = 1). idxf1a A2 21,814 vs g4n3 A2 22,045 (idx false, same node) — not a performance conclusion.
- dbge6a ran with the debug agent and guards; its A1 (21,189) is not comparable to idxf1a.
- Client `err` in tools_hits is request errors of the bench client, not engine faults.
- `analysis/NOTES.prep.20261020.md` dmesg table is a transcription; the raw dmesg output and the exact container command were not saved.
- The extraction commands for `coop.co.yihou` / `coop.dis.yihou.txt` were not recorded.
- `top_k.yihou-skguard3.py`: the workspace copy in `idxshare_dbg_20261020` (15,203 B) differs from the mounted copy in
  `harness/mounts-v0521/src` (15,242 B); the kit contains the mounted one.

## 8. Open points

- Replicates idxf1b (268) / idxf1c (054) were running when the kit was built.
- The fix was validated only on the image tree (9f5d8712 + local diff). No run on upstream main; `patches/0002-*` is a port, compile-checked only.
- Only `EagerRunner._execute_idle` was changed. Whether other runners/paths read a forward-context backend that was not re-initialized was not audited.
- The extra `init_forward_metadata` per idle draft step has no separately measured cost.
- The coop kernel itself has no bound check on `row_start + length <= row width`; the fix removes the bad input, not the missing check.
