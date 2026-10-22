# Patches — what / why / how / context

Only ONE patch differs between the two rounds. Everything else (image, three baseline runtime mounts) is identical and
documented in the baseline kit `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/patches/`; copies of those
mount files are in `../scripts/harness/mounts-v0521/src/`.

## `decode.yihou-runshare2.py` (+ `decode.yihou-runshare2.diff`) — runshare v2

- **What:** decode-only runtime mount over `/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py`.
  Base = image `srt/disaggregation/decode.py` sha256 `97a2abfa…0c1b` (md5 bd1e5fe7). The diff (66 lines of `diff`
  output) adds, in the prealloc admission loop (`pop_preallocated`):
  `used = pool - allocator.available_size()`, `running = Σ running_batch.reqs[*].kv.kv_allocated_len`,
  `nonrunning = used - running`; a request is admitted only if `nonrunning + required_alloc_tokens < (1-SHARE)·pool`
  (`SHARE = YIHOU_DECODE_RUNNING_KV_SHARE`, 0/unset = stock). A non-fitting request is **skipped** (smaller ones
  behind it may enter); once the oldest blocked request has waited > `YIHOU_RUNSHARE_MAX_WAIT_S` (default 5 s) the scan
  **stops** (stock FIFO break, anti-starvation). Rebootstrap requests are exempt; a request may always enter when the
  transfer queue and waiting queue are empty. Every 100th block per DP rank logs
  `YIHOU-RUNSHARE block #n: nonrunning= cand= cap= running= pool= xfer= waiting= waited= skip|stop`.
- **Why:** user rule 2026-10-12: "(要申请的量 + 已经占有的量) − running req 占用的量 < 50 % 才可以" — running requests must
  always keep ≥ 50 % of the decode KV pool so a slow prefill cannot let in-flight (prealloc -> transfer) requests hog it.
  No sglang v0.5.21 flag expresses this (research R1 in the sibling kvcap packup).
- **How:** `config.yihou.runshare2.sh` (overlay; chains `RUNSHARE_BASE_CONFIG`, here `config.yihou.mc4340n2.sh`) sets
  `DECODE_MOUNTS_ADD=$WS/harness/mounts.yihou.decode-runshare2.tsv` and appends
  `YIHOU_DECODE_RUNNING_KV_SHARE=${RUNNING_KV_SHARE:-0.5} YIHOU_RUNSHARE_MAX_WAIT_S=${RUNSHARE_MAX_WAIT_S:-5}` to
  `DECODE_EXTRA_ENV`. The TSV maps the NFS file to the container path (read-only bind mount).
- **Context / evidence:** v1 (FIFO break) regressed C40 warm to 21,271 vs 22,114 (-3.8 %); v2 = skip + 5 s wait.
  v2 functional smoke (64 × ~231k-token prompts, node 102): 64/64 ok, "pre-allocated usage" max 0.42, 0 faults;
  v2 C40 fresh 20,680 (`../results/reference/pc40d-A1-C40-runshare2/`); v2 C40 warm never run. **This packup** adds the
  high-load A/B: C56 A2 17,611 vs 18,016 none (-2.2 %), C64 A3 15,070 vs 15,108 (-0.3 %). In tb56 the cap fired
  throughout (266,980 log lines = ~26.7 M blocked checks summed over 4 DP ranks [inferred ×100], 72 % of logged blocks
  were `stop`, i.e. the head request had waited > 5 s) and held "pre-allocated usage" at max 0.50 (no patch: 0.98).
  Full design + v1 -> v2 history: `../../glm52.v0521-decode-transfer-kvcap.packup_20261013/` (not duplicated).

Files here: `decode.yihou-runshare2.py`, `decode.yihou-runshare2.diff`, `config.yihou.runshare2.sh`,
`mounts.yihou.decode-runshare2.tsv` (byte-identical to the copies under `../scripts/harness/`).
