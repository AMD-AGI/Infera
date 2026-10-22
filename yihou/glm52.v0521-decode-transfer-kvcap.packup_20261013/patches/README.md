# Patches — what / why / how

Image and baseline mounts are unchanged from the baseline kit (`glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/patches/`).
New in this task (all decode-only runtime mounts over `/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py`,
base = image decode.py md5 bd1e5fe7…):

| file | gate | what | result |
|---|---|---|---|
| `decode.yihou-runshare.py` (+ `.diff`) | `YIHOU_DECODE_RUNNING_KV_SHARE=0.5` | user rule in `pop_preallocated`: admit only if (pool used − Σ running kv_allocated_len) + candidate < (1−share)·pool; FIFO break when not; rebootstrap exempt; escape when nothing is in transfer/waiting; logs `YIHOU-RUNSHARE block` every 100th block | functional (cap held at 0.50); C40 warm 21,271 = regression vs 22,114 |
| `decode.yihou-runshare2.py` (+ `.diff`) | + `YIHOU_RUNSHARE_MAX_WAIT_S=5` | v2: a non-fitting request is skipped (smaller ones behind it may enter) unless the blocked one waited > 5 s → FIFO break (anti-starvation) | functional (0.42 max); C40 fresh 20,680; warm A2 not run |
| `config.yihou.runshare.sh` / `runshare2.sh`, `mounts.yihou.decode-runshare*.tsv` | — | harness overlays (decode mount + env) | — |
| `config.yihou.mc4340n.sh` | — | mc4340 config with per-node image IDs for 174/029/137 | — |
| `config.yihou.mc4340prio.sh` | `MC_HIP_STREAM_PRIORITY=high` | KV-send copy-stream priority on the mc4340prio image (102 only) | prepared, not run |
