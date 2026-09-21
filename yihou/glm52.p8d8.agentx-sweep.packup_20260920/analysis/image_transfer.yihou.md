# Image transfer + verification — `infera-sglang:v0519-yihou-0917-nextnfix-hicache`

Task: get the docker image from `crsuse2-m2m-135` onto BOTH `crsuse2-m2m-137`
and `crsuse2-m2m-136`, and prove it arrived intact.

## Result: PASS on both nodes

| node | image id | sglang version | fusion marker¹ | pick_group_bytes² | _tiles_across_lanes³ | can_use_jit⁴ | verdict |
|---|---|---|---|---|---|---|---|
| source `-135` | `fd7220a57b7d` | 0.5.19.dev20260917+ga9fb1c3238 | — (source of record) | — | — | — | reference |
| `-137` (prefill) | `fd7220a57b7d` | 0.5.19.dev20260917+ga9fb1c3238 | PASS (1 hit) | PASS (2 hits) | PASS (2 hits) | PASS (2 hits) | **PASS** |
| `-136` (decode) | `fd7220a57b7d` | 0.5.19.dev20260917+ga9fb1c3238 | PASS (1 hit) | PASS (2 hits) | PASS (2 hits) | PASS (2 hits) | **PASS** |

All three image ids are identical: `fd7220a57b7d` (66.3 GB). Equal to the source
on 135 (verified read-only, not disturbed).

Marker strings (from mission spec), all located under
`/sgl-workspace/sglang/python/sglang/`:
1. `srt/models/glm4_moe.py`: `fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"`
2. `kernels/jit/csrc/kvcacheio/hicache.cuh`: `pick_group_bytes`
3. `kernels/ops/kvcache/hicache.py`: `_tiles_across_lanes`
4. `srt/mem_cache/pool_host/mha.py`: `can_use_jit = (_is_cuda or _is_hip)`

## Two points recorded for accuracy (not glossed)

1. **136's markers were checked IN THE CONTAINER ON 136 itself**, not inferred from
   image-id equality with 137. A throwaway CPU-only container was started on 136 and
   the four files were grepped directly inside it (command below). 137 was verified
   the same way, independently. Neither node's verdict rests on the other's.
2. **The 137 and 136 transfers were run sequentially, 137 first then 136.** Between
   them there was a gap during which 136 had not yet been started — this was the
   planned sequential order, **not a failure and not a stall**. 137's stream finished
   EXIT=0 ("Loaded image"), and 136's stream was launched immediately after. Both
   streams completed EXIT=0. Nothing failed and nothing was lost.

## Transfer method

Streaming `docker save` on 135 piped over ssh through the orchestration host into
`docker load` on the target — no 66 GB tar materialised on shared/NFS storage.
Sequential (137 fully done before 136 started), which is safer than parallel.
135 was read from only; no image/container on 135 or 138 was stopped, removed, or
pruned. The already-running P8D8 deployment on 137/136 was not disturbed —
`docker load` of identical layers is idempotent, and the marker checks used
`--rm` CPU-only containers that never touch the GPUs.

Disk pre-check (docker root `/mnt/m2m_nobackup`, threshold ~120 GB free):
137 = 8.7 TB free, 136 = 7.7 TB free — both well clear.

## Exact commands

Disk pre-check:
```
ssh crsuse2-m2m-137 "df -h /mnt/m2m_nobackup/docker"
ssh crsuse2-m2m-136 "df -h /mnt/m2m_nobackup/docker"
```

Source confirmation on 135 (read-only):
```
ssh crsuse2-m2m-135 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache \
  --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'"
# -> infera-sglang:v0519-yihou-0917-nextnfix-hicache fd7220a57b7d 66.3GB
```

Transfer (run once per target, sequentially):
```
set -o pipefail
ssh crsuse2-m2m-135 "docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache" \
  | ssh crsuse2-m2m-137 "docker load"    # -> Loaded image: ...  EXIT=0
ssh crsuse2-m2m-135 "docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache" \
  | ssh crsuse2-m2m-136 "docker load"    # -> Loaded image: ...  EXIT=0
```

Per-node id confirmation:
```
ssh crsuse2-m2m-137 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache --format '{{.ID}} {{.Size}}'"
ssh crsuse2-m2m-136 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache --format '{{.ID}} {{.Size}}'"
# both -> fd7220a57b7d 66.3GB
```

In-container marker check (run on 137 and on 136, each on its own node):
```
ssh crsuse2-m2m-<NODE> 'docker run --rm --entrypoint bash \
  infera-sglang:v0519-yihou-0917-nextnfix-hicache -lc "
S=/sgl-workspace/sglang/python/sglang
python3 -c \"import sglang;print(sglang.__version__)\"
grep -c '\''fused_shared_experts_architecture = \"GlmMoeDsaForCausalLMNextN\"'\'' \$S/srt/models/glm4_moe.py
grep -c pick_group_bytes      \$S/kernels/jit/csrc/kvcacheio/hicache.cuh
grep -c _tiles_across_lanes   \$S/kernels/ops/kvcache/hicache.py
grep -c '\''can_use_jit = (_is_cuda or _is_hip)'\'' \$S/srt/mem_cache/pool_host/mha.py
"'
# 137 -> 0.5.19.dev20260917+ga9fb1c3238 / 1 / 2 / 2 / 2
# 136 -> 0.5.19.dev20260917+ga9fb1c3238 / 1 / 2 / 2 / 2
```
