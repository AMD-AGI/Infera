# Node preflight — 1P1D on 135 (prefill) / 138 (decode), GPU 2,3,4,5

Date: 2026-09-18. Inspected from control node `crsuse2-slog-006` over ssh.
Read-only inspection only; nothing on the hosts was mutated.

## Verdict

- Devices **2,3,4,5 idle on BOTH nodes**: YES.
- Removed: **nothing** — no `yihou`-named containers exist on either node.
- `check_nodes.sh`: 135 ERROR (only GPU[1], expected), 138 PASS (with one warning, below).
- Rails `ionic_2..5`: all `PORT_ACTIVE`, Ethernet, on both nodes.

## Leftover containers

No containers whose name contains `yihou` on either node, so there was nothing
of ours to remove. All existing containers belong to other people / other work
and are OFF-LIMITS (left untouched). Lists saved to `containers-<node>.txt`.

- 135: `crusoe-vector` (Up, infra), `glm52-server`, `glm52-main-prstack-c8`,
  `topk-dev`, `glm52-qr_int4` — all Exited except crusoe-vector.
- 138: `glm52pd-decode-0-limou` — Exited (0) 5h ago, image
  `infera/engine-sglang:sgl-sep01-aiter-sep06`. Belongs to "limou", NOT ours.
  Stopped, holds no GPU. Name differs from our prefix `glm52-pd-yihou-1p1d`,
  so no name collision. `check_nodes.sh` flags it as a WARNING only. Left in
  place — leader to decide if it should be cleared before launch.

## GPU state (VRAM used / total, instantaneous use%)

Raw in `rocm-smi-<node>.txt`. Total per GPU = 309.22 GB. Idle baseline ≈ 297 MB.

### crsuse2-m2m-135
| GPU | VRAM used | VRAM% | use% | note |
|-----|-----------|-------|------|------|
| 0 | 297 MB | 0% | 0 | idle |
| 1 | 96.94 GB | 31% | 0 | **root K8s vLLM Qwen3-32B pod — expected, DO NOT TOUCH** |
| 2 | 297 MB | 0% | 0 | idle — target |
| 3 | 297 MB | 0% | 0 | idle — target |
| 4 | 297 MB | 0% | 0 | idle — target |
| 5 | 297 MB | 0% | 0 | idle — target |
| 6 | 297 MB | 0% | 0 | idle |
| 7 | 297 MB | 0% | 0 | idle |

### crsuse2-m2m-138
All 8 GPUs at ~297 MB / 0% — fully idle. Targets 2,3,4,5 idle.

## check_nodes.sh verdict

`./check_nodes.sh crsuse2-m2m-135 crsuse2-m2m-138` (full output in
`check_nodes.txt`).

- **135: ERROR — expected and acceptable.** The ONLY failing check is
  `gpu[1]: busy (VRAM=31%)`, which is the live root-owned Kubernetes vLLM pod.
  No threshold was relaxed to hide it; it is recorded as-is. All other GPUs
  (incl. targets 2,3,4,5) PASS idle.
- **138: PASS** on all GPU checks. One WARNING: the exited `glm52pd-decode-0-limou`
  container matches the decode-name heuristic. It is not ours and is stopped.

## Rail / fabric check

`ibv_devinfo -d ionic_{2,3,4,5}` on both nodes (saved to `rails-<node>.txt`):
all four ports `PORT_ACTIVE (4)`, link_layer Ethernet, on 135 and 138. The
known-defective `ionic_7` on 135 is not used by this run and was not inspected.
