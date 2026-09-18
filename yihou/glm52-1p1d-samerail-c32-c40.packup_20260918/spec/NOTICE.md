# GPU-clearing notice — yihou 1P1D C32 bring-up

Authorisation: the user of this session explicitly chose "clear all non-system
GPU load on crsuse2-m2m-135 and -138, including another person's work, but take
an evidence snapshot and post this notice first." The evidence snapshot was
taken first. Timestamps in UTC. Operator: `yihou` (uid 50112975).

This notice records **what actually happened**, not what was planned.

Final node-set decision (mission.md, updated after these findings): keep
**135 (prefill) + 138 (decode)** on **devices 4,5,6,7 only** (P4D4, CONC=32),
sidestepping 135 GPU[1]'s k8s pod. 139/140 rejected (docker masked — below).

## Node crsuse2-m2m-138 — limou container STOP was EXECUTED

- Container `glm52pd-decode-0-limou`, image
  `infera/engine-sglang:sgl-sep01-aiter-sep06`, startedAt 2026-09-17T04:49:00Z
  (Up ~23 h), held ~85–86 % VRAM on all 8 GPUs. Owner inferred from name suffix
  `-limou`. Plain Docker container.
- Action **executed**: `docker stop -t 120 glm52pd-decode-0-limou` → exit 0,
  graceful. Container now `exited`, finishedAt **2026-09-18T04:10:09Z**. All 8
  GPUs on 138 subsequently read 0 % VRAM.
- The leader later sent a HOLD ("do not stop limou yet"); it arrived **after**
  the stop had already completed, so it could not be honoured. The colleague's
  ~23 h warm process is gone (irreversible). Under the final decision 138 is our
  decode node and needed its GPUs freed regardless, so the outcome stands.
- The container was **not** removed (`docker rm` NOT run) and no image was
  `docker rmi`'d. It could be restarted with `docker start …` (a COLD relaunch,
  not a resume). Left exited.
- Evidence: `138-limou-inspect.json`, `138-limou-tail.log`,
  `138-rocm-smi-before.txt`, `138-procs-before.txt`, `138-docker-ps.txt`,
  `138-stop.txt`, `138-rocm-smi-after.txt`.

## Node crsuse2-m2m-135 — NOT CLEARED (blocked; leader verified first-hand)

VRAM on 135 GPU[1] is a **live k8s pod**, not leftover cruft:

- PID 813732 `root` — `vllm serve Qwen/Qwen3-32B --port=8200 …
  --kv-transfer-config {… p2p host 10.42.205.104:7777 …}`, uptime ~22 days.
- PID 818623 `root` `VLLM::EngineCore` (child).
- cgroup `/kubepods/pod5d84e491-6607-4d85-a4de-a3be9080683a/…`.
- Holds **96.94 GB of 309.22 GB** on GPU[1] (leader-verified).

Not cleared because: as `yihou`, `kill -0 813732` → "Operation not permitted";
no passwordless sudo; `crictl` needs sudo. Killing a bare k8s pod PID is also
wrong (kubelet restart) — correct action is `kubectl delete` by the namespace
owner. **Left untouched by design**: the final plan uses devices 4-7, avoiding
GPU[1] entirely, so this pod does not need to be cleared. `check_nodes.sh` will
report GPU[1] busy — that is expected and must NOT be hidden by raising
thresholds. jiejizha's tmux + monitor scripts on 135 hold no VRAM.
Evidence: `135-rocm-smi-before.txt`, `135-procs-before.txt`, `135-docker-ps.txt`,
`135-k8s-pod.txt`.

## Nodes crsuse2-m2m-139 / -140 — GPUs idle but DOCKER MASKED (rejected)

Probed as candidate replacement nodes. All 8 GPUs at 0 % VRAM, no containers,
**but** `docker.service` is **masked / inactive** on both:

- 139: masked; drop-ins `10-spur-authz.conf` + **`99-fleet-hardpull-preserve-
  20260917.conf`** (2026-09-17 19:41 — fleet automation appears to be
  deliberately preserving this node). No `/var/run/docker.sock`.
- 140: masked; only `10-spur-authz.conf`. Stale socket from Aug 27.
- `systemctl start docker` as `yihou` → "Interactive authentication required"
  (no sudo). Cannot enable docker without cluster-admin.
- Consequence: idle GPUs do **not** make them usable. This is why the plan
  stayed on 135+138 rather than moving.
Evidence: `139-docker-daemon.txt`, `140-docker-daemon.txt`,
`check_nodes-139-140.txt`.

## Nothing was removed or `docker rmi`'d anywhere. No file outside `…/yihou-1p1d-c64/` was touched.
