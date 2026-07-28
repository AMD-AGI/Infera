---
name: mooncake-mode-select
description: >-
  Use when bringing up SGLang (or vLLM) PD disaggregation with the Mooncake KV
  transfer backend and you must decide HOW Mooncake registers the KV cache for
  RDMA — bare ibv_reg_mr+peer-mem, ibv_reg_dmabuf_mr (GPUDirect dma-buf), or a
  capped-KV fallback. Trigger whenever the user mentions mooncake, PD / prefill-
  decode disaggregation, KV transfer, ibv_reg_mr / ibv_reg_dmabuf_mr, dma-buf,
  peermem/ODP, MOONCAKE_DISABLE_HIP_DMABUF, or hits KV-registration failures
  (EFAULT on a device pointer, doubled KV pool, SIGSEGV/HIP-209 at KV register,
  "remote mooncake session not alive"), or is choosing which RDMA NIC to pin.
  Runs an environment probe first, recommends a mode with exact env + launch
  flags, RED-FLAGS any bandwidth regression, and ALWAYS asks the user to confirm.
  Grounded in the validated AMD spur (crsuse2-m2m / MI355X + ionic + mlx5) runs.
---

# Mooncake KV-registration mode selection

Mooncake moves the KV cache between prefill and decode over RDMA. To do that it
must **register** the KV cache (GPU VRAM) as an RDMA memory region — and there are
three fundamentally different ways to register device memory, each safe only under
specific kernel + NIC conditions. Pick the wrong one and the engine either EFAULTs
at registration, or silently pins and **doubles** the KV pool until VRAM/KFD is
exhausted (SIGSEGV / HIP-209), or falls back to TCP and crawls.

This skill: **probe the node → recommend a mode → confirm with the user → emit the
exact env + launch flags.** The heavy lifting is one script; your job is to run it,
explain the recommendation, and get the user's decision.

## The three modes (the declaration)

| Mode | Registration | Precondition | KV pool | Image |
|------|--------------|--------------|---------|-------|
| **A** `peermem` | bare `ibv_reg_mr` + a peer-memory kernel module | a peer-mem module is loaded (`nvidia_peermem` / `amdp2p` / MOFED peer-mem client) | not pinned, not doubled; **every rail usable** | stock |
| **B** `dmabuf` | `ibv_reg_dmabuf_mr` (GPUDirect via dma-buf) | **no** peer-mem, but a NIC with **ODP** (on-demand paging, e.g. Mellanox mlx5) | ODP → dynamic attach → not pinned/doubled, **but forced onto that one NIC** | **dma-buf rebuild** (`Dockerfile.sglang.dmabuf`) |
| **C** `cap-kv` | dma-buf with a **capped** KV pool (or a driver-bug workaround) | no peer-mem **and** no ODP NIC | pinned+doubled, so KV must be small enough to fit | **dma-buf rebuild** |

**Default / no-surprise order:** A if peer-mem is present, else B if an ODP NIC
exists, else C. This matches the in-repo default: when peer-mem is available the
image runs bare `ibv_reg_mr` (`MOONCAKE_DISABLE_HIP_DMABUF=1`, see
`infera/engine/rocm_rdma_env.py`). **But always ask the user** — the node can't see
intent (an accepted bandwidth hit, a NIC reserved for something else, a known-bad
rail).

> **Mode C is a STUB.** The detector emits a C recommendation with a `# TODO(cap-kv)`
> placeholder; the exact cap (e.g. `--max-total-tokens`, reduced
> `--mem-fraction-static` so weights + 2×KV-pin fit) is left for a follow-up agent.
> If you land in C, surface the blocker and stop — don't invent the cap.

## Workflow

Create a todo per step so nothing is skipped.

### 1. Probe the environment

Run the detector **inside the engine container on the target GPU node** (ODP and
the dma-buf engine check need `ibv_devinfo` / the mooncake `.so`, which live in the
image — on a bare login host ODP always reads absent). The standard spur pattern is
`spur exec <jobid> docker exec <ctr> …` (see the `spur-interactive-debug` skill).

```bash
# colorized CLI table: node facts + per-NIC capability + ALL modes ranked
python -m infera.tools.preflight.mooncake_mode

# machine-readable + Markdown for an agent / to paste; --out-dir drops all 3 files
python -m infera.tools.preflight.mooncake_mode --emit json --quiet > /tmp/mc_mode.json
python -m infera.tools.preflight.mooncake_mode --out-dir /tmp/mc_mode   # .json/.md/.txt
```

Exit code: `0` if the best pick is a viable full-KV mode (A or B, no user cap), `2`
if the only viable path needs a KV cap (Mode C) or nothing is viable — i.e. `2`
means a human decision (or an image rebuild) is required.

The tool **enumerates all three modes** — each marked viable or blocked *with the
reason* — rather than emitting one recommendation, so you (or the user/agent) see
the whole option space and can pick. It has two layers:

- **Layer 1 (facts):** per RDMA device — vendor/driver, netdev + IPv4, link speed,
  **ODP**, whether **dma-buf is registerable** on it, routable RoCE-v2 **GID index**,
  PCI **BDF**, NUMA — plus node-level **peer-mem**, kernel `CONFIG_PCI_P2PDMA` (now
  also probed via `/proc/kallsyms` when `/boot/config` isn't mounted in the
  container), GPU count/gfx, and the **engine image** dma-buf capability (expected:
  the infera sglang image supports every mode; the probe confirms `USE_HIP_DMABUF`).
- **Layer 2 (options):** modes A/B/C ranked **safety-first, then bandwidth** — each
  with viable/blocked + why, the selected NICs + selection rule, env dict, launch
  flags, aggregate Gb/s, KV-pool full-vs-capped, pin none-vs-2×, and whether a
  **user KV-cap confirmation** is required.

The ranking is the load-bearing bit: a no-pin/no-cap mode on a slower ODP NIC
outranks a faster mode that needs a KV cap — reproducing the crsuse choice of one
200G mlx5 (Mode B) over eight 400G ionic (Mode C). Hand the JSON/Markdown to the
user or an agent; they copy the env + flags for the mode they pick.

If the script isn't importable (e.g. infera not pip-installed in that container),
it's a single self-contained file — copy
`infera/tools/preflight/mooncake_mode.py` in and run it directly with `python3`.

### 2. Read the options (ranked) and the warnings

Each mode block gives viable/blocked + why, the selected NIC(s), whether the
dma-buf image is required, the exact **env dict**, and **launch flags**; the top
`★ best` is the safety-first pick. Read the blocked modes too — their reason is
what tells the user "why not the faster one". Warnings carry severity:

- **`*** PERFORMANCE REGRESSION ***`** — Mode B is forcing all KV onto an ODP NIC
  that is slower and/or fewer than the node's fast rails (the classic spur case:
  one 200G mlx5 vs eight 400G ionic). You **must** surface this to the user in
  **bold red**, verbatim, because it's an accepted-tradeoff decision, not a bug.
- **`*** BLOCKER ***`** — the recommended path can't run as-is (dma-buf image not
  built; or Mode C, no safe full-KV path). Stop and report; don't work around it.
- `[verify]` — couldn't confirm something (e.g. the engine's `USE_HIP_DMABUF`);
  give the user the one-line command to check.

### 3. Confirm the mode with the user — ALWAYS

Present the recommendation **and the alternatives** and ask. Use `AskUserQuestion`
with the recommended mode first (label it "Recommended"). Never launch on the
recommendation alone. Reasons the user may override:

- Accept a bandwidth regression (Mode B on the slow NIC) they already signed off on.
- Force a specific NIC / rail different from the auto-pick.
- Choose Mode A even where B looks safer, or vice-versa, for their own reasons.

If a **perf-regression** warning fired, the confirmation prompt must show it in
**bold red** so the choice is informed.

### 4. Emit the env + launch flags

Once the user picks, print the concrete launch env and flags for that mode. For a
copy-pasteable sglang PD leg, `references/launch_recipes.md` has the validated spur
commands for A, B, and the single-node variants; graft the mode's env dict + flags
onto the project's PD launch script (`examples/deepseek_v4/engine/pd_mooncake/…`).

**Never enable `MC_ENABLE_HIP_TRANSPORT` for cross-node PD** — HIP IPC handles
can't cross process/host boundaries; cross-node KV must stay on RDMA. (It also
fails single-node between two separate P/D processes — see the recipes doc.)

## Deep background

Only load these when you need the *why*, not just the *what*:

- `references/mode_decision.md` — the full decision tree, each precondition's probe
  method, and how the recommendation is computed. Read when a recommendation looks
  wrong or you're extending the logic (e.g. filling Mode C).
- `references/launch_recipes.md` — validated spur launch env + flags per mode, the
  KV-not-doubled check, GID indices, and the ionic/mlx5 gotchas.
- `references/spur_ground_truth.md` — the concrete `crsuse2-m2m` hardware facts and
  the validated-run table this skill is grounded in.
