---
name: infera-hygiene
description: Shared GPU-node manners and cluster traps — free VRAM and foreign containers before a run, ports, NFS/TMPDIR/HOME, preemption, never touching what is not yours
---

# infera-hygiene

Nodes are shared. A measurement taken while someone else's job ran on the same GPUs is void, and so is the
afternoon spent on it.

## Before a run

- Only the task's machines (`$AGENT_SYS_MACHINES`).
- **Foreign load**: `docker ps` (whose containers, on which GPUs), `rocm-smi --showpids`, `squeue -w <node>`.
  Snapshot it into the run directory. Do not kill or remove what you did not start: ask in progress.md (阻塞).
- **VRAM**: every GPU < 1% (or < 400 MiB twice, 60 s apart) before bringing a stack up; the kits' `wait_vram_free`
  reads `mem_info_vram_used` from sysfs. Release after `docker rm` takes 15-40 minutes (HiCache: up to an hour).
- **Ports**: avoid the ephemeral range 32768-60999 and Mooncake's random RPC ports 15000-17000.
- **Environment**: `python -m infera.tools.preflight --dump-path <dir>` (GPU, network, Mooncake); multi-node via
  `infera/tools/preflight/run_preflight_slurm.sh`. A GPU stuck at high utilisation with no process after a memory
  fault needs a node reset: report it, do not loop.

## While it runs

- Long work runs on the node as a self-contained chain (`setsid nohup ... </dev/null &`) with its own guard
  (see `infera-deploy`), and the agent waits for it with a background command that exits on a condition — the
  session can end, the chain keeps the node busy and logs why it stopped.
- Do not edit a running bash script in place: write a new file and `mv` it.
- Preemption (NODE_FAIL, MaxWall): containers may still show `Up` on a node that is no longer yours; check the
  allocation before trusting a result, and void a window it cut.

## Files

- Containers write as root into NFS with root_squash: fix ownership of what they wrote (`chown` from inside the
  container) before the next run reads it.
- `TMPDIR` on NFS breaks some tools (segfaults); `HOME` on a full quota breaks client installs. Keep caches on
  local disk; watch quota on long runs (a full disk corrupts `profile_export.jsonl`).
- Record image digests, not tags.
