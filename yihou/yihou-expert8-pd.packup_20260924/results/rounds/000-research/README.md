# Round000 — environment and preflight

## Target
Establish exact image, clean selected GPUs, source tools and matched-rail GPU transfer before model changes.

## Evidence and iterations
1. Both138/136 idle284MiB/card, matching image SHA. Direct SSH works; nodes absent from current Spur inventory.
2. Serena initial project autodetected bash only; symbol query returned isError=true despite process exit0. Added Python backend in child workspace config; second query returned isError=false with pinned Mooncake symbols.
3. Original GPU transfer probe target136 segfaulted before target.json. Driver also had a duplicate-key merge bug, fixed locally. No transfer success.
4. Hypothesis: torch/Mooncake initialization order. torch-first allocation/register passed but subsequent full probe still faulted at _Buf.fill. This hypothesis did NOT solve the problem.
5. Pure torch without Mooncake reproduced fill_ SIGSEGV on BOTH138 and136. Stream-handle lookup works. HIP_FORCE_DEV_KERNARG and scratch setting variations did not resolve it.
6. Native rocgdb stack enters libamdhip64 from torch FillFunctor. AMD_LOG_LEVEL=4 logs hip_fatbin.cpp:326 'Failed to unbundle code object' before SIGSEGV.
7. Differential: same image/GPU/1024-element fill with TMPDIR on NFS fails, workspace-scoped container tmpfs passes ('filled1.0'). Short NFS bind path still fails. Use tmpfs at a path inside workspace for runtime temporary files. This is a demonstrated workaround; exact NFS/COMGR mechanism remains unidentified.
8. Matched-rail GPU write+byte verification rerun with tmpfs; result pending.

## Commands/artifacts
- scripts/torch_fill.yihou.py: smallest reproduction.
- torch-fill-gdb.log, torch-fill-gdb-registers.log, torch-fill-hipdebug.log: native evidence.
- rdma-first/, rdma-torch-first/: failed preserved attempts.
- rdma-tmpfs/: corrected environment attempt.
- scripts/rdma_probe.yihou.py and scripts/probe_worker.yihou.py: exact probe drivers.

## Scope
No shared model/code edits; no137 interaction. Only task-owned containers created. All selected GPUs returned idle after failed subprocesses. Local shell runs orchestration only; GPU work inside Docker.
