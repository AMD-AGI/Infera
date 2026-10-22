# index_share decode MAF — prep notes (2026-10-20 06:xxZ, before the concentrated debug)

## Source trees (local, /tmp/yihou/idxshare)
- sglang clone (blobless) + worktrees: sgl-image-9f5d (the image's actual HEAD 9f5d8712514c, 2 commits after
  11b1fb21de), sgl-v0521-11b1, sgl-v0519-a9fb (v0.5.19 image base a9fb1c3238). 1205 commits a9fb..11b1.
- The image is NOT clean upstream: `git status` in the image shows local modifications (image_v0521_local.diff,
  353 lines): dsa_backend.py (GLM52_BUG2B `_glm52_match_page_table_rows` trim/edge-pad of page_table rows to the
  top-k row count; eager max_seqlen_k = req_to_token width; draft_extend_v2 no host mirrors), dsa_indexer.py
  (GLM52_P1V2/P1V3 aiter row trim/clip; not our triton path?), speculative_hook.py (hip rejection sampling off on
  PD decode), glm4_moe.py (NextN fused shared experts), prefill.py, pool_host/mla.py, serving_responses.py.
  Plus our runtime mounts (conn xferfix2, prefill nonfinal-event, forward_mla_rocm noabsorbcfg, kv_events noblock,
  pool_host common PR42553, offloader attrs on prefill).

## Kernel log (first-hand, via a CAP_SYSLOG container on m2m-250)
| time (dmesg) | GPU PCI | addr (page) | client | PERMISSION_FAULTS | RW |
|---|---|---|---|---|---|
| 02:50:52 g4n2 | 0003:00:02.0 | 0x7b3ff5400000 | UTCL2 / TCP (0x8) | 0x3 | 0 (read) |
| 04:09:13 g4n2b | 0003:00:03.0 | 0x7b40f0600000 | TCP | 0x5 | 1 (write) |
| 05:34:03 g4n4 | 0003:00:04.0 | 0x7e9b9c200000 | TCP | 0x3 | 0 (read) |
TCP = shader (compute kernel) memory access, not SDMA/DMA -> a kernel touched an address with no valid mapping
(WALKER_ERROR 0, MAPPING_ERROR 0, PERMISSION_FAULTS set). 2 MiB-aligned page bases. dmesg clock ~49 s ahead of
the server log. spur-exec root has no CAP_SYSLOG; `docker run --cap-add SYSLOG` works.

## Prior work on this exact bug (read before acting)
- yihou/dsa.topk.indexer.bug.analysis.md (v0.5.19 tilelang, 09-19).
- infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009/debug/working_process.md (spur):
  R004 c44c (v0521, idx true, C44) decode MAF at 6 min; host stack at fault in fused_dsa_draft_extend_metadata
  (async, not necessarily the faulting kernel). R009: lean ROCm debug agent validated (names kernel + PC under
  graph replay, ~1.00x). R029: seed path read; "kernel reads past the seed" refuted for the triton consumer;
  stale-seed risk noted; upstream #32209 (DP-padded eager draft, OPEN). R031 e2d: idx true + seedguard(count) +
  decode debug agent at C40 ran 60 min, 0 MAF, bad_entries 0 in ~1.24M seed rows.
- Overlay H/config.yihou.seedguard.sh + mounts.yihou.v2-regbase-seedguard.tsv (older mount base; needs merging
  with xferfix2 + gtt mounts and a check that its eagle_worker_v2 copy matches the image file).

## Upstream (gh, 2026-10-20)
#39517 OPEN, #37648 OPEN (GLM-5.3 TP8 EAGLE MAF), #37478 OPEN, #32209 OPEN (PD decode hang DP attn GLM-5.2 MTP),
#32527 closed (deadlock with index_share), #36598 open.

## Proposed first experiment (after g4n3/g4n3b)
idx true, C56, both OAI nodes, decode lean debug agent ON (+ seedguard count, + dmesg capture) -> faulting kernel
name / PC / disassembly within ~30-45 min. Then read that kernel's inputs backward (root-cause tracing).
