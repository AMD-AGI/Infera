# Patches

| file | what | applies to |
|---|---|---|
| `0001-idlefix-vs-image-9f5d8712.diff` | the fix as mounted in idxf1a/b/c and dbge6a (`diff -u` image file vs `scripts/mounts-src/eager_runner.yihou-idlefix.py`) | `/sgl-workspace/sglang` in image `9d755dd6a7bf` (sglang 9f5d8712 + local diff) |
| `0002-idlefix-vs-upstream-main-93088901.diff` | the same change ported to upstream `origin/main` 93088901 `eager_runner.py` (imports already include `get_forward_context` there); comment shortened, no YIHOU tag | sgl-project/sglang main (PR draft; py_compile only, not run) |
| `0000-image-local-diff.vs-9f5d8712.diff` | reference only: the uncommitted modifications already inside the image (`git diff` in the image's tree) | not to apply; documents the base |

## 0001 / 0002 — EagerRunner idle: also init the active forward-context backend

- **What**: in `EagerRunner._execute_idle`, when `forward_batch.batch_size > 0`, after
  `model_runner.attn_backend.init_forward_metadata(forward_batch)`, call `init_forward_metadata(forward_batch)` on
  `get_attn_backend()` too when a forward context is active and its backend is a different object.
- **Why**: layers read the backend of the active forward context. The EAGLE v2 idle draft runs each step under
  `ForwardContext(attn_backend=draft_attn_backend.attn_backends[i])`, while the draft runner's `attn_backend` is the draft-extend backend.
  The step backend therefore kept the `forward_metadata` of an earlier real batch; the DSA indexer's k-only path ranked 2048-wide dummy
  logits with stale lengths (up to 104,740) and `coop_topk_kernel` read past the buffer → decode "Memory access fault".
  Reached only with `index_share_for_mtp_iteration=true` (a seedless PD batch forces the whole DP group into eager drafting).
- **How applied**: bind-mounted over the image file (`mounts.yihou.v2-xferfix2-gtt-idlefix.tsv`, overlay `config.yihou.idlefix.sh`); no rebuild.
- **Context / evidence**: before: 8 faulting rounds, BAD k-only calls in dbge4a/4b/5a/5b. After: dbge6a full A1 with guards 0 BAD / 0 MAF;
  idxf1a A1+A2 0 MAF. See `../README.md` evidence table.
