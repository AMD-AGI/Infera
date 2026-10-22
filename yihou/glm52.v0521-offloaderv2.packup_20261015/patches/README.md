# Patches

## 0001 — YIHOU-OFFLOAD-ATTRS (`offloader.yihou-attrs.py`)

- **What:**
  - `_CpuParamOffloader.create_device_tensor` returns `self._param.to("cuda", non_blocking=True)`.
  - It now also sets `t.is_shuffled = True` when the host parameter carries `is_shuffled`.
  - See `0001-offloader-carry-is_shuffled.diff` against the image file `offloader.py.orig`.
- **Why:** aiter's fused_moe selects its kernel with `getattr(w, "is_shuffled")`. Without the patch, an onloaded device tensor lacks the attribute, so a different kernel path is taken for shuffled weights. This is from the code reading in `analysis/research/offv2_impl.md`; no unpatched round was run to show the effect.
- **How:**
  - `build_offpatch.yihou.py` reads the image's `offloader.py` (from `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/utils/offloader.py`), replaces the method and writes `offloader.yihou-attrs.py`.
  - `mounts.yihou.offload-attrs.tsv` bind-mounts it on the prefill leg at `/sgl-workspace/sglang/python/sglang/srt/utils/offloader.py`.
  - The patched file's sha was checked against the copy mounted in the 136 image during preparation (`analysis/ledger/working_process.md`).
- **Context:**
  - The prefill leg only, and only with `--offload-*` flags. The patch is a no-op when OffloaderV2 is not enabled.
  - Decode does not mount it.
