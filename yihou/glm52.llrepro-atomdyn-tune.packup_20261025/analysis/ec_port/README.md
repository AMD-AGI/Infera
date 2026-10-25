# ec_port.yihou — decode empty_cache patch ("03 ec2-allrank") ported to the llying kit image (2026-10-24)

Source patch: git repo infera.glm52.pd commit 2ecf34d3,
`yihou/glm52.dcp4-decode-gmu.packup_20261023/patches/03-ec2-allrank.decode.diff`, written for sglang `ab7a898b77`
(overlay `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77-memlog-ec2.yihou`; its decode.py differs
from the base overlay by exactly the two 4-line hunks, checked with diff).

Target: image `infera-sglang:v0522-fdbafb60`, `/sgl-workspace/sglang` HEAD e9b22a0818 (git log in image).
The kit decode.py differs from ab7a898b77 decode.py (121 diff lines) -> do NOT mount the ab7a898b77 file.
The patch anchor (`self.process_decode_queue()` + blank + `# Get the next batch to run`) occurs exactly twice in the
kit file (event_loop_normal_disagg_decode line 3049, event_loop_overlap_disagg_decode line 3096); no conflicts.

Files
- `decode.kit-e9b22a08.orig.py`  — copied out of the image (docker run --rm ... cat).
- `decode.kit-e9b22a08.ec2.py`   — orig + the same two hunks (python3 -m py_compile OK on host).
- `03-ec2-allrank.decode.kit-e9b22a08.diff` — diff -u orig ec2 (only the two hunks).

Behaviour (unchanged from the source patch): every 8 decode scheduler loop iterations, on every scheduler
process (each DP rank in DPA4), if torch.cuda.mem_get_info() free < 6 GiB -> torch.cuda.empty_cache().

How to apply (NOT applied; needs a kit edit because DECODE_MOUNT_EXTRA takes one -v and llr7/llr8 already use it
for the KF overlay):
  kit engine/worker.sh:146, replace
    [[ "$role" != decode || -z "${DECODE_MOUNT_EXTRA:-}" ]] || mounts+=(-v "$DECODE_MOUNT_EXTRA")
  with
    [[ "$role" != decode ]] || for m in ${DECODE_MOUNT_EXTRA:-}; do mounts+=(-v "$m"); done
  then in the site env:
    export DECODE_MOUNT_EXTRA="/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay:/yihou_overlay:ro /shared_nfs/yihou_glm52/llrepro_20261023/ec_port.yihou/decode.kit-e9b22a08.ec2.py:/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py:ro"
  Verify after start: docker exec <decode> grep -c "yihou gmu" /sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py  (expect 2)

Caveats (measured elsewhere, not here): the source patch was measured only on a DCP4 decode (TP4 EP4 DCP4) at
0.88/0.92 same-node; patch 03 alone was never run (S2n/S2m also had patches 01+02). It was not run on DPA4 or at 0.95.
