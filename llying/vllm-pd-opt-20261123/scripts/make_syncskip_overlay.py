#!/usr/bin/env python3
"""Build the sync-skip overlay of vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (vLLM 0.34.0).

  make_syncskip_overlay.py ORIGINAL OUT

ROCMAiterMLASparseMetadataBuilder.build() reads seq_lens back to the CPU (a GPU-to-CPU sync) only to fingerprint the
persistent MLA work metadata, whose inputs are the per-row KV lengths clamped to topk_tokens. Under MTP the drafter
rebuilds this metadata eagerly for every draft step (the backend has no update_draft_decode_metadata), so every decode
step pays one sync per draft step. When seq_lens_cpu_upper_bound shows that every row's context already reaches
topk_tokens (plus its query), the clamped lengths are known without the sync; padded rows (upper bound 0, query 0) are
0 on the GPU too (prepare_pos_seq_lens pads them with 0). The fingerprint is then identical to the synced one, so the
relaunch decision and the metadata are unchanged. VLLM_ROCM_SPARSE_MLA_SYNC_SKIP=0 restores the stock path.
It also drops the stream synchronize after a get_mla_metadata_v1 relaunch (the drafter's prefill and decode steps share
one builder, so the fingerprint flips twice per decode step); VLLM_ROCM_SPARSE_MLA_META_SYNC=1 restores it.
"""
import sys

OLD = """            with gpu_sync_allowed():
                seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].cpu().numpy()
            clamped_seq_lens = np.minimum(
                seq_lens_cpu,
                self.topk_tokens,
            )
            clamped_context_lens = np.minimum(
                seq_lens_cpu - seg_lengths,
                self.topk_tokens,
            )
"""

NEW = """            clamped_seq_lens = _saturated_clamped_lens(
                common_attn_metadata.seq_lens_cpu_upper_bound,
                num_reqs,
                seg_lengths,
                self.topk_tokens,
                _SYNC_SKIP_SLACK,
            )
            if clamped_seq_lens is not None:
                clamped_context_lens = clamped_seq_lens[1]
                clamped_seq_lens = clamped_seq_lens[0]
            else:
                with gpu_sync_allowed():
                    seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].cpu().numpy()
                clamped_seq_lens = np.minimum(
                    seq_lens_cpu,
                    self.topk_tokens,
                ).astype(np.int32, copy=False)
                clamped_context_lens = np.minimum(
                    seq_lens_cpu - seg_lengths,
                    self.topk_tokens,
                ).astype(np.int32, copy=False)
            _count_sync_skip(clamped_seq_lens is not None)
"""

OLD_META_SYNC = """                torch.cuda.current_stream(self.device).synchronize()
                self._prev_metadata_key = metadata_key
"""

NEW_META_SYNC = """                if _META_SYNC:
                    torch.cuda.current_stream(self.device).synchronize()
                self._prev_metadata_key = metadata_key
"""

HELPER_ANCHOR = "\n@dataclass\nclass ROCMAiterMLASparseMetadataBuilder("

HELPER = '''

_SYNC_SKIP = os.environ.get("VLLM_ROCM_SPARSE_MLA_SYNC_SKIP", "1") != "0"
# get_mla_metadata_v1 writes device buffers that only later kernels on the same
# stream read (the dense AITER MLA backend calls it on every build with no sync),
# so the host does not need to wait for it.
_META_SYNC = os.environ.get("VLLM_ROCM_SPARSE_MLA_META_SYNC", "0") != "0"
# seq_lens_cpu_upper_bound assumes every draft of the step was accepted, and a
# draft step adds its index on top: allow a few spec steps of overshoot.
_SYNC_SKIP_SLACK = 32
_sync_skip_counts = [0, 0]


def _saturated_clamped_lens(seq_lens_cpu_upper_bound, num_reqs, seg_lengths, topk_tokens, slack):
    """Clamped (seq, context) lengths without reading seq_lens back, or None.

    Real rows qualify when their upper bound minus `slack` still reaches
    topk_tokens + query length (both clamps then give topk_tokens). Padded rows
    (upper bound 0, no query) are 0 on the GPU, so they clamp to 0.
    """
    if not _SYNC_SKIP or seq_lens_cpu_upper_bound is None or num_reqs <= 0:
        return None
    ub = seq_lens_cpu_upper_bound[:num_reqs].numpy()
    if ub.shape[0] != num_reqs or seg_lengths.shape[0] != num_reqs:
        return None
    pad = (ub == 0) & (seg_lengths == 0)
    real = ~pad
    if not real.any():
        return None
    if not bool(np.all(ub[real] - slack >= topk_tokens + seg_lengths[real])):
        return None
    lens = np.where(real, topk_tokens, 0).astype(np.int32)
    return lens, lens


def _count_sync_skip(skipped):
    _sync_skip_counts[0 if skipped else 1] += 1
    n = _sync_skip_counts[0] + _sync_skip_counts[1]
    if n % 20000 == 0:
        logger.info(
            "ROCm sparse MLA metadata: %d of %d builds skipped the seq_lens sync",
            _sync_skip_counts[0],
            n,
        )
'''


def main():
    src, out = sys.argv[1], sys.argv[2]
    s = open(src).read()
    assert s.count(OLD) == 1, "anchor (seq_lens sync block) not found exactly once"
    assert s.count(HELPER_ANCHOR) == 1, "anchor (builder class) not found exactly once"
    assert s.count(OLD_META_SYNC) == 1, "anchor (synchronize after get_mla_metadata_v1) not found exactly once"
    s = s.replace(OLD, NEW)
    s = s.replace(OLD_META_SYNC, NEW_META_SYNC)
    s = s.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    if "\nimport os\n" not in s:
        s = s.replace("\nimport numpy as np\n", "\nimport os\n\nimport numpy as np\n", 1)
    assert "\nimport os\n" in s, "could not add 'import os'"
    assert "logger = " in s, "module has no logger"
    open(out, "w").write(s)


if __name__ == "__main__":
    main()
