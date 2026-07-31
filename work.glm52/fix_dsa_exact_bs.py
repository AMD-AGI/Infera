"""Candidate fix for the aiter-DSA GPU fault on ROCm.

Bug: sglang's _ensure_aiter_dsa_decode_metadata_buffer reuses the persistent
metadata buffers whenever `capacity >= batch_size`. The buffers were sized once
at max_bs. But aiter's metadata/decode kernels compute their indexing from the
batch_size actually passed in (tile_cnt = batch_size * max_qo_tiles_per_batch),
while the buffer layout corresponds to max_bs -> stride mismatch -> OOB write
("Memory access fault ... Write access to a read-only page").

Repro (single GPU, no server), v0.5.16-rocm720-mi35x, GLM-5.2 shapes:
    buffers sized at MAX_BS=64, called with bs=40  -> FAULT
    buffers sized at bs=40,     called with bs=40  -> OK
    bs == MAX_BS (any value 33..64)                -> OK

Fix: require an EXACT capacity match before reusing.
"""
import re, sys, pathlib

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

OLD = """        if (
            self.aiter_dsa_work_metadata is not None
            and self.aiter_dsa_metadata_capacity >= batch_size
            and self.aiter_dsa_metadata_max_seqlen_q == max_seqlen_q"""

NEW = """        if (
            self.aiter_dsa_work_metadata is not None
            and self.aiter_dsa_metadata_capacity == batch_size
            and self.aiter_dsa_metadata_max_seqlen_q == max_seqlen_q"""


def main():
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else TARGET)
    s = p.read_text()
    if NEW in s:
        print("already patched"); return
    if OLD not in s:
        print("PATTERN NOT FOUND -- inspect manually", file=sys.stderr)
        i = s.find("aiter_dsa_metadata_capacity")
        print(s[max(0, i - 400): i + 400], file=sys.stderr)
        sys.exit(1)
    p.write_text(s.replace(OLD, NEW, 1))
    print(f"patched {p}: capacity >= batch_size  ->  capacity == batch_size")


if __name__ == "__main__":
    main()
