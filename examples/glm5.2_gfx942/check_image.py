"""Report which of this repo's sglang patches are present in the running image.

Each needle is a line the corresponding patch script writes and nothing else in
the stock tree produces, so a "no" here means that patch did not go in -- which
the engine does not otherwise report.
"""

import pathlib

import sglang

ROOT = pathlib.Path(sglang.__file__).parent

CHECKS = [
    ("sglang_rocm/host_alloc", "srt/mem_cache/pool_host/common.py", "hipHostMalloc"),
    ("sglang_rocm/staged_wb", "srt/mem_cache/pool_host/mla.py", "is_cuda"),
    ("sglang_disagg/early_send", "srt/disaggregation/mooncake/conn.py", "wait_event"),
]

for name, rel, needle in CHECKS:
    path = ROOT / rel
    if not path.exists():
        print(f"{name:28s} {rel:44s} FILE-MISSING")
        continue
    print(f"{name:28s} {rel:44s} {'YES' if needle in path.read_text() else 'no'}")
