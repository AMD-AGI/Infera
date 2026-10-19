"""CPU check of the piggyback branch in prefill_delayer.llying-piggyback.py.

The cross-rank all-gather is replaced by a fixed tensor (one row per DP rank:
prefillable, watermark, running, max_prefill_bs, queue, timeout, mid_chunk).
Run inside the image: python3 test_piggyback_delayer.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "pd_piggyback", Path(__file__).with_name("prefill_delayer.llying-piggyback.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def delayer(piggyback: bool, rows):
    d = object.__new__(mod.PrefillDelayer)
    d._max_delay_passes = 120
    d._token_usage_low_watermark = None
    d._queue_trigger_enabled = False
    d._max_delay_ms = 5000.0
    d._prefill_max_requests = None
    d._piggyback = piggyback
    d.num_dp_ranks = len(rows)
    d.attn_dp_enabled = True
    d.skip_first_delayer = False
    d._gather_info = lambda **_: torch.tensor(rows, dtype=torch.int64)
    return d


def reason(piggyback, rows, prev=None):
    out = delayer(piggyback, rows)._negotiate_should_allow_prefill_pure(
        prev_state=prev, local_prefillable=True, token_usage=0.5, max_running_requests=128
    )
    return out.output_allow, out.output_reason


#        prefillable wm run bs queue timeout mid_chunk
chunk = [[1, 0, 8, 2, 3, 0, 1],   # DP0 continues a chunk
         [1, 0, 8, 2, 4, 0, 0],   # DP1 has an admissible request
         [0, 0, 8, 2, 0, 0, 0],   # DP2, DP3 have nothing
         [0, 0, 8, 2, 0, 0, 0]]
no_chunk = [r[:6] + [0] for r in chunk]
checks = {
    "piggyback on, a rank mid-chunk -> allow": (reason(True, chunk), (True, "piggyback")),
    "piggyback off, a rank mid-chunk -> delay": (reason(False, chunk), (False, "delay")),
    "piggyback on, nobody mid-chunk -> delay": (reason(True, no_chunk), (False, "delay")),
    "piggyback on, all prefillable -> no_wait": (
        reason(True, [[1] + r[1:] for r in chunk]), (True, "no_wait")),
    "piggyback on, cap reached -> wait_timeout": (
        reason(True, no_chunk, mod._State(delayed_count=119)), (True, "wait_timeout")),
}
failed = 0
for name, (got, want) in checks.items():
    ok = got == want
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got}")
sys.exit(1 if failed else 0)
