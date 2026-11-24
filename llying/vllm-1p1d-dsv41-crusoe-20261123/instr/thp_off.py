"""Imported at interpreter start-up (thp_off.pth): prctl(PR_SET_THP_DISABLE) so this process and its children get no
transparent huge pages, and so cause no direct compaction for them (A/B for the whole-GPU stalls; no root needed)."""
import ctypes

try:
    ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE
except Exception:  # noqa: BLE001
    pass
