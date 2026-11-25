"""Imported at interpreter start-up (vpd_thp_off.pth): prctl(PR_SET_THP_DISABLE) so this process and its children get no
transparent huge pages, hence no THP-driven direct compaction / khugepaged collapse of their anonymous memory.
From dev/pd_opt/vllm_dsv41_agentx (llying/vllm-1p1d-dsv41-crusoe-20261123/instr/thp_off.py), where it removed the
self-inflicted compaction behind KFD queue evictions (hangF: 0 compactions, 0 evictions in 84 min). No root needed."""
import ctypes

try:
    ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE
except Exception:  # noqa: BLE001
    pass
