"""Wrap aiter's DSA entry points to log EXACT tensor shapes at every call, then hand
off to sglang.launch_server. With AMD_SERIALIZE_KERNEL=3 the last line printed before
the fault is the offending launch, with all its shapes -- no guessing.
"""
import sys, os

import torch

TAG = "[TRACE]"
_n = {"meta": 0, "dec": 0}


def _d(x):
    if isinstance(x, torch.Tensor):
        return f"T{tuple(x.shape)}:{str(x.dtype).replace('torch.','')}"
    return repr(x)


def install():
    import aiter
    from aiter.ops import attention as A
    import aiter.mla as M

    _get_meta = A.get_mla_metadata_v1
    _dec = M.mla_decode_fwd

    def get_mla_metadata_v1(*a, **k):
        _n["meta"] += 1
        i = _n["meta"]
        print(f"{TAG} meta#{i} qo={_d(a[0])} kv={_d(a[1])} klpl={_d(a[2])} "
              f"nhead={a[3]} bs_from_qo={a[0].shape[0]-1} "
              f"work_meta={_d(a[6])} work_info={_d(a[7])} work_indptr={_d(a[8])} "
              f"red_indptr={_d(a[9])} red_final={_d(a[10])} red_partial={_d(a[11])} "
              f"topk={k.get('topk')} max_split={k.get('max_split_per_batch')} "
              f"maxq={k.get('max_seqlen_qo')} intra={k.get('intra_batch_mode')}",
              flush=True)
        return _get_meta(*a, **k)

    def mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
                       kv_last_page_lens, max_seqlen_q, *a, **k):
        _n["dec"] += 1
        i = _n["dec"]
        rp = k.get("reduce_partial_map")
        print(f"{TAG} dec#{i} q={_d(q)} kv_buf={_d(kv_buffer)} o={_d(o)} "
              f"qo={_d(qo_indptr)} bs_from_qo={qo_indptr.shape[0]-1} "
              f"kv_indptr={_d(kv_indptr)} kv_indices={_d(kv_indices)} "
              f"klpl={_d(kv_last_page_lens)} maxq={max_seqlen_q} "
              f"red_partial={_d(rp)} nsplit={k.get('num_kv_splits')} "
              f"intra={k.get('intra_batch_mode')}", flush=True)
        return _dec(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
                    kv_last_page_lens, max_seqlen_q, *a, **k)

    A.get_mla_metadata_v1 = get_mla_metadata_v1
    M.mla_decode_fwd = mla_decode_fwd
    # sglang imported them by value; rebind there too
    try:
        import sglang.srt.layers.attention.dsa_backend as DB
        DB.mla_decode_fwd = mla_decode_fwd
        DB.get_mla_metadata_v1 = get_mla_metadata_v1
        print(f"{TAG} rebound in dsa_backend", flush=True)
    except Exception as e:
        print(f"{TAG} rebind dsa_backend failed: {e}", flush=True)
    print(f"{TAG} installed", flush=True)


install()

from sglang.launch_server import __name__ as _  # noqa
import runpy
sys.argv = ["sglang.launch_server"] + sys.argv[1:]
runpy.run_module("sglang.launch_server", run_name="__main__")
