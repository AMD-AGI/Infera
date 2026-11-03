"""Unit checks of the fix, run inside the image with patches/fix mounted over the six files:
  docker run --rm ... -v patches/fix/<f>:/sgl-workspace/sglang/python/sglang/srt/<f>:ro ... python3 test_fix.py MODEL_DIR
"""
import dataclasses
import inspect
import sys
import types

import torch
from transformers import AutoConfig

from sglang.srt.layers.attention import graph_variants as gv
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


class DeviceRead:
    """Stands in for device tensors: any use fails the test."""

    def __getattr__(self, _):
        raise AssertionError("select() touched seq_lens")


hf = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)

# part 1: no dense/sparse pair for target-verify captures; decode and the old call keep it
check("verify capture: no variants", gv.create_attention_graph_variants(hf, ForwardMode.TARGET_VERIFY) is None)
v = gv.create_attention_graph_variants(hf, ForwardMode.DECODE)
check("decode capture: DsaGraphVariants", isinstance(v, gv.DsaGraphVariants))
check("old signature: DsaGraphVariants", isinstance(gv.create_attention_graph_variants(hf), gv.DsaGraphVariants))
topk = v.index_topk

# part 2: select() follows the agreed bound and reads no device tensor
fb = types.SimpleNamespace(dp_decode_graph_max_seq_len=topk, seq_lens_cpu=DeviceRead(), seq_lens=DeviceRead())
check("bound == topk -> dense, no device read", v.select(fb) == gv.DSA_DENSE)
fb.dp_decode_graph_max_seq_len = topk + 1
check("bound > topk -> sparse, no device read", v.select(fb) == gv.DSA_SPARSE)
fb.dp_decode_graph_max_seq_len = 0
check("all ranks idle (bound 0) -> dense on every rank", v.select(fb) == gv.DSA_DENSE)
old = types.SimpleNamespace(dp_decode_graph_max_seq_len=None, seq_lens_cpu=torch.tensor([5, topk + 7]), seq_lens=None)
check("no bound -> image behaviour (host mirror)", v.select(old) == gv.DSA_SPARSE)
old.seq_lens_cpu = torch.tensor([5, 9])
check("no bound -> image behaviour (dense)", v.select(old) == gv.DSA_DENSE)

# the sync row carries the column; local and gathered values
kw = dict(num_dp_ranks=1, tp_size=1, cp_size=1, num_tokens=3, num_tokens_for_logprob=3, can_run_decode_cuda_graph=True,
          can_run_draft_cuda_graph=True, can_run_prefill_cuda_graph=False, is_extend_in_batch=False,
          local_can_run_tbo=False, local_forward_mode=ForwardMode.DECODE.value)
info = dp_attn.MLPSyncBatchInfo(**kw, decode_graph_max_seq_len=1234)
loc, fbk = info._get_local_tensor("cpu"), info._get_fallback_tensor("cpu")
check("row width 12, last column = bound", loc.numel() == 12 and int(loc[-1]) == 1234)
check("fallback row width 12, bound 0", fbk.numel() == 12 and int(fbk[-1]) == 0)
info.finalize_local()
check("single rank: global bound = local", info.global_decode_graph_max_seq_len == 1234)
rows = torch.stack([loc, fbk, loc.clone()])
rows[2, 11] = 4321
check("gathered parse: max over ranks", int(rows[:, 11].max()) == 4321)
src = inspect.getsource(dp_attn.MLPSyncBatchInfo.all_gather)
check("all_gather reads column 11 with max", "tp0_info_cpu[:, 11].max()" in src)

# local bound
dp_attn.max_speculative_num_draft_tokens = lambda: 4
dec = ForwardMode.DECODE
mk = lambda **k: types.SimpleNamespace(forward_mode=dec, batch_size=lambda: 2, **k)
check("host mirror -> exact max", dp_attn._local_decode_graph_max_seq_len(mk(seq_lens_cpu=torch.tensor([10, 2000]), reqs=[])) == 2000)
reqs = [types.SimpleNamespace(seqlen=100), types.SimpleNamespace(seqlen=2040)]
check("spec, no mirror -> committed + 2 x draft", dp_attn._local_decode_graph_max_seq_len(mk(seq_lens_cpu=None, reqs=reqs)) == 2048)
dp_attn.max_speculative_num_draft_tokens = lambda: None
check("no spec, no mirror -> committed + 2", dp_attn._local_decode_graph_max_seq_len(mk(seq_lens_cpu=None, reqs=reqs)) == 2042)
idle = types.SimpleNamespace(forward_mode=ForwardMode.IDLE, batch_size=lambda: 0, seq_lens_cpu=None, reqs=[])
check("idle -> 0", dp_attn._local_decode_graph_max_seq_len(idle) == 0)
check("no batch -> 0", dp_attn._local_decode_graph_max_seq_len(None) == 0)
ext = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND, batch_size=lambda: 2, seq_lens_cpu=torch.tensor([9]), reqs=[])
check("extend -> 0", dp_attn._local_decode_graph_max_seq_len(ext) == 0)

# _update_gather_batch puts the agreed bound on the batch, None when the ranks did not sync
b = types.SimpleNamespace()
dp_attn._update_gather_batch(b, info, require_mlp_tp_gather=True)
check("update_gather_batch: bound set", b.dp_decode_graph_max_seq_len == 1234)
dp_attn._update_gather_batch(b, info, require_mlp_tp_gather=True, skip_global_metadata=True)
check("update_gather_batch: no sync -> None", b.dp_decode_graph_max_seq_len is None)

# the field rides ScheduleBatch.copy, ForwardBatch.init_new and the TBO split
check("ScheduleBatch field", "dp_decode_graph_max_seq_len" in {f.name for f in dataclasses.fields(ScheduleBatch)})
check("ScheduleBatch.copy carries it", "dp_decode_graph_max_seq_len=self.dp_decode_graph_max_seq_len" in inspect.getsource(ScheduleBatch.copy))
check("ForwardBatch field", "dp_decode_graph_max_seq_len" in {f.name for f in dataclasses.fields(ForwardBatch)})
check("ForwardBatch.init_new carries it", "dp_decode_graph_max_seq_len=getattr(batch" in inspect.getsource(ForwardBatch.init_new))
from sglang.srt.batch_overlap import two_batch_overlap

check("TBO split copies it", '"dp_decode_graph_max_seq_len"' in inspect.getsource(two_batch_overlap))

failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} passed" + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
