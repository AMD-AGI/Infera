"""Exact-anchor instrumentation for pinned AUS SGLang; fails on source drift."""
import ast
import os
from pathlib import Path

root = Path(os.environ.get("SGLANG_PATCH_ROOT", "/sgl-workspace/sglang/python/sglang/srt"))

def edit(path, changes):
    target = root / path
    text = target.read_text()
    prefix = suffix = ""
    if path == "disaggregation/decode.py":
        start = text.index("    def pop_preallocated(")
        end = text.index("    @property", start)
        prefix, text, suffix = text[:start], text[start:end], text[end:]
    for old, new in changes:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"{path}: expected one anchor, got {count}: {old[:90]}")
        text = text.replace(old, new, 1)
    result = prefix + text + suffix
    ast.parse(result)
    target.write_text(result)

edit("managers/schedule_batch.py", [
    ('        self.has_log_time_stats = True\n',
     '        from sglang.srt.observability.aus_diag import summary\n'
     '        summary(self)\n        self.has_log_time_stats = True\n'),
    ('                    req._cache_breakdown_computed = True\n',
     '                    req._cache_breakdown_computed = True\n'
     '                    from sglang.srt.observability.aus_diag import emit\n'
     '                    emit(req, "cache", input_tokens=len(req.origin_input_ids),\n'
     '                         cached_device=req.cached_tokens_device, cached_host=req.cached_tokens_host,\n'
     '                         cached_storage=req.cached_tokens_storage, prefill_dp_rank=req.disagg_prefill_dp_rank,\n'
     '                         host_node_id=getattr(req.last_node, "id", None))\n')])

changes = []
for condition, reason in [
    ('self.req_to_token_pool.available_size() <= 0', 'request_pool'),
    ('self.req_to_metadata_buffer_idx_allocator.available_size() <= 0', 'metadata_pool'),
    ('hisparse_req_budget <= 0', 'hisparse_pool')]:
    old = f'            if {condition}:\n                break\n'
    new = f'            if {condition}:\n                from sglang.srt.observability.aus_diag import admission\n                admission(self, decode_req.req, "{reason}")\n                break\n'
    changes.append((old, new))
old = '                > full_allocatable_tokens\n            ):\n'
changes.append((old, old + '                from sglang.srt.observability.aus_diag import admission\n                admission(self, decode_req.req, "kv_budget", required_tokens_for_request, full_allocatable_tokens)\n'))
old = '            dst_kv_indices = self._pre_alloc(\n'
changes.append((old, '            from sglang.srt.observability.aus_diag import admission\n            admission(self, decode_req.req, "admitted", required_tokens_for_request, full_allocatable_tokens)\n' + old))
edit("disaggregation/decode.py", changes)

edit("mem_cache/hiradix_cache.py", [
    ('        self.ongoing_load_back[last_hit_node.id] = last_hit_node\n',
     '        from sglang.srt.observability.aus_diag import emit\n'
     '        emit(None, "host_load_submitted", node_id=last_hit_node.id, tokens=len(host_indices))\n'
     '        self.ongoing_load_back[last_hit_node.id] = last_hit_node\n'),
    ('                end_node = self.ongoing_load_back.pop(ack_id)\n',
     '                from sglang.srt.observability.aus_diag import emit\n'
     '                emit(None, "host_load_ack", node_id=ack_id, batch_bytes=ack.num_bytes)\n'
     '                end_node = self.ongoing_load_back.pop(ack_id)\n')])

# Emit durable entry/state events even for requests that never complete. Existing
# OTLP spans retain detailed chunk timing; do not emit a line on every decode step.
path = root / "observability/req_time_stats.py"
text = path.read_text()
for field in ("scheduler_recv_time", "prefill_bootstrap_queue_entry_time",
              "prefill_transfer_queue_entry_time", "prefill_kv_transfer_finish_time",
              "decode_prealloc_queue_entry_time", "decode_transfer_queue_entry_time"):
    old = f'        self.{field} = ts\n'
    assert text.count(old) == 1, field
    text = text.replace(old, old + f'        from sglang.srt.observability.aus_diag import emit\n        emit(self, "{field}")\n')
ast.parse(text)
path.write_text(text)
