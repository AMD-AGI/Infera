"""Bounded request/admission diagnostics; no tensor reads or GPU synchronization."""
import json
import os
import threading
import time

_streams = {}
_lock = threading.Lock()

def emit(obj, event, **fields):
    directory = os.environ.get("AUS_DIAG_DIR")
    if not directory:
        return
    from sglang.srt.observability.trace import threads_info
    info = threads_info.get(threading.get_native_id())
    ts = getattr(obj, "time_stats", obj)
    ctx = getattr(ts, "trace_ctx", None)
    rid = getattr(obj, "rid", None) or getattr(ctx, "rid", None)
    room = getattr(obj, "bootstrap_room", None) or getattr(ctx, "bootstrap_room", None)
    row = dict(event=event, wall_ns=time.time_ns(), mono_ns=time.perf_counter_ns(),
               role=os.environ.get("AUS_DIAG_ROLE"), pid=os.getpid(),
               dp_rank=getattr(info, "dp_rank", None),
               tp_rank=getattr(info, "tp_rank", None),
               rid=rid if isinstance(rid, str) else None,
               room=str(room) if isinstance(room, (str, int)) else None, **fields)
    # Per-process local files avoid shared-filesystem writes in the scheduler.
    with _lock:
        pid = os.getpid()
        if pid not in _streams:
            os.makedirs(directory, exist_ok=True)
            _streams[pid] = open(f"{directory}/{pid}.jsonl", "a", buffering=1)
        _streams[pid].write(json.dumps(row, separators=(",", ":")) + "\n")

def admission(queue, req, reason, required=None, budget=None):
    now = time.monotonic()
    previous = getattr(req, "_aus_block", None)
    if previous and previous[0] == reason and now - previous[1] < 5:
        return
    req._aus_block = (reason, now)
    emit(req, "admission", reason=reason, required=required, budget=budget,
         input_tokens=len(req.origin_input_ids),
         free_requests=queue.req_to_token_pool.available_size(),
         free_metadata=queue.req_to_metadata_buffer_idx_allocator.available_size(),
         running=len(queue.scheduler.running_batch.reqs),
         waiting=len(queue.scheduler.waiting_queue),
         transfer=len(queue.transfer_queue.queue), prealloc=len(queue.queue))

def summary(req):
    ts = req.time_stats
    names = ("scheduler_recv_time", "wait_queue_entry_time", "forward_entry_time",
             "prefill_finished_time", "completion_time", "bootstrap_done_time",
             "prefill_bootstrap_queue_entry_time", "prefill_transfer_queue_entry_time",
             "prefill_kv_transfer_finish_time", "decode_prealloc_queue_entry_time",
             "decode_transfer_queue_entry_time")
    emit(req, "request_summary", input_tokens=len(req.origin_input_ids),
         output_tokens=len(req.output_ids), cached_device=req.cached_tokens_device,
         cached_host=req.cached_tokens_host, cached_storage=req.cached_tokens_storage,
         prefill_dp_rank=req.disagg_prefill_dp_rank,
         attempts=req.prefill_attempt_count,
         times={name: getattr(ts, name, 0.0) for name in names})
