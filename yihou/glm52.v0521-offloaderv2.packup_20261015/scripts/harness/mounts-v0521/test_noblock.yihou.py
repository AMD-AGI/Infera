# Repro: emit_kv_metrics against a PUSH socket whose endpoint nobody binds.
import importlib.util, sys, types, threading, tempfile, zmq
path = sys.argv[1]
def _mod(name, **kw):
    mod = types.ModuleType(name); mod.__dict__.update(kw); sys.modules[name] = mod
for n in ("sglang", "sglang.srt", "sglang.srt.disaggregation", "sglang.srt.managers"): _mod(n)
_mod("sglang.srt.disaggregation.kv_events", EventPublisherFactory=None, KVEventBatch=object,
     is_kv_publisher_rank=lambda *a: False, select_kv_publisher_dp_rank=lambda *a: 0)
# same body as io_struct.py:2615 (pickle path)
_mod("sglang.srt.managers.io_struct", hook_custom_types=lambda *a: None,
     sock_send=lambda sock, obj, flags=0: sock.send_pyobj(obj, flags=flags))
_mod("sglang.srt.runtime_context", get_parallel=None)
spec = importlib.util.spec_from_file_location("kvpub_under_test", path)
m = importlib.util.module_from_spec(spec); sys.modules["kvpub_under_test"] = m; spec.loader.exec_module(m)
m.get_parallel = lambda: types.SimpleNamespace(dp_rank=0)
ctx = zmq.Context(); s = ctx.socket(zmq.PUSH); s.setsockopt(zmq.SNDHWM, 4)
s.connect(f"ipc://{tempfile.mktemp(prefix='yihou-nobody-')}")
st = types.SimpleNamespace(num_running_reqs=types.SimpleNamespace(total=0), token_usage=0.0,
                           num_queue_reqs=types.SimpleNamespace(total=0), cache_hit_rate=0.0)
p = m.SchedulerKvEventsPublisher(kv_events_config=None, tree_cache=None, send_metrics_from_scheduler=s,
                                 max_running_requests=64, max_total_num_tokens=1000, get_stats=lambda: st)
p.enable_kv_cache_events = True
done = threading.Event()
def run():
    for _ in range(100): p.emit_kv_metrics()
    done.set()
threading.Thread(target=run, daemon=True).start()
print("RESULT", "returned" if done.wait(10) else "BLOCKED(>10s)", flush=True)
import os; os._exit(0)
