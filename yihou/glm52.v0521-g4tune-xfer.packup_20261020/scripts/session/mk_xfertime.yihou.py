# Build conn.yihou-xfertime.py from conn.yihou-regbase.py (task g4tune 2026-10-19). Diagnostic-only timing, no logic change.
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def once(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)

HELPER = '''logger = logging.getLogger(__name__)

# [yihou] YIHOU-XFERTIME diagnostic (mount patch, not upstream): per-chunk timing of the
# prefill KV sender. Enabled only when YIHOU_XFER_LOG_DIR is set; writes buffered TSV lines to
# $YIHOU_XFER_LOG_DIR/xfer.<host>.<pid>.tsv. Never raises into the transfer path.
_YX_DIR = os.environ.get("YIHOU_XFER_LOG_DIR", "")
_yx_lock = threading.Lock()
_yx_buf = []
_yx_state = {"fh": None, "last": 0.0}
_yx_tl = threading.local()


def _yx_emit(fields):
    if not _YX_DIR:
        return
    try:
        line = "\\t".join(str(f) for f in fields) + "\\n"
        with _yx_lock:
            _yx_buf.append(line)
            now = time.time()
            if len(_yx_buf) >= 64 or now - _yx_state["last"] > 2.0:
                if _yx_state["fh"] is None:
                    import socket

                    os.makedirs(_YX_DIR, exist_ok=True)
                    _yx_state["fh"] = open(
                        os.path.join(
                            _YX_DIR, f"xfer.{socket.gethostname()}.{os.getpid()}.tsv"
                        ),
                        "a",
                    )
                _yx_state["fh"].write("".join(_yx_buf))
                _yx_state["fh"].flush()
                _yx_buf.clear()
                _yx_state["last"] = now
    except Exception as e:  # diagnostics must never break the transfer
        logger.warning("YIHOU-XFERTIME emit failed: %r", e)
'''
once("logger = logging.getLogger(__name__)\n", HELPER)

once('''        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
''', '''        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        _yx_t0 = time.time()
        _yx_ret = self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
        _yx_acc = getattr(_yx_tl, "acc", None)
        if _yx_acc is not None:
            _yx_acc[0] += 1
            _yx_acc[1] += len(lengths)
            _yx_acc[2] += sum(lengths)
            _yx_acc[3] += time.time() - _yx_t0
        return _yx_ret
''')

once('''        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=trace_ctx,
                wait_event=wait_event,
            )
        )
''', '''        _yx_chunk = TransferKVChunk(
            room=bootstrap_room,
            prefill_kv_indices=kv_indices,
            index_slice=index_slice,
            is_last_chunk=is_last_chunk,
            prefill_aux_index=aux_index,
            state_indices=state_indices,
            num_kv_tokens=num_kv_tokens,
            trace_ctx=trace_ctx,
            wait_event=wait_event,
        )
        _yx_chunk._yx_enq = time.time()
        _yx_chunk._yx_shard = shard_idx
        _yx_chunk._yx_ports = "+".join(s.rsplit(":", 1)[1] for s in dst_infos)
        _yx_chunk._yx_haswait = int(wait_event is not None)
        self.transfer_queues[shard_idx].put(_yx_chunk)
''')

once('''                kv_chunk = queue.get()
                # teardown() pushes a None sentinel to unblock get() and stop
                # the worker: FastQueue.get() blocks indefinitely, so checking
                # _stopped alone can never wake a parked worker during a role switch.
                if kv_chunk is None:
                    break
''', '''                kv_chunk = queue.get()
                # teardown() pushes a None sentinel to unblock get() and stop
                # the worker: FastQueue.get() blocks indefinitely, so checking
                # _stopped alone can never wake a parked worker during a role switch.
                if kv_chunk is None:
                    break
                _yx_deq = time.time()
                _yx_tl.acc = [0, 0, 0, 0.0]
                _yx_seg = {}
''')

once('''                        kv_chunk.wait_event.synchronize()
                    kv_chunk.wait_event = None
''', '''                        kv_chunk.wait_event.synchronize()
                    kv_chunk.wait_event = None
                _yx_wdone = time.time()
''')

once('''                for req in reqs_to_be_processed:
                    start_ts = time.perf_counter()
''', '''                for req in reqs_to_be_processed:
                    start_ts = time.perf_counter()
                    _yx_seg.setdefault("kv0", time.time())
''')

once('''                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
''', '''                        _yx_seg["kv1"] = time.time()
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
''')

once('''                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
''', '''                            # Only the last chunk we need to send the aux data
                            _yx_seg["aux0"] = time.time()
                            ret = self.send_aux(
''')

once('''                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append((req.endpoint, req.dst_port))
''', '''                            _yx_seg["aux1"] = time.time()
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append((req.endpoint, req.dst_port))
''')

once('''                if staging_deferred:
                    continue

                self._staging_outstanding[kv_chunk.room] -= 1
''', '''                try:
                    _yx_end = time.time()
                    _yx_a = getattr(_yx_tl, "acc", [0, 0, 0, 0.0])
                    _yx_enq = getattr(kv_chunk, "_yx_enq", 0.0)
                    _yx_emit(
                        [
                            "chunk", "%.6f" % _yx_end, self.attn_dp_rank, worker_index,
                            kv_chunk.room, int(bool(kv_chunk.is_last_chunk)),
                            getattr(kv_chunk, "_yx_shard", -1),
                            getattr(kv_chunk, "_yx_ports", ""),
                            getattr(kv_chunk, "_yx_haswait", -1),
                            len(kv_chunk.prefill_kv_indices), kv_chunk.num_kv_tokens,
                            len(reqs_to_be_processed),
                            "%.6f" % _yx_enq, "%.6f" % (_yx_deq - _yx_enq),
                            "%.6f" % (_yx_wdone - _yx_deq),
                            "%.6f" % (_yx_seg.get("kv1", _yx_wdone) - _yx_seg.get("kv0", _yx_wdone)),
                            "%.6f" % (_yx_seg.get("aux0", 0.0) - _yx_seg.get("kv1", 0.0) if "aux0" in _yx_seg else 0.0),
                            "%.6f" % (_yx_seg.get("aux1", 0.0) - _yx_seg.get("aux0", 0.0) if "aux1" in _yx_seg else 0.0),
                            "%.6f" % (_yx_end - _yx_deq),
                            _yx_a[0], _yx_a[1], _yx_a[2], "%.6f" % _yx_a[3],
                            int(staging_deferred),
                        ]
                    )
                    _yx_tl.acc = None
                except Exception as _yx_e:
                    logger.warning("YIHOU-XFERTIME record failed: %r", _yx_e)

                if staging_deferred:
                    continue

                self._staging_outstanding[kv_chunk.room] -= 1
''')
open(dst, "w").write(s)
print("ok")
