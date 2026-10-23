

# ---- yihou dcpx8/dcpx9 (2026-10-22): log-only transfer statistics, enabled by YIHOU_XFER_STATS=1 -----------------------------
# Every 30 s per process: batch_transfer_sync calls, blocks, bytes, wall time inside it, block-size histogram, per
# destination session; send_kvcache_dcp calls and wall time; DCP pack (gather + sync) calls and wall time.
if os.environ.get("YIHOU_XFER_STATS") == "1":
    import functools as _yh_ft
    from collections import defaultdict as _yh_dd

    _YH_EDGES = np.array([4096, 16384, 65536, 262144, 1048576], dtype=np.int64)

    class _YihouXferStats:
        def __init__(self):
            self.lock = threading.Lock()
            self.t_last = time.monotonic()
            self._reset()

        def _reset(self):
            self.calls = 0
            self.blocks = 0
            self.nbytes = 0
            self.t_xfer = 0.0
            self.t_xfer_max = 0.0
            self.hist = np.zeros(len(_YH_EDGES) + 1, dtype=np.int64)
            self.sess = _yh_dd(lambda: [0, 0, 0.0])
            self.dcp_calls = 0
            self.t_dcp = 0.0
            self.pack_calls = 0
            self.t_pack = 0.0
            self.busy = _yh_dd(float)

        def add_xfer(self, sess, lengths, dt):
            arr = np.asarray(lengths, dtype=np.int64)
            h = np.bincount(np.searchsorted(_YH_EDGES, arr, side="right"), minlength=len(_YH_EDGES) + 1)
            with self.lock:
                self.calls += 1
                self.blocks += int(arr.size)
                nb = int(arr.sum())
                self.nbytes += nb
                self.t_xfer += dt
                self.t_xfer_max = max(self.t_xfer_max, dt)
                self.hist += h
                s = self.sess[sess]
                s[0] += 1
                s[1] += nb
                s[2] += dt
                self.busy[threading.current_thread().name] += dt
            self.maybe_log()

        def add(self, kind, dt):
            with self.lock:
                if kind == "dcp":
                    self.dcp_calls += 1
                    self.t_dcp += dt
                else:
                    self.pack_calls += 1
                    self.t_pack += dt

        def maybe_log(self):
            now = time.monotonic()
            if now - self.t_last < 30.0:
                return
            with self.lock:
                if now - self.t_last < 30.0:
                    return
                win = now - self.t_last
                self.t_last = now
                calls, blocks, nbytes, t_xfer = self.calls, self.blocks, self.nbytes, self.t_xfer
                msg = (
                    f"[yihou-xfer] win={win:.1f}s calls={calls} blocks={blocks} "
                    f"bytes={nbytes / 1e9:.3f}GB rate={nbytes / 1e9 / win:.2f}GB/s "
                    f"t_xfer={t_xfer:.2f}s max={self.t_xfer_max:.3f}s "
                    f"per_call_GBps={(nbytes / 1e9 / t_xfer) if t_xfer else 0:.2f} "
                    f"avg_block={(nbytes / blocks) if blocks else 0:.0f}B "
                    f"hist(<4K,<16K,<64K,<256K,<1M,>=1M)={self.hist.tolist()} "
                    f"dcp_calls={self.dcp_calls} t_dcp={self.t_dcp:.2f}s "
                    f"pack_calls={self.pack_calls} t_pack={self.t_pack:.2f}s "
                    f"pack_sub(index,launch,sync,n)={[round(x, 2) for x in _yh_dcp_pack_t()]} "
                    f"busy={{{', '.join(f'{k}:{v:.1f}' for k, v in sorted(self.busy.items()))}}} "
                    f"sess={{{', '.join(f'{k}:{v[0]}/{v[1] / 1e9:.2f}GB/{v[2]:.1f}s' for k, v in sorted(self.sess.items()))}}}"
                )
                self._reset()
                _yh_dcp_pack_t(reset=True)
            logger.info(msg)

    def _yh_dcp_pack_t(reset=False):
        # dcpx9: sub-timers of dcp_pack.try_pack_dcp_src (present only with the dcp_pack.yihou-nowait.py mount).
        import sglang.srt.disaggregation.common.dcp_pack as _m
        t = getattr(_m, "YH_T", None)
        if t is None:
            return []
        out = list(t)
        if reset:
            t[0] = t[1] = t[2] = 0.0
            t[3] = 0
        return out

    _YH = _YihouXferStats()

    _yh_orig_transfer_data = MooncakeKVManager._transfer_data

    @_yh_ft.wraps(_yh_orig_transfer_data)
    def _yh_transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return _yh_orig_transfer_data(self, mooncake_session_id, transfer_blocks)
        t0 = time.perf_counter()
        ret = _yh_orig_transfer_data(self, mooncake_session_id, transfer_blocks)
        _YH.add_xfer(mooncake_session_id, [b[2] for b in transfer_blocks], time.perf_counter() - t0)
        return ret

    MooncakeKVManager._transfer_data = _yh_transfer_data

    _yh_orig_send_dcp = MooncakeKVManager.send_kvcache_dcp

    @_yh_ft.wraps(_yh_orig_send_dcp)
    def _yh_send_dcp(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return _yh_orig_send_dcp(self, *a, **k)
        finally:
            _YH.add("dcp", time.perf_counter() - t0)

    MooncakeKVManager.send_kvcache_dcp = _yh_send_dcp

    import sglang.srt.disaggregation.common.dcp_pack as _yh_dcp_pack

    _yh_orig_pack = _yh_dcp_pack.try_pack_dcp_src

    @_yh_ft.wraps(_yh_orig_pack)
    def _yh_pack(*a, **k):
        t0 = time.perf_counter()
        try:
            return _yh_orig_pack(*a, **k)
        finally:
            _YH.add("pack", time.perf_counter() - t0)

    _yh_dcp_pack.try_pack_dcp_src = _yh_pack
