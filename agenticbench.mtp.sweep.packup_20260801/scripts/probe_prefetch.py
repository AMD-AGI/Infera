#!/usr/bin/env python3
"""Instrument every exit path of SGLang's L3 prefetch, in ONE round.

WHY. The kvd restart-replay came back gets=0, hits=0, misses=0, sets +15,781 --
the engine re-STORED the replayed prompts instead of reading them. misses=0 with
gets=0 means kvd was never even ASKED, so this is not a key mismatch downstream;
the query never left the engine. Something upstream returns before
`storage_backend.batch_exists`.

`prefetch_from_storage` (hiradix_cache.py:1585) has FOUR independent early
returns, and reading the source cannot tell which one fires:

    1. not self.enable_storage                    -> storage never wired
    2. prefetch_length < self.prefetch_threshold  -> prompt too short after
                                                     page alignment
    3. cache_controller.prefetch_rate_limited()   -> prefetch_tokens_occupied
                                                     >= prefetch_capacity_limit
    4. host_indices is None (twice)               -> host pool exhausted, even
                                                     after evict_host

Infera's own guard (`hicache_validate.py`) documents a FIFTH story -- that
`prefetch_capacity_limit = 0.8*(host-device)` collapses to 0 -- but in THIS
sglang build line 467 reads `int(0.5 * self.mem_pool_host.size)`, which is
~356K tokens, not 0. So that comment does not describe this version, and the
guard also bails whenever --hicache-size is set (which it is). Do not trust
either; measure.

This wraps the real method, logs which branch each call takes together with the
values that decided it, and forwards to the original. It changes no behaviour.

Run INSIDE the engine container BEFORE the engine starts... except we cannot
restart cheaply, so instead this is applied as a sitecustomize-style patch file
that the next engine boot picks up. Usage:

    probe_prefetch.py install     # write the patch into sglang
    probe_prefetch.py uninstall   # remove it
"""
import importlib.util
import os
import sys

MARKER = "INFERA_PREFETCH_PROBE"

PATCH = '''
# ---- {marker}: instrument every early return of prefetch_from_storage ----
# Added by work.agenticbench.mtp/scripts/probe_prefetch.py. Removes cleanly.
# Logs which of the four exits fires and the numbers that decided it, then
# forwards to the original. No behaviour change.
{marker} = "installed"


def _infera_install_prefetch_probe():
    import logging
    _log = logging.getLogger("infera.prefetch_probe")
    _cls = HiRadixCache
    if getattr(_cls, "_infera_probed", False):
        return
    _orig = _cls.prefetch_from_storage
    _state = {{"n": 0}}

    def probed(self, req_id, last_host_node, new_input_tokens, last_hash=None,
               prefix_keys=None):
        _state["n"] += 1
        n = _state["n"]
        if n > 60:                      # bound the log; the pattern is set by then
            return _orig(self, req_id, last_host_node, new_input_tokens,
                         last_hash, prefix_keys)
        try:
            key = RadixKey(new_input_tokens,
                           extra_key=last_host_node.key.extra_key,
                           is_bigram=self.is_eagle).page_aligned(self.page_size)
            plen = len(key)
            cc = self.cache_controller
            limited = cc.prefetch_rate_limited()
            avail = cc.mem_pool_host.available_size()
            _log.warning(
                "{marker} #%d enable_storage=%s prefetch_len=%d threshold=%d "
                "rate_limited=%s occupied=%s capacity=%s host_avail=%d "
                "host_size=%s last_hash=%s ntok=%d",
                n, self.enable_storage, plen, self.prefetch_threshold, limited,
                getattr(cc, "prefetch_tokens_occupied", "?"),
                getattr(cc, "prefetch_capacity_limit", "?"),
                avail, getattr(cc.mem_pool_host, "size", "?"),
                str(last_hash)[:16], len(new_input_tokens),
            )
            if not self.enable_storage:
                _log.warning("{marker} #%d EXIT=enable_storage_false", n)
            elif plen < self.prefetch_threshold:
                _log.warning("{marker} #%d EXIT=below_threshold %d<%d", n, plen,
                             self.prefetch_threshold)
            elif limited:
                _log.warning("{marker} #%d EXIT=rate_limited", n)
            elif avail < plen:
                _log.warning("{marker} #%d WARN=host_pool_tight avail=%d need=%d",
                             n, avail, plen)
            else:
                _log.warning("{marker} #%d PROCEEDS to storage query", n)
        except Exception as e:          # never break the engine for a probe
            _log.warning("{marker} #%d probe error %s: %s", n, type(e).__name__, e)
        return _orig(self, req_id, last_host_node, new_input_tokens, last_hash,
                     prefix_keys)

    _cls.prefetch_from_storage = probed
    _cls._infera_probed = True


_infera_install_prefetch_probe()
# ---- end {marker} ----
'''.format(marker=MARKER)


def target():
    spec = importlib.util.find_spec("sglang")
    root = list(spec.submodule_search_locations)[0]
    p = os.path.join(root, "srt", "mem_cache", "hiradix_cache.py")
    if not os.path.isfile(p):
        sys.exit(f"not found: {p}")
    return p


def drop_pyc(path):
    d = os.path.join(os.path.dirname(path), "__pycache__")
    base = os.path.basename(path)[:-3]
    n = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith(base + "."):
                os.remove(os.path.join(d, f))
                n += 1
    return n


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "install"
    p = target()
    src = open(p).read()
    if act == "install":
        if MARKER in src:
            print(f"[probe] already installed: {p}")
            return 0
        open(p, "w").write(src + "\n" + PATCH)
        print(f"[probe] installed into {p}; dropped {drop_pyc(p)} stale .pyc")
    elif act == "uninstall":
        if MARKER not in src:
            print("[probe] not installed")
            return 0
        head = src.split(f"\n# ---- {MARKER}")[0]
        open(p, "w").write(head)
        print(f"[probe] removed from {p}; dropped {drop_pyc(p)} stale .pyc")
    else:
        sys.exit("usage: probe_prefetch.py install|uninstall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
