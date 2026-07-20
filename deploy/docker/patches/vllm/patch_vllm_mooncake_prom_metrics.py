###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Give vLLM's MooncakeConnector the ``build_prom_metrics`` it is missing.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py`
loop). Idempotent and self-locating, so a vLLM version bump degrades gracefully.

Why: MooncakeConnector implements ``get_kv_connector_stats`` (it emits per-step
transfer stats) but NOT ``build_prom_metrics``, so it inherits the base
classmethod that returns None. ``MultiConnector.build_prom_metrics`` only
registers children whose ``build_prom_metrics`` returns non-None, so Mooncake
never lands in the Prometheus registry -- and then
``MultiKVConnectorPromMetrics.observe`` asserts on the first stats collection
after the first request:

    AssertionError: MooncakeConnector is not contained in the list of
    registered connectors with Prometheus metrics support: dict_keys([...])

and kills the engine. This is only reachable under MultiConnector -- a
single-connector setup never takes that path -- i.e. exactly the PD + kvd-L3
recipe: MultiConnector(InferaKvdConnector + MooncakeConnector) with engine
metrics enabled (no --disable-log-stats). Infera PR #178 fixed the InferaKvd
half of this same bug and stopped there, so KVD=1 with stats still died -- which
is why the Kimi-K2.6 PD benchmarks had to run with L3 off or stats off, and why
kvd's own L3 telemetry was invisible.

Patching at image-build time (rather than monkey-patching from Infera's Python
at runtime) is deliberate: the runtime approach loses an import-order race --
MultiConnector calls Mooncake's build_prom_metrics before the Infera module that
would patch it is imported. Editing the vendored file removes the race entirely.

The added ``build_prom_metrics`` returns a no-op observe() adapter, mirroring
InferaKvdPromMetrics: Mooncake's transfer telemetry already flows through
``get_kv_connector_stats``; this only satisfies the registration contract. It
invents no counters and alters no existing metric -- the ``vllm:*prefix_cache_*``
counters we want come from the engine's own accounting, which was merely gated
behind engine metrics being enabled at all.

The right long-term home is upstream vLLM (Mooncake should implement this); this
patch no-ops the moment upstream adds the method.

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_vllm_mooncake_prom_metrics.py
"""

import ast
import os
import sys

try:
    import vllm
except Exception:
    print("mooncake-prom: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
)
if not os.path.exists(f):
    print(f"mooncake-prom: {f} not found (no mooncake connector in this image) — skipping")
    sys.exit(0)

src = open(f).read()

MARKER = "def build_prom_metrics"
if MARKER in src:
    print("mooncake-prom: MooncakeConnector already defines build_prom_metrics — skipping")
    sys.exit(0)

# Anchor: append the method right after Mooncake's build_kv_connector_stats
# classmethod, which sits inside the MooncakeConnector class body. If the anchor
# is gone the connector layout has drifted from what this patch targets -> fail
# loudly rather than silently produce a broken engine.
ANCHOR = (
    "    @classmethod\n"
    "    def build_kv_connector_stats(\n"
    "        cls, data: dict[str, Any] | None = None\n"
    "    ) -> KVConnectorStats | None:\n"
    "        return MooncakeKVConnectorStats(data=data or {})\n"
)

if ANCHOR not in src:
    print(
        "mooncake-prom: ERROR anchor (build_kv_connector_stats) not found — "
        "MooncakeConnector layout changed; refusing to patch blindly",
        file=sys.stderr,
    )
    sys.exit(1)

ADDITION = """
    # --- Infera patch: register with vLLM's per-connector Prometheus metrics ---
    # MooncakeConnector emits stats via get_kv_connector_stats() above but did not
    # implement build_prom_metrics(), so under MultiConnector with metrics on the
    # engine asserts and dies on the first observe(). observe() is a deliberate
    # no-op (Mooncake stats already flow through get_kv_connector_stats); this only
    # satisfies the registration contract. See patch_vllm_mooncake_prom_metrics.py.
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorPromMetrics,
        )

        class _MooncakePromMetrics(KVConnectorPromMetrics):
            def observe(self, transfer_stats_data, engine_idx: int = 0) -> None:
                return

        return _MooncakePromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
"""

out = src.replace(ANCHOR, ANCHOR + ADDITION, 1)

# sanity: it must parse
try:
    ast.parse(out)
except SyntaxError as exc:
    print(f"mooncake-prom: ERROR patched file does not parse: {exc}", file=sys.stderr)
    sys.exit(1)

open(f, "w").write(out)
print(f"mooncake-prom: patched {f} (added MooncakeConnector.build_prom_metrics)")
