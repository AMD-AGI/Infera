###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The guard on --disaggregation-decode-enable-radix-cache.

infera appends that flag to the decode leg's argv when KV events are on, and
SGLang rejects it in more than one situation. Hybrid SWA/SSM models are the
expensive one: that rejection is raised inside build_kv_cache -- after the
weights are loaded -- so the guard has to catch it here or the decode leg dies
~2.5 minutes into startup. --enable-hisparse and dcp_size > 1 are rejected at
argument-resolution time, cheaply, but name a flag the operator never passed.

Guarded by importorskip("sglang") like the other args tests: the module
imports sglang.srt.server_args at load time.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang import args as args_mod  # noqa: E402
from infera.engine.sglang.args import no_clear_event_reason, parse_sglang_args  # noqa: E402

_DECODE = [
    "--model-path",
    "Qwen/Qwen3-0.6B",
    "--etcd-endpoint",
    "127.0.0.1:2379",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "mooncake",
]

_FLAG = "--disaggregation-decode-enable-radix-cache"


def _reason(monkeypatch, value):
    monkeypatch.setattr(args_mod, "_decode_radix_cache_unsupported_reason", lambda _sa: value)


def test_appends_for_a_supported_model(monkeypatch):
    _reason(monkeypatch, None)
    assert _FLAG in parse_sglang_args(_DECODE).sglang_argv


def test_skips_for_a_mamba_ssm_model(monkeypatch):
    _reason(monkeypatch, "Mamba/SSM")
    assert _FLAG not in parse_sglang_args(_DECODE).sglang_argv


def test_skips_for_a_hybrid_swa_model(monkeypatch):
    _reason(monkeypatch, "sliding window attention (SWA)")
    assert _FLAG not in parse_sglang_args(_DECODE).sglang_argv


def test_says_which_family_it_skipped_for(monkeypatch, caplog):
    _reason(monkeypatch, "Mamba/SSM")
    with caplog.at_level("INFO", logger=args_mod.logger.name):
        parse_sglang_args(_DECODE)
    assert "Mamba/SSM" in caplog.text


def test_an_explicit_flag_is_left_alone(monkeypatch):
    """The operator asked for it; the guard is only for what infera appends."""
    _reason(monkeypatch, "Mamba/SSM")
    argv = parse_sglang_args([*_DECODE, _FLAG]).sglang_argv
    assert argv.count(_FLAG) == 1


def test_prefill_leg_is_untouched(monkeypatch):
    _reason(monkeypatch, None)
    prefill = [*_DECODE[:-4], "--disaggregation-mode", "prefill", *_DECODE[-2:]]
    assert _FLAG not in parse_sglang_args(prefill).sglang_argv


def test_guard_never_raises_when_the_model_config_is_unreadable(monkeypatch):
    """A config we cannot read must not take the worker down here -- the
    engine fails for real a few lines later if it is genuinely broken."""

    class _Boom:
        def get_model_config(self):
            raise RuntimeError("no config.json")

    assert args_mod._decode_radix_cache_unsupported_reason(_Boom()) is None


def test_the_unreadable_config_warning_names_a_lever_that_works(caplog):
    """The warning used to say: pass the flag explicitly to take the decision
    out of infera's hands. It is a force-*on* -- the append is gated on the flag
    being absent -- so following that advice forwarded it to SGLang and produced
    the very ValueError the warning is about. There was no working opt-out."""

    class _Boom:
        def get_model_config(self):
            raise RuntimeError("no config.json")

    with caplog.at_level("WARNING", logger=args_mod.logger.name):
        args_mod._decode_radix_cache_unsupported_reason(_Boom())
    assert "--no-enable-kv-events" in caplog.text


def test_kv_events_off_is_that_lever(monkeypatch):
    """And it works for a model the guard would otherwise append for."""
    _reason(monkeypatch, None)
    argv = parse_sglang_args([*_DECODE, "--no-enable-kv-events"]).sglang_argv
    assert _FLAG not in argv


def test_guard_never_raises_when_sglangs_private_module_moves(monkeypatch):
    """``sglang.srt.configs.hybrid_arch`` is private API that moves between
    releases. A rename must degrade to the warning path like any other
    unreadable config, not propagate an ImportError out of argv parsing."""
    # A None entry in sys.modules is what makes `import x` raise ImportError.
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.hybrid_arch", None)

    class _Unused:
        def get_model_config(self):
            raise AssertionError("the import fails before the config is read")

    assert args_mod._decode_radix_cache_unsupported_reason(_Unused()) is None


def test_the_guard_still_speaks_sglangs_own_api(caplog):
    """The one test here that is not allowed to patch the predicate away.

    Every behavioural test above replaces ``_decode_radix_cache_unsupported_reason``
    with a constant, because that is the only way to exercise both branches
    without keeping two real models around. The cost is that none of them touch
    the private SGLang API the guard is built on -- ``hybrid_arch``'s five
    helpers, ``ModelConfig.is_hybrid_swa``, ``ModelSpec.uses_mamba_radix_cache``
    -- and the guard swallows every exception by design. A rename in that module
    therefore turns the guard into a permanent ``None`` with CI fully green, and
    the 2.5-minute-late ``build_kv_cache`` crash is back on the next hybrid
    model. That is the failure this file exists to prevent, so something has to
    call the real thing.

    Qwen3-0.6B is dense, so the *answer* is ``None`` either way. What is asserted
    is how it got there: the predicates ran to the end rather than the ``except``
    clause catching an ImportError or an AttributeError.
    """
    sa = parse_sglang_args(_DECODE).server_args
    with caplog.at_level("WARNING", logger=args_mod.logger.name):
        assert args_mod._decode_radix_cache_unsupported_reason(sa) is None
    assert "could not determine" not in caplog.text, (
        "the guard fell into its own exception path -- sglang's private hybrid "
        "API moved, and the guard is now silently answering 'supported' for "
        "every model, hybrid ones included"
    )


# --- the rejections that fire before the weights load -------------------------


class _ArgOnly:
    """A server_args whose model config must never be reached.

    ``--enable-hisparse`` and ``dcp_size > 1`` are rejected by
    ``pd_disaggregation_hook`` off the arguments alone. Reading them off the
    model config instead would put them behind the guard's ``except``, where an
    unreadable config answers "supported" -- so this stands in for that config
    and fails loudly if anything tries.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_model_config(self):
        raise AssertionError("an argv-only rejection must not need the model config")


def test_hisparse_is_rejected_without_reading_the_config():
    reason = args_mod._decode_radix_cache_unsupported_reason(_ArgOnly(enable_hisparse=True))
    assert reason is not None and "--enable-hisparse" in reason


def test_decode_dcp_is_rejected_without_reading_the_config():
    reason = args_mod._decode_radix_cache_unsupported_reason(_ArgOnly(dcp_size=2))
    assert reason is not None and "--dcp-size" in reason


def test_an_older_sglang_without_those_flags_is_not_a_rejection(caplog):
    """Both flags postdate SGLang releases we still run against. Absent, there
    is nothing to pre-empt -- and, crucially, no warning either: the getattr
    defaults must not read as an unreadable config."""

    class _Old:
        def get_model_config(self):
            raise RuntimeError("stop here; the point is what came before")

    with caplog.at_level("WARNING", logger=args_mod.logger.name):
        assert args_mod._decode_radix_cache_unsupported_reason(_Old()) is None


def test_the_skip_message_reads_for_a_flag_and_not_only_a_model_family(monkeypatch, caplog):
    """The caller used to interpolate the reason into "incompatible with %s
    models", which was true of the only two reasons it could get. A flag name
    in that slot read as "incompatible with --enable-hisparse models"."""
    _reason(monkeypatch, "--enable-hisparse")
    with caplog.at_level("INFO", logger=args_mod.logger.name):
        assert _FLAG not in parse_sglang_args(_DECODE).sglang_argv
    assert "incompatible with --enable-hisparse;" in caplog.text


# --- what the startup flush is allowed to wait for ----------------------------


def test_a_guarded_decode_leg_has_no_clear_event_to_wait_for(monkeypatch):
    """The guard's refusal and the startup flush have to agree.

    Skipping the flag leaves the decode leg on SGLang's ``ChunkCache``, whose
    ``reset()`` is ``pass``. ``/flush_cache`` still answers 200, so
    ``anchor_kv_chain`` could not tell that nothing had happened: it spent its
    whole ~10s budget immediately before ``register()`` and then warned that the
    router's chain has no anchor -- on a leg that has no chain. Kimi-K3 is
    exactly this case, which makes it the normal decode leg, not a corner.
    """
    _reason(monkeypatch, "Mamba/SSM")
    args = parse_sglang_args(_DECODE)
    assert _FLAG not in args.sglang_argv
    reason = no_clear_event_reason(args)
    assert reason is not None and "ChunkCache" in reason


def test_a_decode_leg_with_the_radix_cache_does_have_one(monkeypatch):
    """And when the flag went in, the radix tree is there to be cleared, so the
    flush is worth doing -- this is the leg whose anchor really was lost."""
    _reason(monkeypatch, None)
    args = parse_sglang_args(_DECODE)
    assert _FLAG in args.sglang_argv
    assert no_clear_event_reason(args) is None


def test_an_aggregated_worker_on_chunk_cache_has_no_clear_event_either(monkeypatch):
    """--disable-radix-cache is the other way onto ChunkCache, and it needs no
    PD at all. Uncovered, an aggregated worker started with it spent the whole
    anchor budget immediately before register() and then blamed a missing anchor
    on a worker that emits no KV events in the first place."""
    _reason(monkeypatch, None)
    agg = ["--model-path", "Qwen/Qwen3-0.6B", "--etcd-endpoint", "127.0.0.1:2379"]
    reason = no_clear_event_reason(parse_sglang_args([*agg, "--disable-radix-cache"]))
    assert reason is not None and "--disable-radix-cache" in reason
    assert no_clear_event_reason(parse_sglang_args(agg)) is None


def test_a_decode_leg_is_read_off_the_argv_not_the_resolved_attribute(monkeypatch):
    """SGLang's pd_disaggregation_hook sets server_args.disable_radix_cache =
    True for *every* decode leg, before infera appends the flag that turns it
    back off. Reading that attribute in decode mode -- the obvious way to write
    the check added above -- reports ChunkCache for a leg that will run a radix
    cache, and skips the one flush that had a chain to re-anchor."""
    _reason(monkeypatch, None)
    args = parse_sglang_args(_DECODE)
    assert _FLAG in args.sglang_argv
    assert args.server_args.disable_radix_cache is True, (
        "sglang stopped forcing the attribute for decode legs; the argv read "
        "below is now belt-and-braces rather than load-bearing"
    )
    assert no_clear_event_reason(args) is None


def test_a_prefill_leg_is_flushed_as_before(monkeypatch):
    """The prefill leg keeps a radix cache regardless of that flag; it is also
    the leg that carries prefix-aware routing, so it is the one that must not
    lose its anchor."""
    _reason(monkeypatch, None)
    prefill = [*_DECODE[:-4], "--disaggregation-mode", "prefill", *_DECODE[-2:]]
    assert no_clear_event_reason(parse_sglang_args(prefill)) is None
