"""Check Infera's argv forwarding without loading an engine or model."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

FLAG = "--disaggregation-decode-enable-radix-cache"
OPT_IN = "SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC"


@pytest.fixture
def args_module(monkeypatch):
    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model-path")
            parser.add_argument("--disaggregation-mode", default="decode")
            parser.add_argument("--disaggregation-transfer-backend", default="mooncake")
            parser.add_argument("--speculative-algorithm")
            parser.add_argument("--speculative-eagle-topk", type=int, default=1)
            parser.add_argument(FLAG, action="store_true")
            parser.add_argument("--enable-hierarchical-cache", action="store_true")
            parser.add_argument("--kv-cache-dtype", default="auto")
            parser.add_argument("--disable-radix-cache", action="store_true")
            parser.add_argument("--hicache-ratio", type=float, default=1.5)
            parser.add_argument("--hicache-write-policy", default="write_through")
            parser.add_argument("--hicache-io-backend", default="kernel")
            parser.add_argument("--hicache-mem-layout", default="page_first")

        @staticmethod
        def from_cli_args(parsed):
            # Model SGLang's constructor boundary, before Infera can return argv.
            if (
                parsed.disaggregation_mode == "decode"
                and parsed.enable_hierarchical_cache
                and not parsed.disaggregation_decode_enable_radix_cache
            ):
                raise ValueError("HiCache and ChunkCache are mutually exclusive")
            return parsed

    for name in ("sglang", "sglang.srt", "sglang.srt.server_args"):
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules["sglang.srt.server_args"].ServerArgs = ServerArgs
    path = Path(__file__).resolve().parents[3] / "infera/engine/sglang/args.py"
    spec = importlib.util.spec_from_file_location("_infera_radix_args_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_decode_radix_cache_unsupported_reason", lambda _: None)
    monkeypatch.delenv(OPT_IN, raising=False)
    return module


def argv(*extra):
    return [
        "--model-path",
        "test-model",
        "--discovery-backend",
        "etcd",
        "--etcd-endpoint",
        "127.0.0.1:2379",
        *extra,
    ]


@pytest.mark.parametrize("enabled", [None, "0", "true", "1"])
def test_speculative_radix_requires_exact_opt_in(args_module, monkeypatch, enabled):
    if enabled is not None:
        monkeypatch.setenv(OPT_IN, enabled)
    parsed = args_module.parse_sglang_args(argv("--speculative-algorithm", "EAGLE"))
    assert (FLAG in parsed.sglang_argv) == (enabled == "1")
    assert parsed.sglang_argv.count("--speculative-algorithm") == 1


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_non_speculative_decode_keeps_existing_default(args_module, monkeypatch, enabled):
    monkeypatch.setenv(OPT_IN, enabled)
    assert FLAG in args_module.parse_sglang_args(argv()).sglang_argv


@pytest.mark.parametrize("reason", ["Mamba/SSM", "hybrid SWA", "--enable-hisparse", "DCP"])
def test_opt_in_preserves_model_and_topology_rejections(args_module, monkeypatch, reason):
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(args_module, "_decode_radix_cache_unsupported_reason", lambda _: reason)
    assert (
        FLAG
        not in args_module.parse_sglang_args(argv("--speculative-algorithm", "EAGLE")).sglang_argv
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--disaggregation-mode", "prefill"],
        ["--disaggregation-transfer-backend", "mori"],
        ["--no-enable-kv-events"],
    ],
)
def test_opt_in_does_not_enable_unrelated_paths(args_module, monkeypatch, extra):
    monkeypatch.setenv(OPT_IN, "1")
    parsed = args_module.parse_sglang_args(argv("--speculative-algorithm", "EAGLE", *extra))
    assert FLAG not in parsed.sglang_argv


def test_explicit_flag_is_not_duplicated(args_module, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    parsed = args_module.parse_sglang_args(argv("--speculative-algorithm", "EAGLE", FLAG))
    assert parsed.sglang_argv.count(FLAG) == 1


@pytest.mark.parametrize("algorithm", [None, "EAGLE", "NEXTN"])
def test_hicache_receives_radix_before_server_args_resolution(args_module, monkeypatch, algorithm):
    monkeypatch.setenv(OPT_IN, "1")
    spec_args = [] if algorithm is None else ["--speculative-algorithm", algorithm]
    hicache_args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.5",
        "--hicache-write-policy",
        "write_through",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
    ]
    parsed = args_module.parse_sglang_args(argv(*spec_args, *hicache_args))
    assert parsed.server_args.disaggregation_decode_enable_radix_cache is True
    assert parsed.server_args.enable_hierarchical_cache is True
    assert parsed.sglang_argv.count(FLAG) == 1
    for flag in hicache_args:
        assert flag in parsed.sglang_argv


@pytest.mark.parametrize("reason", ["Mamba/SSM", "hybrid SWA", "--enable-hisparse", "DCP"])
def test_hicache_auto_radix_rejects_unsupported_models(args_module, monkeypatch, reason):
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(args_module, "_decode_radix_cache_unsupported_reason", lambda _: reason)
    with pytest.raises(ValueError, match="Decode HiCache requires a supported radix cache"):
        args_module.parse_sglang_args(
            argv("--speculative-algorithm", "EAGLE", "--enable-hierarchical-cache")
        )


def test_hicache_does_not_override_explicit_disable(args_module, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    with pytest.raises(ValueError, match="--disable-radix-cache"):
        args_module.parse_sglang_args(argv("--enable-hierarchical-cache", "--disable-radix-cache"))


def test_hicache_under_mtp_still_requires_opt_in(args_module):
    with pytest.raises(ValueError, match="mutually exclusive"):
        args_module.parse_sglang_args(
            argv("--speculative-algorithm", "EAGLE", "--enable-hierarchical-cache")
        )


def test_hicache_explicit_radix_flag_remains_supported(args_module, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    parsed = args_module.parse_sglang_args(
        argv("--speculative-algorithm", "EAGLE", "--enable-hierarchical-cache", FLAG)
    )
    assert parsed.sglang_argv.count(FLAG) == 1
    assert parsed.server_args.enable_hierarchical_cache is True


def test_hicache_does_not_require_kv_event_publication(args_module, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    parsed = args_module.parse_sglang_args(
        argv(
            "--speculative-algorithm",
            "EAGLE",
            "--enable-hierarchical-cache",
            "--no-enable-kv-events",
        )
    )
    assert parsed.enable_kv_events is False
    assert parsed.server_args.disaggregation_decode_enable_radix_cache is True
    assert parsed.sglang_argv.count(FLAG) == 1
