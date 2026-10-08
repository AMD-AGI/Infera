"""Exercise the real release hooks' Decode admission branch before/after patching."""

import ast
import importlib.util
import logging
import marshal
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "deploy/docker/patches/sglang_decode_radix_spec/patch_decode_radix_spec.py"
FIXTURES = Path(__file__).parent / "fixtures/decode_radix_spec"
spec = importlib.util.spec_from_file_location("patch_decode_radix_spec", PATCH)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


@pytest.fixture(params=["0.5.18", "0.5.19"])
def source(request):
    return (FIXTURES / f"pd_disaggregation_v{request.param}.py.txt").read_text()


def decode_branch(source, monkeypatch, **overrides):
    tree = ast.parse(source)
    hook = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "handle_pd_disaggregation"
    )
    branch = next(
        n
        for n in hook.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        in ("server_args.disaggregation_mode == 'decode'", "cfg.disaggregation_mode == 'decode'")
    )
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_infera_decode_radix_spec_allowed"
    ]
    cfg = SimpleNamespace(
        disaggregation_decode_enable_radix_cache=True,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        enable_hisparse=False,
        disaggregation_transfer_backend="mooncake",
        enable_dp_attention=False,
        disable_radix_cache=True,
        disaggregation_mode="decode",
        dcp_size=1,
    )
    cfg.__dict__.update(overrides)
    # Run the upstream radix branch verbatim; only the resolution container is a stand-in.
    module = ModuleType("sglang.srt.arg_groups.overrides")
    module.resolved_view = lambda args: args
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def declare_resolution(args, _source, **updates):
        args.__dict__.update(updates)

    scope = dict(
        os=os,
        cfg=cfg,
        server_args=cfg,
        logger=logging.getLogger(__name__),
        resolved_view=module.resolved_view,
        declare_resolution=declare_resolution,
    )
    dcp_checks = [
        n
        for n in hook.body
        if isinstance(n, ast.If)
        and "disaggregation_mode == 'decode'" in ast.unparse(n.test)
        and "dcp_size > 1" in ast.unparse(n.test)
    ]
    code = ast.Module(body=[*helpers, *dcp_checks, branch.body[0]], type_ignores=[])
    exec(compile(code, "upstream_decode_radix_branch", "exec"), scope)
    return cfg


def test_stock_release_rejects_even_when_infera_opt_in_is_set(source, monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", "1")
    with pytest.raises(ValueError, match="incompatible.*speculative"):
        decode_branch(source, monkeypatch)


@pytest.mark.parametrize("algorithm", ["EAGLE", "NEXTN"])
def test_patched_hook_enables_radix_only_for_supported_chain_drafts(source, monkeypatch, algorithm):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", "1")
    cfg = decode_branch(patcher.patch_source(source), monkeypatch, speculative_algorithm=algorithm)
    assert cfg.disable_radix_cache is False


@pytest.mark.parametrize("opt_in", [None, "0", "true"])
def test_runtime_default_remains_rejected(source, monkeypatch, opt_in):
    monkeypatch.delenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", opt_in)
    with pytest.raises(ValueError, match="incompatible.*speculative"):
        decode_branch(patcher.patch_source(source), monkeypatch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculative_eagle_topk": 2},
        {"dcp_size": 2},
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_algorithm": "NGRAM"},
        {"enable_hisparse": True},
        {"disaggregation_transfer_backend": "fake"},
    ],
)
def test_opt_in_does_not_bypass_other_rejections(source, monkeypatch, overrides):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", "1")
    with pytest.raises(ValueError):
        decode_branch(patcher.patch_source(source), monkeypatch, **overrides)


def test_non_speculative_and_chunk_cache_paths_are_unchanged(source, monkeypatch):
    monkeypatch.delenv("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", raising=False)
    assert (
        decode_branch(
            patcher.patch_source(source), monkeypatch, speculative_algorithm=None
        ).disable_radix_cache
        is False
    )
    assert (
        decode_branch(
            patcher.patch_source(source),
            monkeypatch,
            disaggregation_decode_enable_radix_cache=False,
        ).disable_radix_cache
        is True
    )


def test_patch_is_idempotent_and_replaces_stale_bytecode(source, tmp_path):
    target = tmp_path / "pd_disaggregation_hook.py"
    target.write_text(source)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "pd_disaggregation_hook.stale.pyc").write_bytes(b"stale")
    assert patcher.apply(target)
    assert not patcher.apply(target)
    assert not (cache / "pd_disaggregation_hook.stale.pyc").exists()
    bytecode = Path(importlib.util.cache_from_source(str(target))).read_bytes()
    module_code = marshal.loads(bytecode[16:])
    hook_code = next(
        c
        for c in module_code.co_consts
        if getattr(c, "co_name", None) == "handle_pd_disaggregation"
    )
    assert "_infera_decode_radix_spec_allowed" in hook_code.co_names


def test_unknown_guard_shape_fails_without_modifying_source(source, tmp_path):
    target = tmp_path / "hook.py"
    drifted = source.replace('"with speculative decoding "', '"changed upstream restriction "')
    target.write_text(drifted)
    with pytest.raises(ValueError, match="unsupported"):
        patcher.apply(target)
    assert target.read_text() == drifted


def test_default_mi35x_image_applies_engine_patch_without_enabling_runtime_flag():
    dockerfile = (REPO / "deploy/docker/Dockerfile.sglang").read_text()
    assert "ARG APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=1" in dockerfile
    assert (
        "COPY deploy/docker/patches/sglang_decode_radix_spec/patch_decode_radix_spec.py"
        in dockerfile
    )
    assert "python /tmp/patch_decode_radix_spec.py" in dockerfile
    assert "ENV SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC" not in dockerfile
