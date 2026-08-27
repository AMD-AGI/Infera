###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Per-architecture targeting of the e2e case tables, as pure logic.

The e2e suites need 8 GPUs and half an hour, but the rules deciding *which*
architecture they target and *which* knobs a case gets there are plain dict
merging — so they belong in the unit tier, where a mistake costs a second
instead of a node reservation.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.e2e.harness import arch, images
from tests.e2e.harness.matrix import apply_arch_overlay, expand_cases

# Every engine's real case table, so the "same matrix on both arches" invariant
# is checked against what actually ships, not a fixture.
MATRIX_MODULES = [
    "tests.e2e.pd_mixed.sglang.matrix",
    "tests.e2e.pd_mixed.vllm.matrix",
    "tests.e2e.pd_mixed.atom.matrix",
    "tests.e2e.pd_disag.sglang.matrix",
    "tests.e2e.pd_disag.vllm.matrix",
    "tests.e2e.pd_disag.atom.matrix",
]


@pytest.fixture
def no_gpu(monkeypatch):
    """Pin the probe to "no GPU visible" so resolution tests are deterministic
    whether they run on a login host or on an MI355X node."""
    monkeypatch.setattr(arch, "probe_arch", lambda: None)


# --- overlay merge ---


def test_no_overlay_is_the_identity():
    opts = {"args": ["--a"], "env": {"X": "1"}, "server_ready_timeout": 900}
    assert apply_arch_overlay(opts, "gfx950") == opts


def test_other_archs_overlay_is_dropped_not_applied():
    opts = {"args": ["--base"], "gfx942": {"args": ["--other"]}}
    assert apply_arch_overlay(opts, "gfx950") == {"args": ["--base"]}


def test_args_replace_wholesale():
    opts = {
        "args": ["--attention-backend", "triton", "--kv-cache-dtype", "fp8_e4m3"],
        "gfx942": {"args": ["--attention-backend", "aiter"]},
    }
    assert apply_arch_overlay(opts, "gfx942") == {"args": ["--attention-backend", "aiter"]}


def test_setup_and_timeout_replace():
    opts = {
        "setup": ["pip install amd-quark"],
        "server_ready_timeout": 900,
        "gfx942": {"setup": ["pip install amd-quark==0.9"], "server_ready_timeout": 2400},
    }
    merged = apply_arch_overlay(opts, "gfx942")
    assert merged["setup"] == ["pip install amd-quark==0.9"]
    assert merged["server_ready_timeout"] == 2400


def test_env_merges_per_key():
    opts = {
        "env": {"KEEP": "1", "OVERRIDE": "base"},
        "gfx942": {"env": {"OVERRIDE": "arch", "ADD": "1"}},
    }
    assert apply_arch_overlay(opts, "gfx942")["env"] == {
        "KEEP": "1",
        "OVERRIDE": "arch",
        "ADD": "1",
    }


def test_env_none_deletes_the_base_key():
    opts = {"env": {"DROP": "1", "KEEP": "1"}, "gfx942": {"env": {"DROP": None}}}
    assert apply_arch_overlay(opts, "gfx942")["env"] == {"KEEP": "1"}


def test_env_survives_an_overlay_that_does_not_mention_it():
    opts = {"env": {"KEEP": "1"}, "gfx942": {"args": ["--x"]}}
    assert apply_arch_overlay(opts, "gfx942")["env"] == {"KEEP": "1"}


def test_overlay_may_introduce_env():
    opts = {"args": ["--x"], "gfx942": {"env": {"HSA_NO_SCRATCH_RECLAIM": "1"}}}
    assert apply_arch_overlay(opts, "gfx942")["env"] == {"HSA_NO_SCRATCH_RECLAIM": "1"}


def test_skip_reaches_the_params_and_only_on_that_arch():
    table = [[True, "Qwen/Qwen3-0.6B", 1, False, False, {"gfx942": {"skip": "no FP4 kernel"}}]]
    assert expand_cases(table, arch="gfx942")[0].skip_reason == "no FP4 kernel"
    assert expand_cases(table, arch="gfx950")[0].skip_reason == ""


@pytest.mark.parametrize(
    "opts",
    [
        {"arg": ["--typo"]},  # knob typo
        {"gfx940": {"args": []}},  # architecture typo
        {"gfx942": {"agrs": []}},  # knob typo inside an overlay
        {"gfx942": {"gfx950": {}}},  # nested overlay
    ],
)
def test_unknown_keys_raise(opts):
    with pytest.raises(ValueError, match="unknown key"):
        apply_arch_overlay(opts, "gfx942")


# --- architecture resolution ---


def test_a_declaration_overrides_the_live_gpu(monkeypatch):
    monkeypatch.setattr(arch, "probe_arch", lambda: "gfx950")
    monkeypatch.setenv(arch.ARCH_ENV, "gfx942")
    assert arch.target_arch() == "gfx942"


def test_the_live_gpu_decides_when_nothing_is_declared(monkeypatch):
    monkeypatch.delenv(arch.ARCH_ENV, raising=False)
    monkeypatch.setattr(arch, "probe_arch", lambda: "gfx942")
    assert arch.target_arch() == "gfx942"


def test_default_when_there_is_nothing_to_go_on(monkeypatch, no_gpu):
    monkeypatch.delenv(arch.ARCH_ENV, raising=False)
    assert arch.target_arch() == arch.DEFAULT_ARCH == "gfx950"


def test_blank_declaration_reads_as_unset(monkeypatch, no_gpu):
    monkeypatch.setenv(arch.ARCH_ENV, "  ")
    assert arch.target_arch() == arch.DEFAULT_ARCH


@pytest.mark.parametrize("value", ["gfx90a", "GFX942", "gfx9421", "mi325x"])
def test_unsupported_declaration_raises(monkeypatch, value):
    monkeypatch.setenv(arch.ARCH_ENV, value)
    with pytest.raises(RuntimeError, match=arch.ARCH_ENV):
        arch.target_arch()


# --- the declared-vs-actual guard ---


def test_guard_reports_a_contradiction(monkeypatch):
    monkeypatch.setenv(arch.ARCH_ENV, "gfx942")
    monkeypatch.setattr(arch, "probe_arch", lambda: "gfx950")
    problem = arch.check_arch()
    assert problem and "gfx942" in problem and "gfx950" in problem


def test_guard_is_quiet_when_they_agree(monkeypatch):
    monkeypatch.setenv(arch.ARCH_ENV, "gfx942")
    monkeypatch.setattr(arch, "probe_arch", lambda: "gfx942")
    assert arch.check_arch() is None


def test_guard_is_quiet_with_no_gpu_to_check_against(monkeypatch, no_gpu):
    monkeypatch.setenv(arch.ARCH_ENV, "gfx942")
    assert arch.check_arch() is None


def test_guard_is_quiet_when_nothing_was_declared(monkeypatch):
    """An auto-detected arch cannot contradict itself; only a declaration can."""
    monkeypatch.delenv(arch.ARCH_ENV, raising=False)
    monkeypatch.setattr(arch, "probe_arch", lambda: "gfx942")
    assert arch.check_arch() is None


# --- reading a remote node's arch out of rocminfo ---

# Trimmed from real `rocminfo` output on an MI300X: CPU agents first, and each GPU
# agent carries both a bare gfx name and an ISA name that also matches "Name:".
ROCMINFO = """\
Agent 1
  Name:                    Intel(R) Xeon(R) Platinum 8480+
  Marketing Name:          Intel(R) Xeon(R) Platinum 8480+
Agent 3
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
"""


def test_the_gpu_agent_wins_over_the_cpu_agents_listed_before_it():
    assert arch.parse_rocminfo(ROCMINFO) == "gfx942"


def test_the_isa_name_is_not_mistaken_for_the_arch():
    """`amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-` is also a "Name:" line, and feeding
    it to the image table would miss every entry."""
    assert "amdgcn" not in (arch.parse_rocminfo(ROCMINFO) or "")


@pytest.mark.parametrize("text", ["", "Name: Intel(R) Xeon(R)\n", "rocminfo: command not found"])
def test_output_with_no_gpu_agent_reads_as_cannot_tell(text):
    assert arch.parse_rocminfo(text) is None


# --- images ---


def test_sglang_is_the_only_engine_with_a_gfx942_image():
    assert images.engine_image("sglang", "gfx942") != images.engine_image("sglang", "gfx950")
    assert images.engine_image("vllm", "gfx942") == images.engine_image("vllm", "gfx950")
    assert images.engine_image("atom", "gfx942") == images.engine_image("atom", "gfx950")


def test_the_gfx942_sglang_image_uses_the_gfx942_dockerfile():
    tag, dockerfile = images.engine_image("sglang", "gfx942")
    assert "gfx942" in tag
    assert dockerfile == "deploy/docker/Dockerfile.sglang.gfx942"


def test_every_engine_and_arch_pair_is_registered():
    for engine in ("sglang", "vllm", "atom"):
        for gfx in arch.SUPPORTED_ARCHS:
            assert images.engine_image(engine, gfx)


def test_an_unregistered_pair_raises_instead_of_guessing():
    with pytest.raises(RuntimeError, match="no e2e image registered"):
        images.engine_image("trtllm", "gfx950")


def test_run_tests_sh_names_the_same_images_the_harness_launches():
    """run_tests.sh keeps its own copy of this table — it runs on a bare host, before any
    container, so it cannot import one that needs httpx and pytest. Two copies drift into
    `docker build A` followed by `docker run B`, which reports as a missing image rather
    than as a disagreement, so pin them here instead."""
    script = (Path(__file__).resolve().parents[3] / "tests" / "run_tests.sh").read_text()
    for engine in ("sglang", "vllm", "atom"):
        for gfx in arch.SUPPORTED_ARCHS:
            for value in images.engine_image(engine, gfx):
                assert value in script, f"run_tests.sh does not name {value} ({engine}/{gfx})"


# --- the property the whole design exists to keep ---


@pytest.mark.parametrize("module", MATRIX_MODULES)
def test_the_matrix_is_the_same_matrix_on_both_architectures(module):
    """Same rows, same ids, same order — an overlay may only change the knobs. One that
    adds, drops or renames a case means the two arms stopped testing the same thing."""
    cases = importlib.import_module(module).CASES
    ids = {gfx: [p.id() for p in expand_cases(cases, arch=gfx)] for gfx in arch.SUPPORTED_ARCHS}
    assert ids["gfx942"] == ids["gfx950"]
