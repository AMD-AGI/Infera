###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Pure-logic checks for model staging and speculative-decoding guards."""

from __future__ import annotations

import importlib
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from tests.e2e.harness import (
    EngineAdapter,
    cluster,
    disagg_fixtures,
    launcher,
    resources,
    speculation,
)
from tests.e2e.harness.params import DisaggRole, EngineParams


class _Adapter(EngineAdapter):
    def gpus_per_worker(self, params):
        return 1

    def build_argv(self, params, *, port, host, server_ctx, gpu_ids):
        return []


@pytest.mark.parametrize(
    ("engine", "expected"),
    [("sglang", "5400"), ("vllm", "5400"), ("atom", None)],
)
def test_worker_env_propagates_engine_ready_timeout(engine, expected):
    adapter = _Adapter()
    adapter.engine = engine
    env = adapter.worker_env(EngineParams(server_ready_timeout=5400), gpu_ids=[2, 3])

    assert env["HIP_VISIBLE_DEVICES"] == "2,3"
    assert env.get("INFERA_ENGINE_READY_TIMEOUT") == expected


def test_explicit_engine_ready_timeout_wins():
    adapter = _Adapter()
    adapter.engine = "vllm"
    params = EngineParams(
        extra_env=(("INFERA_ENGINE_READY_TIMEOUT", "6000"),),
        server_ready_timeout=5400,
    )

    assert adapter.worker_env(params, gpu_ids=[0])["INFERA_ENGINE_READY_TIMEOUT"] == "6000"


@pytest.mark.parametrize(
    ("module_name", "adapter_name", "has_engine_timeout"),
    [
        ("tests.e2e.pd_disag.sglang.conftest", "SglangDisaggAdapter", True),
        ("tests.e2e.pd_disag.vllm.conftest", "VllmDisaggAdapter", True),
        ("tests.e2e.pd_disag.atom.conftest", "AtomDisaggAdapter", False),
    ],
)
def test_disagg_adapter_preserves_base_worker_env(
    monkeypatch, module_name, adapter_name, has_engine_timeout
):
    monkeypatch.setenv("INFERA_E2E_GFX_ARCH", "gfx942")
    module = importlib.import_module(module_name)
    adapter = getattr(module, adapter_name)()
    params = EngineParams(
        extra_env=(("CASE_ENV", "kept"),),
        server_ready_timeout=5400,
    )

    env = adapter.disagg_worker_env(
        params,
        DisaggRole.DECODE,
        advertise_host="192.0.2.10",
        gpu_ids=[4, 5],
        gid_index="3",
    )

    assert env["HIP_VISIBLE_DEVICES"] == "4,5"
    assert env["CASE_ENV"] == "kept"
    assert ("INFERA_ENGINE_READY_TIMEOUT" in env) is has_engine_timeout
    if has_engine_timeout:
        assert env["INFERA_ENGINE_READY_TIMEOUT"] == "5400"


def test_vllm_mixed_rejects_dp_attention():
    module = importlib.import_module("tests.e2e.pd_mixed.vllm.conftest")
    with pytest.raises(ValueError, match="unsupported"):
        module.VllmAdapter().build_argv(
            EngineParams(dp_attention=True),
            port=8000,
            host="127.0.0.1",
            server_ctx={"etcd_endpoint": "127.0.0.1:2379", "etcd_prefix": "/test/"},
            gpu_ids=[0],
        )


def test_vllm_disagg_rejects_dp_attention(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_GFX_ARCH", "gfx942")
    module = importlib.import_module("tests.e2e.pd_disag.vllm.conftest")
    with pytest.raises(ValueError, match="unsupported"):
        module.VllmDisaggAdapter().build_disagg_argv(
            EngineParams(dp_attention=True),
            DisaggRole.DECODE,
            port=8000,
            host="0.0.0.0",
            server_ctx={"etcd_endpoint": "127.0.0.1:2379", "etcd_prefix": "/test/"},
            advertise_host="192.0.2.10",
            gpu_ids=[0],
        )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--speculative-algorithm", "EAGLE"), True),
        (("--speculative-config", '{"method":"mtp"}'), True),
        (("--speculative-config", "{}"), False),
        (("--method", "mtp"), True),
        (("--method", "other"), False),
        ((), False),
    ],
)
def test_mtp_request_detection(args, expected):
    assert speculation.mtp_requested(EngineParams(extra_args=args)) is expected


def test_spec_counters_sum_rank_samples():
    metrics = """
vllm:spec_decode_num_draft_tokens_total{rank="0"} 3
vllm:spec_decode_num_draft_tokens_total{rank="1"} 4
vllm:spec_decode_num_accepted_tokens_total 5
unrelated_metric 99
"""
    assert speculation._spec_counters(metrics) == {
        "vllm:spec_decode_num_draft_tokens_total": 7.0,
        "vllm:spec_decode_num_accepted_tokens_total": 5.0,
    }


def _mock_metrics(monkeypatch, text):
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.pop("timeout", None)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=text))
        return real_client(transport=transport, timeout=5)

    monkeypatch.setattr(speculation.httpx, "AsyncClient", client)


@pytest.mark.asyncio
async def test_requested_but_inactive_speculation_fails(monkeypatch):
    _mock_metrics(monkeypatch, "vllm:spec_decode_num_draft_tokens_total 0")
    params = EngineParams(extra_args=("--speculative-config", '{"method":"mtp"}'))
    with pytest.raises(AssertionError, match="drafted nothing"):
        await speculation.report_speculation(8000, params, engine="vllm")


@pytest.mark.asyncio
async def test_active_speculation_passes(monkeypatch):
    _mock_metrics(monkeypatch, "vllm:spec_decode_num_draft_tokens_total 4")
    params = EngineParams(extra_args=("--speculative-config", '{"method":"mtp"}'))
    await speculation.report_speculation(8000, params, engine="vllm")


@pytest.mark.asyncio
async def test_requested_vllm_without_counters_fails(monkeypatch):
    _mock_metrics(monkeypatch, "")
    params = EngineParams(extra_args=("--speculative-config", '{"method":"mtp"}'))
    with pytest.raises(AssertionError, match="cannot verify"):
        await speculation.report_speculation(8000, params, engine="vllm")


def test_explicit_disagg_nodes_turn_environment_skip_into_failure(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_NODES", "node-a,node-b")
    with pytest.raises(pytest.fail.Exception, match="bad disagg environment"):
        disagg_fixtures._skip_or_fail("bad disagg environment")


def test_direct_pytest_without_nodes_can_skip(monkeypatch):
    monkeypatch.delenv("INFERA_E2E_NODES", raising=False)
    with pytest.raises(pytest.skip.Exception, match="not configured"):
        disagg_fixtures._skip_or_fail("not configured")


def test_spur_attached_srun_reuses_holder_allocation(monkeypatch):
    monkeypatch.setattr(cluster, "_SPUR", True)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("INFERA_E2E_SLURM_PARTITION", "amd-spur")
    monkeypatch.setenv("INFERA_E2E_RESERVATION", "expired-reservation")
    monkeypatch.setenv("INFERA_E2E_SRUN_EXTRA", "-A wrong-account -q wrong-qos")
    monkeypatch.setenv("INFERA_E2E_STEP_SRUN_EXTRA", "--cpu-bind=none")

    argv = cluster.srun_argv("node-a")

    assert argv[:5] == ["srun", "-N1", "-n1", "-w", "node-a"]
    assert "-p" not in argv
    assert not any(arg.startswith("--reservation") for arg in argv)
    assert "--jobid" not in argv
    assert "wrong-account" not in argv
    assert argv[-1] == "--cpu-bind=none"


def test_spur_detached_srun_keeps_submission_placement(monkeypatch):
    monkeypatch.setattr(cluster, "_SPUR", True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setenv("INFERA_E2E_SLURM_PARTITION", "amd-spur")
    monkeypatch.setenv("INFERA_E2E_RESERVATION", "ci-reservation")

    argv = cluster.srun_argv("node-a")

    assert argv[:7] == ["srun", "-N1", "-n1", "-p", "amd-spur", "-w", "node-a"]
    assert "--reservation=ci-reservation" in argv


def test_step_access_reports_pending_allocation(monkeypatch):
    cancelled = []

    def pending(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            ["srun"],
            kwargs["timeout"],
            stderr=b"srun: Pending job allocation 100477...\n",
        )

    monkeypatch.setattr(cluster, "run_on_node", pending)
    monkeypatch.setattr(cluster.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        cluster.subprocess,
        "run",
        lambda argv, **kwargs: cancelled.append(argv) or SimpleNamespace(returncode=0),
    )

    with pytest.raises(
        RuntimeError,
        match=r"node-a did not start within 7s: srun: Pending job allocation 100477",
    ):
        cluster.require_step_access("node-a", timeout=7)
    assert cancelled == [["scancel", "100477"]]


def test_disagg_cleanup_and_launch_are_job_scoped(monkeypatch, tmp_path):
    calls = []

    def run(node, argv, *, timeout):
        calls.append((node, argv, timeout))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("INFERA_E2E_JOB_TAG", "run/engine")
    monkeypatch.setattr(launcher, "_srun", run)
    remote = launcher.SrunDockerLauncher(
        image="test-image",
        dockerfile="Dockerfile",
        log_dir=str(tmp_path),
    )

    remote.cleanup_stale(["node-a"])
    cleanup_script = calls[-1][1][-1]
    assert "label=infera.e2e.job_tag=run-engine" in cleanup_script
    assert "name=infera-e2e-" not in cleanup_script
    assert "name=infera-utest-" not in cleanup_script

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    remote.cleanup_stale(["node-a"])
    cleanup_script = calls[-1][1][-1]
    assert "--filter label=infera.e2e.job_tag " in cleanup_script
    assert "name=infera-e2e-" in cleanup_script
    assert "name=infera-utest-" in cleanup_script
    assert r"$0 !~ /infera\.e2e\.job_tag=/" in cleanup_script

    remote._run("node-a", "test-container", "test-image", [], ["true"])
    launch_argv = calls[-1][1]
    assert launch_argv[launch_argv.index("--label") + 1] == "infera.e2e.job_tag=run-engine"


def test_staged_model_with_config_passes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    resources.require_model_staged(EngineParams(model=str(tmp_path)))


def test_staged_model_without_config_fails(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="config.json is unreadable"):
        resources.require_model_staged(EngineParams(model=str(tmp_path)))


def test_remote_model_id_is_left_to_the_engine():
    resources.require_model_staged(EngineParams(model="org/model-not-local"))
