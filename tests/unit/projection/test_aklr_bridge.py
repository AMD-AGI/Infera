###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A replay's kernel tasks become AKLR schemas, and AKLR kernels' measured speedups come back as speedups.

The targets are the tasks an AKLR contract expresses, one per schema, with the
shapes the replay runs as its workloads; the speedups are the kernels' times
against the baseline at those shapes, matched back to the ops that run them.
"""

from __future__ import annotations

import json

import pytest

from infera.projection.core.projection.inference_projection import aklr_bridge as ab
from infera.projection.core.projection.inference_projection import kernel_speedups as ksp


def _case(m, n, k, calls, us, **extra):
    return {
        "dims": {"m": m, "n": n, "k": k, "groups": 1, "topk": 0, **extra},
        "calls": calls,
        "est_us": us,
    }


TASKS = {
    "totals": {"est_gpu_ms": {"engine": 1000.0}},
    "tasks": [
        {
            "op": "attn_qkv",
            "kind": "gemm",
            "dtype": "bf16",
            "cases": [_case(16384, 5120, 2880, 10, 300.0), _case(64, 5120, 2880, 1000, 10.0)],
        },
        # another op of the same shape shares the schema
        {
            "op": "attn_k",
            "kind": "gemm",
            "dtype": "bf16",
            "cases": [_case(64, 5120, 2880, 1000, 20.0)],
        },
        {
            "op": "attn_o",
            "kind": "gemm",
            "dtype": "bf16",
            "cases": [_case(64, 2880, 4096, 1000, 8.0)],
        },
        {
            "op": "rmsnorm",
            "kind": "norm",
            "dtype": "bf16",
            "cases": [{"dims": {"tokens": 64, "width": 2880}, "calls": 2000, "est_us": 4.0}],
        },
        # no AKLR contract: MXFP4 experts, grouped
        {
            "op": "moe_gate_up",
            "kind": "gemm",
            "dtype": "mxfp4",
            "cases": [_case(64, 6144, 3072, 1000, 7.0, groups=128, topk=4)],
        },
    ],
}


def _by_name(found):
    return {t.name: t for t in found}


def test_targets_one_per_schema_with_the_replays_shapes():
    found = _by_name(ab.targets(TASKS))
    assert set(found) == {"gemm_n5120_k2880", "gemm_n2880_k4096", "add_rms_norm_hidden_size2880"}
    qkv = found["gemm_n5120_k2880"]
    assert qkv.contract == "gemm@1.0" and qkv.const == {"n": 5120, "k": 2880}
    assert [w.value for w in qkv.workloads] == [64, 16384]
    m64 = qkv.workloads[0]
    assert (
        m64.calls == 2000
        and m64.est_us == pytest.approx(15.0)
        and sorted(m64.ops) == ["attn_k", "attn_qkv"]
    )
    # 2000 x 15 us + 10 x 300 us of 1000 ms
    assert qkv.share_pct == pytest.approx(3.3)
    norm = found["add_rms_norm_hidden_size2880"]
    assert (
        norm.var_axis == "n_tokens"
        and norm.var_dim == "tokens"
        and norm.const == {"hidden_size": 2880}
    )


def test_targets_filters_by_size_share_and_op():
    decode = _by_name(ab.targets(TASKS, max_var=512))
    assert [w.value for w in decode["gemm_n5120_k2880"].workloads] == [64]
    assert set(ab.targets(TASKS, min_share_pct=2.0, max_var=512)[0].ops) == {"attn_k", "attn_qkv"}
    assert [t.name for t in ab.targets(TASKS, ops=["attn_o"])] == ["gemm_n2880_k4096"]


def test_targets_round_trip(tmp_path):
    found = ab.targets(TASKS)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(ab.targets_document(found)))
    again = ab.load_targets(str(path))
    assert [t.to_dict() for t in again] == [t.to_dict() for t in found]


def test_campaign_script_prepares_then_loops():
    qkv = _by_name(ab.targets(TASKS))["gemm_n5120_k2880"]
    text = ab.campaign_script(
        qkv,
        aklr_root="/a",
        schema_path="/a/s.json",
        run_dir="/r",
        hours=3,
        model="m",
        gpu=2,
        anchor_start=True,
    )
    assert (
        "forge_task prepare --schema-file /a/s.json --run-dir /r --kernel-backend triton --anchor start"
        in text
    )
    assert "/r/forge_loop.sh --max-hours 3 --model m --gpu 2 --lanes 1" in text
    assert text.rstrip().endswith("--roofline-ceiling on")
    assert "AKLR_BRIEF" not in text
    norm = _by_name(ab.targets(TASKS))["add_rms_norm_hidden_size2880"]
    with pytest.raises(ValueError, match="starting kernel"):
        ab.campaign_script(norm, aklr_root="/a", schema_path="/s", run_dir="/r", hours=1, model="m")


DEPLOYMENT = {
    "model": "gpt_oss_120B",
    "gpu_arch": "mi355x",
    "peak_hbm_gbps": 8000.0,
    "architecture": {"num_layers": 36},
}


def test_the_brief_states_share_bytes_and_floor(tmp_path):
    found = ab.targets(TASKS)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(ab.targets_document(found, deployment=DEPLOYMENT)))
    dep = ab.load_deployment(str(path))
    assert dep == {
        "model": "gpt_oss_120B",
        "gpu_arch": "mi355x",
        "peak_hbm_gbps": 8000.0,
        "num_layers": 36,
    }
    attn_o = _by_name(found)["gemm_n2880_k4096"]
    # m=64: a (64 x 4096) + b (2880 x 4096) + out (64 x 2880), 2 bytes each, over 8 TB/s
    assert ab.bytes_moved(attn_o, 64) == (64 * 4096 + 2880 * 4096 + 64 * 2880) * 2
    text = ab.brief(attn_o, dep)
    assert "| `m=64` | 1000 | 8.00 | 24.49 MB | 3.06 µs |" in text
    assert "36 layers" in text
    script = ab.campaign_script(
        attn_o, aklr_root="/a", schema_path="/s", run_dir="/r", hours=1, model="m", context=text
    )
    lines = script.splitlines()
    assert lines.index("AKLR_BRIEF") > lines.index(
        "  cat >> /r/workspace/program.md <<'AKLR_BRIEF'"
    )
    assert "  git -C /r/workspace commit -qam 'forge_task: where the time goes'" in lines
    norm = _by_name(found)["add_rms_norm_hidden_size2880"]
    assert "layers" not in ab.brief(norm, dep)


def _run_dir(tmp_path, base, best):
    run = tmp_path / "run"
    (run / "workspace" / "forge_experiments").mkdir(parents=True)
    (run / "baseline.json").write_text(json.dumps({"case_times": base}))
    state = {"baseline_case_times": {}, "best_case_times": best}
    (run / "workspace" / "forge_experiments" / "run_state.json").write_text(json.dumps(state))
    return run


def test_campaign_speedups_match_back_to_the_ops(tmp_path):
    qkv = _by_name(ab.targets(TASKS))["gemm_n5120_k2880"]
    run = _run_dir(tmp_path, {"m=64": 0.010, "m=16384": 0.300}, {"m=64": 0.005, "m=16384": 0.400})
    doc = ab.speedups([(qkv, ab.campaign_times(str(run), "m"), "aklr:test")])
    sp = ksp.Speedups.parse(doc)
    row = {"op": "attn_k", "kind": "gemm", "n": 5120, "k": 2880, "m": 64}
    assert sp.factor(row, 20.0) == pytest.approx(2.0)
    # a workload the kernel lost stays as measured
    assert sp.factor({**row, "op": "attn_qkv", "m": 16384}, 300.0) == pytest.approx(0.75)
    # other shapes and ops are untouched
    assert sp.factor({**row, "m": 128}, 20.0) == 1.0
    assert sp.factor({**row, "n": 2880}, 20.0) == 1.0


def test_bench_speedups_credit_only_beating_the_library_too(tmp_path):
    qkv = _by_name(ab.targets(TASKS))["gemm_n5120_k2880"]
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {"cases": {"m=64": {"baseline_ms": 0.012, "library_ms": 0.009, "kernel_ms": 0.006}}}
        )
    )
    (entry,) = ab.speedups([(qkv, ab.bench_times(str(path), "m"), "aklr:test")])["entries"][:1]
    assert entry["baseline_us"] == pytest.approx(9.0) and entry["tuned_us"] == pytest.approx(6.0)


def test_committed_schema_speedups(tmp_path):
    norm = _by_name(ab.targets(TASKS))["add_rms_norm_hidden_size2880"]
    path = tmp_path / "schema.json"
    workloads = [
        {"axes": {"n_tokens": 64}, "accuracy": "passed", "speedup": 1.25},
        {"axes": {"n_tokens": 128}, "accuracy": "failed", "speedup": 3.0},
    ]
    path.write_text(json.dumps({"workloads": workloads}))
    entries = ab.speedups([(norm, ab.committed_times(str(path), "n_tokens"), "aklr:c")])["entries"]
    assert len(entries) == 1
    sp = ksp.Speedups.parse({"entries": entries})
    assert sp.factor(
        {"op": "rmsnorm", "kind": "norm", "width": 2880, "tokens": 64}, 4.0
    ) == pytest.approx(1.25)
