"""CPU-only tests of the kit logic (no GPU, no engine import). Run: python3 -m pytest tests -q"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "sglang_inproc" / "bench"), str(ROOT / "vllm_inproc" / "bench")]

import redact  # noqa: E402
import stats  # noqa: E402
import verify_point  # noqa: E402
import collect  # noqa: E402
import step_bench  # noqa: E402
import engine_step_bench as esb  # noqa: E402
import forward_bench  # noqa: E402

FAKE_HF = "hf_" + "A" * 30
FAKE_GH = "ghp_" + "b" * 36


def test_redact_env_names_and_values():
    env = {"HF_TOKEN": FAKE_HF, "MY_API_KEY": "x", "PATH": "/usr/bin", "NOTE": f"use {FAKE_GH} here",
           "URL": "https://user:pw@example.com/x", "HYPERLOOM_PASSWORD": "p"}
    r = redact.redact_env(env)
    assert r["HF_TOKEN"] == r["MY_API_KEY"] == r["HYPERLOOM_PASSWORD"] == redact.REDACTED
    assert r["PATH"] == "/usr/bin"
    assert FAKE_GH not in r["NOTE"] and redact.REDACTED in r["NOTE"]
    assert "pw" not in r["URL"]


def test_redact_argv_flag_value_and_inline():
    argv = ["--model", "/m", "--hf-token", FAKE_HF, "--api-key=abc", "x=" + FAKE_GH]
    r = redact.redact_argv(argv)
    assert r[:2] == ["--model", "/m"]
    assert r[3] == redact.REDACTED and "abc" not in r[4] and FAKE_GH not in r[5]


def test_percentile_matches_linear():
    xs = [1, 2, 3, 4, 10]
    assert stats.percentile(xs, 50) == 3
    assert stats.percentile(xs, 90) == pytest.approx(7.6)
    s = stats.summarize([2.0, 4.0])
    assert s["median"] == 3.0 and s["n"] == 2


def test_sglang_local_batch_and_owned_flags():
    assert step_bench.local_batch(64, 4) == 16
    with pytest.raises(SystemExit):
        step_bench.local_batch(10, 4)
    with pytest.raises(SystemExit):
        step_bench.check_argv(["--cuda-graph-max-bs-decode", "8"])
    with pytest.raises(SystemExit):
        step_bench.check_argv(["--speculative-algorithm", "DSPARK"])
    step_bench.check_argv(["--tp-size", "4"])


def _so(d):
    return SimpleNamespace(num_scheduled_tokens=d, total_num_scheduled_tokens=sum(d.values()))


def test_vllm_classify_chunked_prefill_then_decode():
    computed = {}
    assert esb.classify(_so({"a": 10, "b": 6}), computed, 10) == (2, 2, 16, 0)
    assert esb.classify(_so({"a": 1, "b": 4}), computed, 10) == (2, 1, 4, 1)
    assert esb.classify(_so({"a": 6, "b": 6}), computed, 10) == (2, 0, 0, 12)


def test_vllm_steady_window():
    rows = [{"n_prefill_reqs": 1, "n_reqs": 2, "out_tokens": 1}, {"n_prefill_reqs": 0, "n_reqs": 2, "out_tokens": 7},
            {"n_prefill_reqs": 0, "n_reqs": 1, "out_tokens": 3}, {"n_prefill_reqs": 0, "n_reqs": 2, "out_tokens": 0}]
    assert esb.steady_window(rows, 2) == [rows[1]]


def test_forward_case_file(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# x\nverify=4*6@1024:5  # tail\n\nprefill=128\n")
    assert forward_bench.read_case_file(str(f)) == ["verify=4*6@1024:5", "prefill=128"]


def test_presets_case_file_parses_with_upstream_grammar():
    import re
    rx = re.compile(r"^(?:(\d+)\*)?(\d+)(?:@(\d+))?(c)?(?::(\d+))?$")  # copy of vllm shapes._REQ_RE
    for line in forward_bench.read_case_file(str(ROOT / "vllm_inproc/presets/dsv41_agentx_cases.txt")):
        _, body = line.split("=", 1)
        for item in body.split(","):
            assert rx.match(item.strip()), item


def _mk_point(d, result, exit_code=0, prov=True):
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(result))
    (d / "launch_status.json").write_text(json.dumps({"exit_code": exit_code}))
    if prov:
        for f in verify_point.PROV:
            (d / f).write_text("{}")


def test_verify_step_bench_pass_and_fail(tmp_path):
    good = {"tool": "step_bench", "engine": "sglang", "complete": True, "output_len": 5,
            "decode_step_s": {"n": 4}, "tpot_ms": 10.0, "decode_tok_s": 1600.0, "batch_size_global": 16,
            "input_len": 8, "num_gpus": 4, "prefill_tok_s": 1.0, "decode_tok_s_per_gpu": 400.0}
    _mk_point(tmp_path / "ok", good)
    assert verify_point.main([str(tmp_path / "ok")]) == 0
    _mk_point(tmp_path / "bad", {**good, "decode_step_s": {"n": 3}}, exit_code=1)
    assert verify_point.main([str(tmp_path / "bad")]) == 1
    _mk_point(tmp_path / "leak", good)
    (tmp_path / "leak" / "env.json").write_text(json.dumps({"X": FAKE_HF}))
    assert verify_point.main([str(tmp_path / "leak")]) == 1
    assert collect.main([str(tmp_path), "--output", str(tmp_path / "s.csv")]) == 1
    assert "ok" in (tmp_path / "s.csv").read_text()


def test_expand_args_shell():
    out = subprocess.run(["bash", "-c", f"source {ROOT}/common/lib.sh; kit_expand_args {ROOT}/vllm_inproc/presets/dsv41_dspark_synth351.args"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0] == "--speculative-config" and out[1].startswith('{"method":"dspark"')


def test_no_secrets_in_repo_files():
    pats = redact._TOKEN_VALUE_RES
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".sh", ".md", ".args", ".env", ".txt"} and "tests" not in p.parts:
            t = p.read_text(errors="replace")
            assert not any(r.search(t) for r in pats), p


class _FakeSched:
    def __init__(self):
        self.calls = []

    def update_from_output(self, so, mo):
        self.calls.append(so)


class _FakeEngine:
    """Batch-queue engine: step k schedules batch k and completes batch k-1 (step 0 completes nothing)."""

    def __init__(self, sched, batches):
        self.sched, self.batches, self.k = sched, batches, 0

    def add_request(self, *a):
        pass

    def has_unfinished_requests(self):
        return self.k <= len(self.batches)

    def step(self):
        if self.k >= 1:
            self.sched.update_from_output(self.batches[self.k - 1], None)
        self.k += 1
        return []


def test_vllm_run_attributes_by_completion_under_batch_queue():
    s = _FakeSched()
    batches = [_so({"a": 4, "b": 4}), _so({"a": 6, "b": 6}), _so({"a": 6, "b": 6})]
    eng = _FakeEngine(s, batches)
    rows, _ = esb.run(eng, s, [("a", None, None), ("b", None, None)], isl=4, timed=False)
    assert [r["n_prefill_reqs"] for r in rows] == [2, 0, 0]
    assert rows[0]["carried_s"] > 0 and rows[1]["carried_s"] == 0
    assert [r["decode_tokens"] for r in rows] == [0, 12, 12]
    assert s.update_from_output.__name__ == "update_from_output"  # restored


def test_redact_docker_inspect_env_and_cmd():
    obj = [{"Config": {"Env": ["HF_TOKEN=" + FAKE_HF, "PATH=/bin", "NOTE=x " + FAKE_GH], "Cmd": ["--api-key", "abc"]},
            "Args": ["--hf-token=" + FAKE_HF]}]
    r = redact.redact_docker_inspect(obj)[0]
    s = json.dumps(r)
    assert FAKE_HF not in s and FAKE_GH not in s and "abc" not in s
    assert "PATH=/bin" in r["Config"]["Env"]


def test_verify_scans_config_and_inspect(tmp_path):
    good = {"tool": "infer", "mode": "throughput", "complete": True}
    _mk_point(tmp_path / "p", good)
    (tmp_path / "p" / "container-inspect.json").write_text(json.dumps({"Env": ["T=" + FAKE_GH]}))
    assert verify_point.main([str(tmp_path / "p")]) == 1
