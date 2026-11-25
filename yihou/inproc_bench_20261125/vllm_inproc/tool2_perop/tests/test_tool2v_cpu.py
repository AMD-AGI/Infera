"""CPU-only tests of vLLM tool 2's pure logic (no GPU, no vllm import).
Run: python3 -m pytest vllm_inproc/tool2_perop/tests -q"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

T2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(T2), str(T2 / "dsv41")]

import kernel_breakdown  # noqa: E402
import model_bench as mb  # noqa: E402
import shape_export  # noqa: E402
import vllm_bench as vb  # noqa: E402


def test_tree_structure():
    assert mb.RESIDUALS["layer"] == ["engram", "input_layernorm", "mhc_attn", "mhc_post", "attn",
                                     "post_attention_layernorm", "mhc_ffn", "mlp"]
    assert mb.RESIDUALS["mlp"] == ["mlp.experts"]
    assert mb.RESIDUALS["mlp.experts"] == ["mlp.experts.gate", "mlp.experts.shared_experts"]
    assert "attn.wqkv_a" in mb.RESIDUALS["attn"] and "attn.<rest>" not in mb.RESIDUALS["attn"]
    for p in mb.PARENTS:
        assert p not in mb.LEAVES
    assert {"attn.<rest>", "mlp.<rest>", "mlp.experts.<rest>", "layer.<rest>"} <= set(mb.LEAVES)


def _full_kind_times(kind="k"):
    t = {f"{kind}|layer": 20.0}
    for c in mb.RESIDUALS["layer"]:
        t[f"{kind}|{c}"] = 1.0
    t[f"{kind}|attn"] = 8.0
    for c in mb.RESIDUALS["attn"]:
        t[f"{kind}|{c}"] = 0.3
    t[f"{kind}|mlp"] = 6.0
    t[f"{kind}|mlp.experts"] = 5.0
    t[f"{kind}|mlp.experts.gate"] = 0.2
    t[f"{kind}|mlp.experts.shared_experts"] = 0.7
    return t


def test_leaves_partition_a_layer_exactly():
    times = _full_kind_times()
    times["k|engram_lookup"] = 0.4
    times.update({"embed_tokens": 0.1, "hc_post": 0.05, "logits": 0.3})
    mb.add_residuals(times, {"k": 3})
    assert times["k|mlp.experts.<rest>"] == pytest.approx(5.0 - 0.9)
    assert times["k|mlp.<rest>"] == pytest.approx(1.0)
    totals = mb.leaf_totals(times, {"k": 3}, {"hc_post": 3})
    assert sum(totals.values()) == pytest.approx(3 * (20.0 + 0.4) + 0.1 + 3 * 0.05 + 0.3)
    assert totals["hc_post"] == pytest.approx(0.15)


def test_child_without_parent_not_summed():
    times = {"k|mlp.experts.gate": 1.0, "k|attn.wq_b": 2.0, "k|attn": 3.0}
    totals = mb.leaf_totals(times, {"k": 1})
    assert "mlp.experts.gate" not in totals and totals["attn.wq_b"] == 2.0


def test_residual_owner():
    assert mb.residual_owner("k|mlp.experts.gate") == "k|mlp.experts.<rest>"
    assert mb.residual_owner("k|attn.wo_b") == "k|attn.<rest>"
    assert mb.residual_owner("k|mhc_attn") == "k|layer.<rest>"
    assert mb.residual_owner("k|engram_lookup") == ""
    assert mb.residual_owner("draft|attn") == ""
    note = mb.skipped_note({"k|attn.mla": "RuntimeError: a|b", "logits": "x"})
    assert "`attn.<rest>`" in note and "nowhere" in note and "a\\|b" in note
    cov = mb.coverage_note(["k|input_layernorm", "norm", "k|attn"], {"k|attn": 1})
    assert "input_layernorm" in cov and "unattributed" in cov


def _layer(ratio, comp=False, idx=False, engram=False, hash_=False):
    attn = SimpleNamespace(compress_ratio=ratio, compressor=object() if comp else None, indexer=object() if idx else None)
    ffn = SimpleNamespace(experts=object(), gate=SimpleNamespace(tid2eid=object() if hash_ else None))
    return SimpleNamespace(attn=attn, ffn=ffn, engram=object() if engram else None)


def test_layer_kind_names_match_sglang_port():
    assert mb.layer_kind(_layer(0), 0, 20) == "moe+c0"
    assert mb.layer_kind(_layer(0, engram=True), 1, 20) == "moe+c0+engram"
    assert mb.layer_kind(_layer(2, True, True), 2, 20) == "moe+c2+kvsrc+indexer"
    assert mb.layer_kind(_layer(1, False, True), 24, 20) == "moe+c1+indexer+late"
    assert mb.layer_kind(_layer(1), 25, None) == "moe+c1"
    assert mb.layer_kind(_layer(0, hash_=True), 0, None) == "moe+hash+c0"


def test_ids_and_rotation():
    assert mb._ids([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    MB = 1 << 20
    assert mb.rotation_plan(300 * MB) == 0
    assert mb.rotation_plan(200 * MB) == 2
    assert mb.rotation_plan(9 * MB) == 56
    assert mb.rotation_plan(1 * MB) == 0  # 256 slots x 1 MB fit the MALL: AKLR does not rotate


def test_aklr_draw_and_fixture_logits():
    a = vb.aklr_draw(64, 384, 6, "cpu")
    b = vb.aklr_draw(64, 384, 6, "cpu")
    assert torch.equal(a, b) and a.shape == (64, 6)
    assert all(len(set(r.tolist())) == 6 for r in a)
    lg = vb.fixture_logits(a, 384, torch.float32)
    top = lg.topk(6, dim=1).indices.sort(1).values
    assert torch.equal(top, a.sort(1).values)
    # sqrt(softplus) is monotonic: same top-k after the scoring function
    sc = torch.sqrt(torch.nn.functional.softplus(lg))
    assert torch.equal(sc.topk(6, dim=1).indices.sort(1).values, a.sort(1).values)


class _Gate(torch.nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, 8), requires_grad=False)
        self.e_score_correction_bias = torch.nn.Parameter(torch.ones(n), requires_grad=False)

    def forward(self, x):
        return x @ self.weight.T, None


class _Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.n_activated_experts = 4


def test_balanced_routing_hook_and_restore():
    moe = _Moe()
    br = vb.BalancedRouting([(moe, moe.gate)])
    assert float(moe.gate.e_score_correction_bias.sum()) == 0.0
    out, _ = moe.gate(torch.randn(5, 8))
    assert br.fired == [1]
    assert torch.equal(out.topk(4, 1).indices.sort(1).values, vb.aklr_draw(5, 16, 4, "cpu").sort(1).values)
    out2, _ = moe.gate(torch.randn(5, 8))
    assert out2 is out  # cached per row count
    br.remove()
    assert float(moe.gate.e_score_correction_bias.sum()) == 16.0
    out3, _ = moe.gate(torch.zeros(5, 8))
    assert float(out3.abs().sum()) == 0.0


@dataclass
class _QA:
    data: torch.Tensor
    scale: torch.Tensor


def test_snapshot_versions_and_dataclass():
    memo = {}
    t = torch.ones(3)
    a = vb.snapshot((t, {"x": t}), memo)
    assert a[0] is a[1]["x"] and a[0] is not t
    t.add_(1)
    b = vb.snapshot(t, memo)
    assert b is not a[0] and float(b[0]) == 2.0 and float(a[0][0]) == 1.0
    q = vb.snapshot(_QA(torch.zeros(2), torch.ones(1)), memo)
    assert isinstance(q, _QA)


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq_b = torch.nn.Linear(4, 4, bias=False)

    def _wq_b_proj(self, x):
        return self.wq_b(x)

    def forward(self, x):
        return self._wq_b_proj(x) * 2


def test_record_and_replay_module_and_call_site():
    attn = _Attn()
    ops = [mb.Op("k|attn", module=attn), mb.Op("k|attn.wq_b", site=(attn, "_wq_b_proj"), wmod=attn.wq_b),
           mb.Op("k|never", module=torch.nn.Linear(1, 1))]
    saved, calls = {}, {}
    x = torch.randn(3, 4)
    with mb.record_inputs(ops, saved, calls):
        ref = attn(x)
        attn(x)
    assert list(saved) == ["k|attn", "k|attn.wq_b"] and calls == {"k|attn": 2, "k|attn.wq_b": 2}
    assert "_wq_b_proj" not in attn.__dict__  # call site restored
    assert torch.allclose(mb.make_call(saved["k|attn"])(), ref)
    assert torch.allclose(mb.make_call(saved["k|attn.wq_b"])(), ref / 2)
    assert "never" in mb.coverage_note(mb.op_names(ops), saved)


def test_shared_function_site_with_match():
    ns = SimpleNamespace(fn=lambda tag, x: x + tag)
    ops = [mb.Op("a|row", site=(ns, "fn"), match=lambda a, kw: a[0] == 1),
           mb.Op("b|row", site=(ns, "fn"), match=lambda a, kw: a[0] == 2)]
    saved = {}
    with mb.record_inputs(ops, saved):
        ns.fn(2, torch.zeros(1))
        ns.fn(1, torch.zeros(1))
        ns.fn(3, torch.zeros(1))
    assert set(saved) == {"a|row", "b|row"} and saved["a|row"].args[0] == 1
    assert float(mb.make_call(saved["b|row"])()[0]) == 2.0


def test_plan_prefill():
    sys.path.insert(0, str(T2 / "dsv41"))
    import profile_prefill as pp

    args = SimpleNamespace(ISL=40000, prefix_cache_hit=0.5)
    steps = pp.plan_prefill(args, 16384, 128)
    assert [(s.prefix, s.extend) for s in steps] == [(19968, 16384), (36352, 3648)]
    with pytest.raises(ValueError):
        pp.plan_prefill(SimpleNamespace(ISL=128, prefix_cache_hit=1.0), 16384, 128)


def test_kernel_totals_multiplicity():
    times = {"k|attn": 1.0, "k|attn.wq_b": 0.2, "hc_post": 0.1}
    prof = {"k|attn.wq_b": {("gemm", "api"): (0.2, 1, ())}, "hc_post": {("post", "api"): (0.1, 1, ())}}
    tot = mb.kernel_totals(prof, {"k": 5}, times, {"hc_post": 3})
    assert tot["attn.wq_b"][("gemm", "api")][0] == pytest.approx(1.0)
    assert tot["hc_post"][("post", "api")][1] == 3


def test_kernel_breakdown_vllm_marks():
    assert "vllm" in kernel_breakdown._PACKAGES
    row = kernel_breakdown.TimedRow.of(1.0, {("ncclDevKernel_Generic_1", "x"): (0.5, 1, ()), ("gemm", "y"): (0.5, 1, ())})
    assert row.collective_ms == 0.5
    api = kernel_breakdown.python_api({"name": "/usr/local/lib/python3.12/dist-packages/vllm/models/x.py(12): f"})
    assert api == "vllm.models.x.f:12"


def test_shape_export(tmp_path):
    rec = shape_export.Recorder(str(tmp_path), "t", rows_now=lambda: ["k|attn"])
    lin = torch.nn.Linear(8, 4, bias=False)
    rec.module_call("k|x", lin, (torch.zeros(2, 8),), {})
    rec.module_call("k|x", lin, (torch.zeros(2, 8),), {})
    rec._api("aiter.gemm", "vllm.x", (torch.zeros(2, 8, dtype=torch.bfloat16),), {"n": 3})
    p = rec.save()
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert lines[0]["count"] == 2 and lines[0]["weights"]["weight"]["shape"] == [4, 8]
    assert lines[1]["rows"] == ["k|attn"] and lines[1]["args"][0]["dtype"] == "bfloat16"
    assert shape_export.gemm_dims(1338, torch.zeros(1792, 5120)) == {"M": 1338, "N": 1792, "K": 5120, "weight_dtype": "float32"}
    out = shape_export.write_shapes(tmp_path / "s.json", {"phase": "decode"}, [{"row": "k|x"}])
    assert json.loads(out.read_text())["schema"] == "tool2-operator-shapes/1"


def test_run_script_dry_run():
    sh = T2 / "scripts" / "run_t2v.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    out = subprocess.run(["bash", str(sh), "p1", "decode", "--ISL", "10"], capture_output=True, text=True,
                         env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin", "NPROC": "2",
                              "ARGS_FILES": str(T2 / "presets" / "dsv41_dp2.args")})
    assert out.returncode == 0, out.stderr
    assert "torchrun" in out.stdout and "profile_decode.py" in out.stdout and "--data-parallel-size" in out.stdout
    assert "/out/iterations/p1" in out.stdout


def test_no_secrets_in_tool():
    import re

    pat = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
    for p in T2.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".sh", ".md", ".args"):
            assert not pat.search(p.read_text(errors="ignore")), p


def test_read_isl_file(tmp_path):
    f = tmp_path / "isl.txt"
    f.write_text("# header\n100\n\n200  # c\n")
    assert mb.read_isl_file(str(f)) == [100, 200]


def test_snapshot_inference_tensors():
    with torch.inference_mode():
        t = torch.ones(4)
        memo = {}
        a = vb.snapshot(t, memo)
        assert vb.snapshot(t, memo) is a  # unchanged: one clone
        t.add_(1)
        b = vb.snapshot(t, memo)
        assert b is not a and float(b[0]) == 2.0 and float(a[0]) == 1.0
