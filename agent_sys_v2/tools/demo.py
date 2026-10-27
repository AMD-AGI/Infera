"""Make a demo AGENT_SYS_HOME: one infera-perf project in the middle of its second round, with three machines: round
one done (the template's five tasks, two experiments, the close that sealed v1), round two's intake and plan done, its
experiments in most states (one still a draft), its close waiting for them (a children dependency), and the ledger's
curves (v0, v1, two InferenceX references, the experiments' working-point points, their base arms as repeats), so the
console can be looked at without running anything.

  python tools/demo.py <home>          # a new directory, or a demo home made before (it is replaced)
  AGENT_SYS_TMUX=asys-demo agent-sys-v2 serve --home <home> --sock /tmp/agent_sys-$(id -u)/demo.sock

The numbers are made up, but our points are converted from AgentX output (the real agentx_conc40.json of the tests,
rescaled, beside an AIPerf export with its window and node, measured one after another on the task's machine), GSM8K
from lm_eval samples, and the A/B verdicts written by ab_judge.py, as a task would; two verdicts are spoiled on purpose
(one edited by hand, one from before verdicts had inputs) and show as unverified, round one's experiment was judged
against v0, and one point is an AgentX fast run (screening).
Every run of the project replays a recorded session (tests/fixtures/events) instead of calling a model: starting a
task here costs nothing.
"""

import contextlib
import copy
import importlib.util
import io
import json
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from agent_sys_v2 import ledger, ops, skills
from agent_sys_v2.model import new_state, write_state

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "tests" / "fixtures" / "events"
RAW = ROOT / "tests" / "fixtures" / "agentx" / "agentx_conc40.json"
AIPERF = ROOT / "tests" / "fixtures" / "aiperf" / "profile_export_aiperf.json"
MARK = ".agent_sys_demo"  # in a home this script made: only such a home is replaced
DONE = (
    "状态: 已完成\n下一步: 无\n完成时间: 2026-10-21 10:00\n\n- [x] 部署\n- [x] 质量门\n- [x] 测量\n"
)


def ab_judge():
    spec = importlib.util.spec_from_file_location(
        "ab_judge", skills.SHIPPED / "infera-experiment/scripts/ab_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agentx(conc, x, y):
    """An agentx_conc<C>.json: the tests' real one at another concurrency, input tok/s/GPU y, P90 interactivity x."""
    agg = copy.deepcopy(json.loads(RAW.read_text(encoding="utf-8")))
    agg["conc"] = conc
    per_gpu = agg["request_metrics"]["throughput"]["per_gpu"]
    scale = y / per_gpu["input_tput_tps"]
    for k in ("input_tput_tps", "total_tput_tps", "output_tput_tps"):
        per_gpu[k] = round(per_gpu[k] * scale, 5)
    agg["request_metrics"]["latency"]["full_response_itl"]["p90"] = round(1 / x, 6)
    return agg


def aiperf(conc, start, node, duration=3600):
    """AIPerf's export of a point: the tests' real one with its phase, window and endpoint set."""
    export = json.loads(AIPERF.read_text(encoding="utf-8"))
    export["input_config"]["phases"][0].update(concurrency=conc, duration=float(duration))
    export["input_config"]["endpoint"]["urls"] = [f"http://{node}:18000"]
    export["start_time"] = start.isoformat()
    export["end_time"] = (start + timedelta(seconds=duration + 26)).isoformat()
    return export


def lm_eval(d, total, wrong):
    """An lm_eval GSM8K run (--log_samples) under d: questions 0..total-1, every wrong-th answered wrong."""
    d.mkdir(parents=True)
    rows = [
        json.dumps({"doc_id": i, "filter": f, "exact_match": 0.0 if i % wrong == 3 else 1.0})
        for i in range(total)
        for f in ("strict-match", "flexible-extract")
    ]
    (d / "samples_gsm8k_2026-10-21T00-00-00.jsonl").write_text("\n".join(rows) + "\n")
    return d


def main(home):
    home = Path(home)
    if home.exists() and any(home.iterdir()) and not (home / MARK).exists():
        raise SystemExit(
            f"{home} is not empty and not a demo home made before: give a new directory"
        )
    shutil.rmtree(home, ignore_errors=True)
    ops.init_home(home)
    (home / MARK).write_text("made by tools/demo.py, which replaces this home when it runs again\n")
    pdir = ops.new_project(home, "glm52-agentx", "infera-perf", title="GLM-5.2 4P4D AgentX")
    ops.start_project(pdir)  # the template's tasks reviewed: round one is under way
    cfg = (pdir / "project.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace('goal: ""', 'goal: "P90 interactivity ≥ 40 下的输入 tok/s/GPU +10%"')
    cfg = cfg.replace(
        "machines: []",
        "machines:\n"
        + "".join(
            f"  - {{name: smc-a16-{n}, gpus: 8, note: MI355X}}\n" for n in (19, 20, 21)
        ).rstrip(),
    )
    cfg = cfg.replace("working_point: {}", "working_point: {x: 40}")
    cfg += f"  replay: {{events: {EVENTS / 'message_mid_command.jsonl'}, speed: 5}}   # demo: runs replay, no model\n"
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")

    def task(tid, body=None, fm="", state=None, progress=None, report=None, results=()):
        """body None: the prompt the template brought stays."""
        path = pdir / "tasks" / f"{tid}.md"
        if body is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"---\n{fm}---\n{body}\n" if fm else f"{body}\n", encoding="utf-8")
        ws = pdir / "workspaces" / tid
        if state:
            write_state(ws, state)
        if progress:
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "progress.md").write_text(progress, encoding="utf-8")
        if report:
            (ws / "report.md").write_text(report, encoding="utf-8")
        for name, doc in results:
            (ws / "results").mkdir(parents=True, exist_ok=True)
            (ws / "results" / f"{name}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=1)
            )

    def st(status, since, detail=None, **more):
        s = new_state(status, detail)
        s.update(since=since, created="2026-10-20T01:00:00Z", **more)
        s["started"] = since if status != "queued" else ""
        return s

    def res(label, conc, x, y, t, series, role="candidate", round_=1, verdict=None, change=""):
        """A result written by hand: a reference, or a verdict from before verdicts had inputs."""
        doc = {
            "schema": "infera.result/v1",
            "label": label,
            "time": t,
            "role": role,
            "round": round_,
            "series": series,
            "change": change,
            "metrics": {"input_tput_per_gpu": y, "p90_intvty": x},
            "workload": {"client": "agentx", "conc": conc, "duration_s": 3600},
        }
        if verdict:
            doc["verdict"] = verdict
        return doc

    slots = {}

    def measure(
        tid, label, conc, x, y, when, series, role="candidate", round_=1, change="", fast=False
    ):
        """An AgentX run directory under the task's log/, converted into its results/ as a task does: measured on
        the task's machine, each window 85 min after the task's one before (redeploy and warm-up between)."""
        ws = pdir / "workspaces" / tid
        run = ws / "log" / "agentx" / f"{label}-c{conc}"
        (run / "aiperf_artifacts").mkdir(parents=True)
        (run / f"agentx_conc{conc}.json").write_text(json.dumps(agentx(conc, x, y), indent=1))
        (run / "runtime.env").write_text(
            f"CONC={conc}\nDURATION=3600\nAIPERF_WARMUP_REQUESTS_PER_LANE=10\n"
            + ("AIPERF_EXPERIMENTAL_FAST=1\n" if fast else "")
        )
        k = slots[tid] = slots.get(tid, -1) + 1
        start = datetime.fromisoformat(when.rstrip("Z")) + timedelta(minutes=85 * k)
        node = re.search(r"machines: \[([^\]]+)\]", (pdir / "tasks" / f"{tid}.md").read_text())
        (run / "aiperf_artifacts/profile_export_aiperf.json").write_text(
            json.dumps(
                aiperf(conc, start, node[1] if node else "smc-a16-19", 1200 if fast else 3600)
            )
        )
        written, errors = ledger.convert_agentx(
            [run], ws / "results", series, round_, role, label, change, when
        )
        assert written and not errors, errors
        return written[0]

    def gsm8k(tid, label, series, total, wrong):
        """An lm_eval GSM8K run under the task's log/, converted with ledger gsm8k as a task does."""
        ws = pdir / "workspaces" / tid
        run = lm_eval(ws / "log" / "gsm8k" / label, total, wrong)
        argv = [
            "gsm8k",
            str(run),
            "--out",
            str(ws / "results"),
            "--label",
            label,
            "--series",
            series,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            assert ledger.main(argv) == 0

    judge = ab_judge()

    def ab(base, cand, *more):
        with contextlib.redirect_stdout(io.StringIO()):
            argv = ["--baseline-files", *map(str, base), "--candidate-files", *map(str, cand)]
            assert judge.main([*argv, *more, "--write"]) == 0

    seen = {"seen_at": "2026-10-23T00:00:00Z"}

    def done(tid, when, cost, report, results=(), body=None, fm="", **more):
        task(
            tid,
            body,
            fm,
            state=st("done", when, finished=when, cost_usd=cost, **seen, **more),
            progress=DONE,
            report=report,
            results=results,
        )

    # ---------------------------------------------------------------- round one, done
    done(
        "intake/r1",
        "2026-10-20T02:00:00Z",
        3.1,
        "## 结论\n\nGLM-5.2 MXFP4，ATOM 4P4D，AgentX；工作点 P90 interactivity ≥ 40。\n",
    )
    done(
        "survey/r1",
        "2026-10-20T04:00:00Z",
        12.5,
        "## 结论\n\nIX 上最好的 AMD 条目在工作点领先 v0 约 20%；配置差异 6 项。\n",
        [
            (
                f"ix-atom-c{c}",
                res(
                    f"ix-atom-c{c}", c, x, y, "2026-10-20T03:30:00Z", "ix:atom-mi355x", "reference"
                ),
            )
            for c, x, y in (
                (16, 78.0, 11500.0),
                (32, 66.0, 18900.0),
                (64, 49.0, 26300.0),
                (96, 38.0, 29800.0),
            )
        ]
        + [
            (
                f"ix-b200-c{c}",
                res(
                    f"ix-b200-c{c}", c, x, y, "2026-10-20T03:40:00Z", "ix:sglang-b200", "reference"
                ),
            )
            for c, x, y in ((16, 95.0, 14200.0), (32, 84.0, 23500.0), (64, 66.0, 33800.0))
        ],
    )
    done(
        "baseline/r1",
        "2026-10-21T06:00:00Z",
        41.2,
        "## 结论\n\nv0：工作点（X ≥ 40）输入 tok/s/GPU 23.7K，噪声 ±0.5%。\n",
    )
    for tid in ("baseline/r1", "profile/r1"):
        ops.update_task(pdir, tid, {"machines": ["smc-a16-19"]})
    for i, (c, x, y) in enumerate(
        (
            (16, 72.0, 9800.0),
            (32, 61.0, 16200.0),
            (48, 52.0, 20100.0),
            (64, 44.0, 22800.0),
            (96, 33.0, 25400.0),
        )
    ):
        measure("baseline/r1", "v0", c, x, y, f"2026-10-21T0{i + 1}:00:00Z", "v0", "baseline")
    measure("baseline/r1", "v0-rep1", 64, 44.2, 22700.0, "2026-10-21T05:40:00Z", "v0", "baseline")
    gsm8k("baseline/r1", "v0-full", "v0", 1319, 25)
    done(
        "profile/r1",
        "2026-10-21T14:00:00Z",
        96.0,
        "## 结论\n\nprefill 饱和（FCFS 队头阻塞）[M]；decode 最忙 rank 决定步长 [M]；API 进程 gen-2 GC [I]。\n",
    )
    done(
        "plan/r1",
        "2026-10-21T16:00:00Z",
        18.0,
        "## 结论\n\n2 个实验，2 台节点；本轮 ≤ 40 节点·小时。\n",
    )
    expect = "expect: {{delta_pct: {d}, level: L4, falsify: 工作点 ΔY < 3%}}\n"

    def experiment(name, title, machine, plan, delta, state, report, change):
        task(
            f"experiment/{name}",
            f"## 任务描述\n\n- 假设：{change}\n",
            f"title: {title}\nparent: {plan}\nmachines: [{machine}]\n" + expect.format(d=delta),
            state,
            DONE if report else None,
            report,
        )

    # SPF: judged against v0 by ab_judge, with two base arms; still holds, and says it was against v0
    when = "2026-10-22T02:00:00Z"
    experiment(
        "r1-spf",
        "SPF 调度",
        "smc-a16-19",
        "plan/r1",
        [8, 14],
        st("done", when, finished=when, cost_usd=30.0, **seen),
        "## 结论\n\naccept：C64 +7.3%（相对两次 v0 复测）。\n",
        "--schedule-policy shortest-prefill-first",
    )
    tid = "experiment/r1-spf"
    base = [
        measure(tid, f"r1-spf-base{k}", 64, x, y, when, "v0", "baseline")
        for k, x, y in ((1, 44.1, 22750.0), (2, 44.0, 22850.0))
    ]
    change = "--schedule-policy shortest-prefill-first"
    cand = [
        measure(tid, f"r1-spf-{k}", 64, x, y, when, "r1-spf", change=change)
        for k, x, y in ((1, 44.5, 24400.0), (2, 44.6, 24500.0))
    ]
    gsm8k(tid, "r1-spf-200", "r1-spf", 200, 25)
    ab(base, cand, "--mechanism", "prefill 排队 -80%，decode 步长不变")
    # DCP4: a verdict from before verdicts had inputs (written by hand): shown as it is, unverified
    when = "2026-10-22T05:00:00Z"
    experiment(
        "r1-dcp4",
        "DCP4 decode",
        "smc-a16-21",
        "plan/r1",
        [3, 6],
        st("done", when, finished=when, cost_usd=31.0, **seen),
        "## 结论\n\nreject。\n",
        "decode DCP4",
    )
    task(
        "experiment/r1-dcp4",
        results=[
            (
                "r1-dcp4-c64",
                res(
                    "r1-dcp4-c64",
                    64,
                    39.0,
                    21000.0,
                    when,
                    "r1-dcp4",
                    change="decode DCP4",
                    verdict={"status": "valid", "decision": "reject", "vs_baseline_pct": -7.9},
                ),
            )
        ],
    )
    # close/r1 sealed v1: its curve is v1's reference
    done(
        "close/r1",
        "2026-10-22T20:00:00Z",
        52.0,
        "## 结论\n\n封版 v1（SPF）：工作点 +9.8%。\n",
        body="## 任务描述\n\n- 第 1 轮收尾：集成、封版、归档、复盘，起草下一轮的 intake。\n",
        fm="title: 第 1 轮收尾\ndepends_on: [{task: plan/r1, kind: children}]\nmachines: [smc-a16-19]\n",
    )
    for i, (c, x, y) in enumerate(
        (
            (16, 74.0, 10900.0),
            (32, 63.0, 17600.0),
            (48, 54.0, 21800.0),
            (64, 46.0, 24600.0),
            (96, 35.0, 27200.0),
        )
    ):
        measure("close/r1", "v1", c, x, y, f"2026-10-22T1{i + 4}:00:00Z", "v1", "baseline")
    gsm8k("close/r1", "v1-full", "v1", 1319, 26)
    (pdir / "workspaces/close/r1/stack.json").write_text('{"version": "v1", "sealed": true}\n')
    # round one sealed v1: what close/r1 writes
    cfg = re.sub(r"(?m)^round: 1", "round: 2", cfg)
    cfg = re.sub(r"(?m)^version: v0", "version: v1", cfg)
    cfg = cfg.replace("baseline_series: v0", "baseline_series: v1")
    (pdir / "project.yaml").write_text(cfg, encoding="utf-8")

    # ---------------------------------------------------------------- round two, under way
    done(
        "intake/r2",
        "2026-10-23T01:00:00Z",
        2.4,
        "## 结论\n\n第 2 轮：目标不变，基线 v1。\n",
        body="## 任务描述\n\n- 第 2 轮开始：按问卷 rN 更新项目规则。\n",
        fm="title: 第 2 轮开始\ndepends_on: [close/r1]\n",
    )
    done(
        "plan/r2",
        "2026-10-23T02:00:00Z",
        9.0,
        "## 结论\n\n4 个实验，3 台节点。\n",
        body="## 任务描述\n\n- 第 2 轮计划。\n",
        fm="title: 第 2 轮计划\ndepends_on: [intake/r2]\n",
    )
    # KV-aware routing: accepted against two v1 repeats
    when = "2026-10-23T08:00:00Z"
    experiment(
        "r2-kvroute",
        "KV 感知路由",
        "smc-a16-19",
        "plan/r2",
        [5, 10],
        st("done", when, finished=when, cost_usd=33.0, **seen),
        "## 结论\n\naccept：C64 +11%（相对两次 v1 复测）。\n",
        "router: kv-aware",
    )
    tid = "experiment/r2-kvroute"
    base = [
        measure(tid, f"r2-kvroute-base{k}", 64, x, y, when, "v1", "baseline", 2)
        for k, x, y in ((1, 46.0, 24550.0), (2, 46.1, 24700.0))
    ]
    cand = [
        measure(
            tid,
            f"r2-kvroute-{k}",
            64,
            x,
            y,
            when,
            "r2-kvroute",
            round_=2,
            change="router: kv-aware",
        )
        for k, x, y in ((1, 46.3, 27200.0), (2, 46.2, 27400.0))
    ]
    # a screening point: AgentX fast mode, in the ledger, no input of the A/B
    measure(tid, "r2-kvroute-fast", 64, 46.5, 27900.0, when, "r2-kvroute", round_=2, fast=True)
    gsm8k(tid, "r2-kvroute-200", "r2-kvroute", 200, 25)
    ab(base, cand, "--mechanism", "prefix 命中率 41% → 68%")
    # tokenizer workers: judged noise, then the decision edited by hand: unverified
    when = "2026-10-23T11:00:00Z"
    experiment(
        "r2-tokenizer",
        "tokenizer workers ×4",
        "smc-a16-20",
        "plan/r2",
        [1, 3],
        st("in_review", when, {"kind": "review"}, finished=when, cost_usd=31.0),
        "## 结论\n\naccept。\n",
        "--tokenizer-worker-num 4",
    )
    tid = "experiment/r2-tokenizer"
    base = [measure(tid, "r2-tokenizer-base", 64, 46.0, 24600.0, when, "v1", "baseline", 2)]
    cand = [
        measure(
            tid,
            "r2-tokenizer",
            64,
            46.2,
            25000.0,
            when,
            "r2-tokenizer",
            round_=2,
            change="--tokenizer-worker-num 4",
        )
    ]
    ab(base, cand)
    spoiled = json.loads(cand[0].read_text(encoding="utf-8"))
    spoiled["verdict"]["decision"] = "accept"
    cand[0].write_text(json.dumps(spoiled, ensure_ascii=False, indent=1))
    # HiCache: voided (other, with its evidence); its checks failed on something else
    tid = "experiment/r2-hicache"
    task(
        tid,
        "## 任务描述\n\n- 假设：HiCache write-back\n",
        "title: HiCache write-back\nparent: plan/r2\nmachines: [smc-a16-21]\n"
        + expect.format(d=[4, 8]),
        st(
            "needs_you",
            "2026-10-23T13:00:00Z",
            {"kind": "checks_failed", "failed": ["results"], "passed": 1, "total": 2},
            cost_usd=22.4,
        ),
        "状态: 已完成\n下一步: 无\n",
    )
    void = measure(
        tid, "r2-hicache", 64, 45.0, 27100.0, "2026-10-23T12:30:00Z", "r2-hicache", round_=2
    )
    log = pdir / "workspaces" / tid / "log/rocm-smi.txt"
    log.write_text("GPU3  98%  pid 41871 (not ours: someone else's sweep)\n")
    with contextlib.redirect_stdout(io.StringIO()):
        argv = ["--void", "other", "--reason", "测量窗口内有别人的 sweep 跑在 GPU 3"]
        assert (
            judge.main([*argv, "--evidence", str(log), "--candidate-files", str(void), "--write"])
            == 0
        )
    task(
        "experiment/r2-gc",
        "## 任务描述\n\n- 假设：API 进程 gen-2 GC 停顿\n",
        "title: API 进程 GC\nparent: plan/r2\nmachines: [smc-a16-21]\n"
        + expect.format(d=[1.2, 1.8]),
        st("queued", "2026-10-23T05:00:00Z"),
    )
    task(
        "experiment/r2-router",
        "## 任务描述\n\n- 假设：decode 请求均衡\n",
        "title: decode request balance\nparent: plan/r2\nmachines: [smc-a16-20]\n"
        + expect.format(d=[2, 5]),
    )
    task(
        "close/r2",
        "## 任务描述\n\n- 第 2 轮收尾：集成、封版、归档、复盘，起草下一轮的 intake。\n",
        "title: 第 2 轮收尾\ndepends_on: [{task: plan/r2, kind: children}]\nmachines: [smc-a16-19]\n",
        st("queued", "2026-10-23T03:00:00Z"),
    )
    return pdir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(main(sys.argv[1]))
