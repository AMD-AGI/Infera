"""Make a demo AGENT_SYS_HOME: one infera-perf project in the middle of its first round, with three machines: the
template's five tasks done, the plan's experiments in most states (one still a draft), the round's close waiting for
them (a children dependency), and the ledger's curves (the baseline v0, two InferenceX references, the experiments'
working-point points), so the console can be looked at without running anything.

  python tools/demo.py <home>          # a new directory, or a demo home made before (it is replaced)
  AGENT_SYS_TMUX=asys-demo agent-sys-v2 serve --home <home> --sock /tmp/agent_sys-$(id -u)/demo.sock

The numbers are made up. Every run of the project replays a recorded session (tests/fixtures/events) instead of
calling a model: starting a task here costs nothing.
"""

import json
import shutil
import sys
from pathlib import Path

from agent_sys_v2 import ops
from agent_sys_v2.model import new_state, write_state

EVENTS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "events"
MARK = ".agent_sys_demo"  # in a home this script made: only such a home is replaced
DONE = (
    "状态: 已完成\n下一步: 无\n完成时间: 2026-10-21 10:00\n\n- [x] 部署\n- [x] 质量门\n- [x] 测量\n"
)


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

    def res(
        label,
        conc,
        x,
        y,
        t,
        series,
        role="candidate",
        decision=None,
        status="valid",
        change="",
        notes="",
    ):
        doc = {
            "schema": "infera.result/v1",
            "label": label,
            "time": t,
            "role": role,
            "round": 1,
            "series": series,
            "change": change,
            "metrics": {
                "input_tput_per_gpu": y,
                "p90_intvty": x,
                "p90_e2e_norm_intvty": round(x * 0.52, 1),
                "p90_ttft": round(1.2 + conc / 40, 2),
            },
            "workload": {"client": "agentx", "conc": conc, "duration_s": 3600},
        }
        if decision or status == "void":
            doc["verdict"] = {
                "status": status,
                **({"decision": decision} if decision else {}),
                "notes": notes,
            }
        return doc

    seen = {"seen_at": "2026-10-23T00:00:00Z"}

    def done(tid, when, cost, report, results=(), **more):
        task(
            tid,
            state=st("done", when, finished=when, cost_usd=cost, **seen, **more),
            progress=DONE,
            report=report,
            results=results,
        )

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
    v0 = (
        (16, 72.0, 9800.0),
        (32, 61.0, 16200.0),
        (48, 52.0, 20100.0),
        (64, 44.0, 22800.0),
        (96, 33.0, 25400.0),
    )
    done(
        "baseline/r1",
        "2026-10-21T06:00:00Z",
        41.2,
        "## 结论\n\nv0：工作点（X ≥ 40）输入 tok/s/GPU 23.7K，噪声 ±0.5%。\n",
        [
            (f"v0-c{c}", res(f"v0-c{c}", c, x, y, f"2026-10-21T0{i + 1}:00:00Z", "v0", "baseline"))
            for i, (c, x, y) in enumerate(v0)
        ]
        + [
            (
                "v0-c64-repeat",
                res("v0-c64-repeat", 64, 44.2, 22700.0, "2026-10-21T05:40:00Z", "v0", "baseline"),
            )
        ],
    )
    for tid in ("baseline/r1", "profile/r1"):
        ops.update_task(pdir, tid, {"machines": ["smc-a16-19"]})
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
        "## 结论\n\n6 个实验，3 台节点；本轮 ≤ 80 节点·小时。\n",
    )
    experiments = [
        (
            "r1-spf",
            "SPF 调度",
            "smc-a16-19",
            "done",
            44.6,
            25900.0,
            "accept",
            "--schedule-policy shortest-prefill-first",
            [8, 14],
        ),
        (
            "r1-tokenizer",
            "tokenizer workers ×4",
            "smc-a16-20",
            "in_review",
            45.1,
            23800.0,
            "accept",
            "--tokenizer-worker-num 4",
            [1, 3],
        ),
        (
            "r1-dcp4",
            "DCP4 decode",
            "smc-a16-21",
            "done",
            39.0,
            21000.0,
            "reject",
            "decode DCP4",
            [3, 6],
        ),
    ]
    expect = "expect: {{delta_pct: {d}, level: L4, falsify: 工作点 ΔY < 3%}}\n"
    for i, (name, title, machine, status, x, y, decision, change, delta) in enumerate(experiments):
        when = f"2026-10-22T{2 + 3 * i:02d}:00:00Z"
        task(
            f"experiment/{name}",
            f"## 任务描述\n\n- 假设：{change}\n",
            f"title: {title}\nparent: plan/r1\nmachines: [{machine}]\n" + expect.format(d=delta),
            st(
                status,
                when,
                {"kind": "review"} if status == "in_review" else None,
                finished=when,
                cost_usd=30.0 + i,
                **(seen if status == "done" else {}),
            ),
            DONE,
            f"## 结论\n\n{decision}。\n",
            [
                (
                    f"{name}-c64",
                    res(f"{name}-c64", 64, x, y, when, name, decision=decision, change=change),
                )
            ],
        )
    task(
        "experiment/r1-hicache",
        "## 任务描述\n\n- 假设：HiCache write-back\n",
        "title: HiCache write-back\nparent: plan/r1\nmachines: [smc-a16-21]\n"
        + expect.format(d=[4, 8]),
        st(
            "needs_you",
            "2026-10-23T03:00:00Z",
            {"kind": "checks_failed", "failed": ["results"], "passed": 1, "total": 2},
            cost_usd=22.4,
        ),
        "状态: 已完成\n下一步: 无\n",
        results=[
            (
                "r1-hicache-c64",
                res(
                    "r1-hicache-c64",
                    64,
                    45.0,
                    27100.0,
                    "2026-10-23T02:30:00Z",
                    "r1-hicache",
                    status="void",
                    notes="测量窗口内有别人的 sweep 跑在 GPU 3",
                ),
            )
        ],
    )
    task(
        "experiment/r1-gc",
        "## 任务描述\n\n- 假设：API 进程 gen-2 GC 停顿\n",
        "title: API 进程 GC\nparent: plan/r1\nmachines: [smc-a16-21]\n"
        + expect.format(d=[1.2, 1.8]),
        st("queued", "2026-10-23T05:00:00Z"),
    )
    task(
        "experiment/r1-router",
        "## 任务描述\n\n- 假设：decode 请求均衡\n",
        "title: decode request balance\nparent: plan/r1\nmachines: [smc-a16-20]\n"
        + expect.format(d=[2, 5]),
    )
    task(
        "close/r1",
        "## 任务描述\n\n- 第 1 轮收尾：集成、封版、归档、复盘，起草下一轮的 intake。\n",
        "title: 第 1 轮收尾\ndepends_on: [{task: plan/r1, kind: children}]\nmachines: [smc-a16-19]\n",
        st("queued", "2026-10-22T00:00:00Z"),
    )
    return pdir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(main(sys.argv[1]))
