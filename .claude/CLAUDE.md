# 用户的规矩 —— 读任何别的东西之前先读这里

> 用户 2026-09-05:*「拜托你把我说的话都沉淀到 CLAUDE.md 的最前面,每次都忘。」*
> 下面每一条都是用户亲口说过、而且**我至少忘过一次**的。忘记的代价写在每条后面,
> 那不是修辞,是当天的账。

## 集群

1. **burst-qos 节点上,凡是不是我们和系统的 GPU 占用,直接杀掉。**
   先看 QoS(`squeue -u $USER -o "%.9i %.18N %.12q"`):`amd-burst-qo` 适用,
   `amd-primus-q` 不适用。我方 = 带 `infera_e2e_run` 标签;系统 = 集群监控;
   其余全杀,**不论它跑了多久、不论谁还持有那台机器的配额**。
   用 `docker stop -t 10`,**永远不要 `rm -f`**。
   **这条覆盖「你没创建的容器不归你删」和「两个持有者时归属不可判定」。**
   *代价:我拿后两条当挡箭牌拖了四轮,217 在别人 34% 的作业下空转数小时。*

2. **不允许申请机器,只允许查询和使用。** 不发 `sbatch` / `salloc`。
   机器不够就说不够,不要自己解决。

3. **删除铁律:路径不含 `yihou` 或 `/tmp` 子串的东西,一律不删。**
   宿主机上如此,docker 挂进宿主机的目录里同样如此。

4. **算力要算「还剩多久」,不是「已用多久」。** 作业上限 8 小时,`squeue` 的 `%M`
   是已用。发一条线之前先算剩余时间够不够这条线跑完。
   *代价:昨天四个 hold 死在 8 小时整点的半途。*

## 怎么用算力

5. **所有任务一律 TP4,不要占满八卡。** 一台 8 卡机同时跑两条线(0–3 / 4–7)。

6. **跨节点先试。** 失败了分析失败原因,不行再回退到同一节点。不要预先把它当路障。

## 本轮的目标与顺序

7. **本轮不关注端到端是否有真实性能提升,跑通即可。**

8. **五个阶段,按这个顺序:**
   ① 造一份新的 handoff + 重构一遍所有 validator
   ② 单独并行跑每个 mock,清掉参差
   ③ 串通 mock
   ④ 单独跑每个真实
   ⑤ 串通整个真实
   两条线之间不断调试,按需修改 handoff(定义和造出来的实例)与 validator。

9. **e2e 调通阶段可以先把 task 从 program 改成 ai**,让 agent 吸收 handoff 里的
   细微差异,避免不必要的失败——**前提是把 markdown 写明白**。
   (`agent/runner.py:801`:环境变量无法指示 agent,只有 brief 能。)

10. **造出来的 handoff,在评估影响不大时可以手工微调**,不必为此重跑真实负载。

## 记录

11. **框架的 bug,无论修了没修,统一记进 `bug.record.<date>.md`。**

12. **调试过程中每一次 validator 失败都要记下来**,便于后续分析。

## 工作方式

13. **小事别找我确认**,用户不一定一直在。
14. **工作用英文,向用户汇报用中文。**

---

# Task — Handoff refine: chain the five LLM e2e optimisation stages into one graph

Build **one** task package,
`agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/`, whose graph runs
stage 1 → 5 in a **single `agent-sys run`**, to the rules in the repo-root
`mission.md` (rewritten 2026-09-03).

Three deliverables, in order:

1. **Definitions rewritten** to mission.md's general rules + per-module list.
2. **Mock e2e green** — mock agents fed by the 25 real sealed handoffs in
   `/shared_nfs/yihou/agent_sys/cheat_for_mock/`; one run walks all five stages
   and every validator passes.
3. **Real e2e** — staged, one stage promoted from mock to real at a time.

Decided with the user 2026-09-03: **new package, the five demos stay untouched**;
**all three items this round**; **team = 5 module owners + 1 checkpoint writer**
after a solo contract freeze.

## Background

The five stages exist today as five *separate* packages (`deploy-demo`,
`profiling-demo`, `analyze-demo`, `kernel-opt-demo`, `integration-demo`), each
driven to a real cluster run on 2026-09-02: **45 sealed handoffs, 50 validator
PASS, 2 documented refusals**, ~3.5k lines of YAML and **~20k lines of proven
`.py`/`.sh`**.

They are not a flow. **A handoff only travels inside one run's graph**, so five
packages are five runs and nothing chains. The seams recorded in
`handoff.analysis.md` — one `kernel_table` name over two `content_type`s,
"workset" spanning two kinds with different required files, two same-named
`check_service_live`s — are exactly what side-by-side authoring produces.

**This is a refine of definitions, not a rewrite of bodies.** The 20k lines of
assets are the only thing here that has ever produced a number; they move and
adapt, they do not get re-derived.

## Context — this environment, measured 2026-09-03

| | |
|---|---|
| where I am | login node, **no GPU**, no direct docker daemon |
| GPU holds | `106250`→`crsuse2-m2m-061` (10.245.159.129), `106253`→`crsuse2-m2m-031` (10.245.144.239); 8 h each from ~12:40 UTC; both reachable, both carrying **other tenants' containers** |
| reaching them | `spur exec <jobid> bash -c '...'` — exec namespace; docker talks to the **host** daemon, but the **filesystem identity is you, not root** (measured 2026-09-04: `id -u` → `50112975`, writes land `-rw-r--r-- yihou ubuntu`). Matters because `/home` is `sec=sys` NFS, where a root-squashed write would map to `nobody` and leave a tree nobody can clean up. |
| shared FS | `/shared_nfs` 360 T, **46 T free**, shared by every spur node — this is how "remote" works |
| scratch | `/shared_nfs/yihou/agent_sys/ws_handoff_refine/` — all temp activity lives here |
| mock inputs | `/shared_nfs/yihou/agent_sys/cheat_for_mock/` — 25 real sealed handoffs, one folder per kind |
| fast loop | `python3 -m agent_sys.cli.main show --package <dir> --var …` loads and type-checks every yaml in **< 1 s** |

## Key references

- **`mission.md`** (repo root) — the authority. Every requirement below traces to
  a numbered item there.
- `handoff.analysis.md` (repo root) — the 26 kinds, their validators, and the
  three cross-stage seams.
- `/shared_nfs/yihou/agent_sys/cheat_for_mock/README.md` — **four things that
  will mislead you**, including a `kernel_table` that is a 34-row synthetic seed
  and an `integration_report` carrying a *refused* verdict.
- `/shared_nfs/yihou/agent_sys/debugging/integration/DELIVERY-NOTE-FROM-LEADER.md`
  — why that refusal was the validator working, and why the 5 % / 10 % bars must
  **not** be widened.
- `/shared_nfs/yihou/agent_sys/temp/leader/repair_modes.py` — restores the file
  modes a past `chmod -R 777` erased from sealed handoffs.
- `agent_sys/docs/design.md` §13 (WIP) — the runtime tree.
- `agent_sys/spec_loader/validate.py:34-56` — the `jsonschema` idiom to copy.

## Core principles

1. **Read the artefact, not the exit code.** Every acceptance claim names a file
   to open and a condition that fails.
2. **`items_schema` is not a schema layer.** Measured: for a file/tree item
   `handoff/content.py:184-197` validates the *filename string*, never the
   contents, and the schema is never exported to a body. Mission rule G2 needs
   real schemas under `assets/schemas/`, loaded by producer **and** validator.
3. **Every identifier bound on a shared host is a parameter.** Container names,
   ports, workdir, served model name. `: "${VAR:=…}"`, never `export VAR=`.
4. **Never delete anything on `/shared_nfs` whose path lacks the substring
   `yihou`.** Never `docker rm -f` a container you did not create. Never
   `agent-sys run --clean` on a shared root — it removes *every* run.
5. **Research → gather → analyse → plan → work.** Temp activity in
   `ws_handoff_refine/`; the repo receives only `e2e-flow/` and `todo.md`.
6. Bugs in `agent_sys` are recorded under
   `agent_sys/examples/llm_e2e_performance_optimization/temp/bugs/` first, then
   worked around; fixed only when the evidence is unambiguous.
7. Work in English; report to the user in Chinese.

## Other notable details

### Two mission requirements that read as traps

- **Rule 7, "all tasks share one docker container".** Module 5 needs **two** by
  construction — a container holds one state for its life, which is the whole
  reason the two-arm design exists. G5.1's *"如果不行，再考虑…启动不同的 docker
  container"* grants the exception: modules 1–4 share one, module 5 brings up
  its own two arms from the same image and the same `environment` record.
- **M5.1.3, Python runtime hijack.** M5.3 says *"首先记入 todo … 但现在就这样吧"*
  — record the disagreement in `todo.md`, **keep `overlay_files`**.

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer.

```bash
git commit -s -m "..."
git config user.name && git config user.email
```

Sign off **as yourself** — never a colleague's line, never a bot identity.

### Branch / PR

Branch `dev.yihou.aiopt.task_package.concat`. Activity is limited to
`agent_sys/examples/llm_e2e_performance_optimization/` plus the root-level
`*.md` notes this effort writes.
