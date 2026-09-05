# 用户的规矩 —— 读任何别的东西之前先读这里

> 用户 2026-09-05:*「拜托你把我说的话都沉淀到 CLAUDE.md 的最前面,每次都忘。」*
> 下面每一条都是用户亲口说过、而且**我至少忘过一次**的。忘记的代价写在每条后面,
> 那不是修辞,是当天的账。

## 集群

1. **我们持有的所有机器上,凡是不是我们和系统的 GPU 占用,一律杀掉,然后并行安排
   工作。** 用户 2026-09-05 放宽:**不分 QoS**,`amd-primus-q` 同样适用。
   (最初只说 burst-qos,同日放宽到全部持有的机器。)
   我方 = 带 `infera_e2e_run` 标签;系统 = 集群监控;
   其余全杀,**不论它跑了多久、不论谁还持有那台机器的配额**。
   **只杀 GPU 占用**——不占卡的 CPU 容器不动。
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
15. **算力宁可撞墙也别空着。** 一段跑不完的时间照样有价值——它能提前撞出问题并
    记录下来。用户 2026-09-05:*「可以发,这样遇到问题可以及时发现记录。」*

---

# 调试技巧 —— 每一条都是当天赔出来的

## 循环的形状,比循环里的努力重要

- **一次全链运行只返回一个 bit。** 图停在第一个 invalid handoff,所以串行跑链
  每次只暴露「下一个错」。*代价:两天,换回一个「rung 1 通过」,而 rung 1 就是
  stage 1 单跑。*
- **先跑最便宜的那个实验。** 同样的信息量:并行探针 3 分钟,串行全链 48 小时。
  mock 在前、GPU 在后;能在登录节点判的,不要占卡。
- **能并行发现的,不要串行发现。** 阶梯(逐级提升)的唯一作用是可归因,而
  **输入劫持并行注入同样可归因**。`mock_root` + `mock_stages` 就是劫持机制。

## 不要相信工具,先证伪它

- **我的探针一小时内被 owner 抓出四个缺陷**:空 args(26 行平凡通过)、
  README 取名导致索引到两天前的旧语料、`${var:-default}` 未替换(10 行崩溃被
  读成驳回)、`overrides` 参数从没被传。**每次都是 owner 在自己的材料上复现后
  告诉我的。**
- **消息可以撤回,提交不会。** 一句关于 `--kernel` 的错误说法在 90 分钟里流通了
  三手(m4 提出→自己撤回→我转发→m3 写进代码作为修复理由)。**抵达持久化存储的
  恰好是错的那一个。** 而且它落在最糟的位置:**一个由假理由支撑的正确修复,会被
  下一个认真的人拆掉**——他去核理由,发现不成立,于是移除守卫。
- **`ast.parse` 只证明语法,对名字解析一无所知。** m1 用它验了两次并称之为
  「verified」,而文件里带着一个 `NameError` 躺在共享树里若干分钟。**没有装
  pyflakes 时用 `symtable`。** 一般形式:**仪器无法以你需要的方式失败,而它的
  输出不会告诉你这一点。**
- **错误字符串是真值。** `command -v` 以 `exec: bash: not found` 失败,变量非空,
  于是「这个容器有该程序」。和 `find` 失败成零同形——**只因为答案荒谬才被抓住。**
- **重建是对共享文件的破坏性操作,哪怕它里面每一处改动都对。** checkpoint 重新
  生成失败账本(142+/179−),每一条都是改进,却把 readme-cn 同一小时里的五处
  更正一起生成掉了——**提交本身无法区分「正确的重建」和「错误的合并」**。
  同形状:`git commit -- <path>` 取的是工作树。**推论:一个文件只能有一个 owner。**
- **一个「曾经为真」的值,可以经由**指令**抵达,而不只是经由工具或记忆。** 我让
  m4 把 217 上测的 13.3% 写进 287 的产物。工具和记忆都有防御(重测、重读、查
  分母),**协调者的指令没有——它抵达时已自带授权**。owner 必须像核查工具一样
  核查交给他的指令。
- **坏掉的仪器,只有在它和你已知的事情矛盾时才会被抓住。** m3 抓住 `find` 的空
  结果,不是因为谨慎,是因为两条命令之前他刚 `ls` 过那个目录。**如果那条命令是
  他跑的第一条,他会信。这种情况至今没有任何防御。** 推论:一个工具给出的
  **第一个**结果,和一个**确证**你已有判断的结果,是最不该信的两种。
- **派活时明说「如果那行是工具的假象,就停下来告诉我」。** 这一句比工具本身值钱:
  它给了人「交回空结果」的许可,四个 owner 因此没有改动任何一个 validator。
- **通过的那一侧往往更危险。** 一个被剥掉阈值的 validator 是**平凡通过**的,
  所以 26 个 PASS 是最弱的证据,不是最强的。

## 零和缺席

- **一个「零」到底是「没记录」还是「没发生」,不跑控制组分不出来。**
  踩过三次:`zone: NONE`(成功的任务也是 NONE,该字段根本不存)、
  `ps --ppid` 无子进程(几十分钟后才查,子进程早退了)、
  `ws_handoff_refine` 零写入(目录是只读挂载,再忙也是零)。
  **先测分母,再报分子。**
- **崩溃会挡在真实驳回前面。**「崩了」不等于「否则会通过」——10 行崩溃里有 4 行
  修好后照样驳回。
- **「暴露」不等于「原因」。** 数「有多少行含模板」是暴露面,数「有多少行崩在
  模板上」才是原因。

## 读产物,不读退出码

- **`SAVE_OK` 只是退出码**;拿 `docker load` 去验才知道 tar 完整。
- **失败命令的输出会被写进产物字段。** `engine_argv.txt` 里存着
  `Error response from daemon: ...`,而三个 validator 全部通过——
  **有内容的错误文件比空文件更难发现。**
- **`| head` / `| tail` 会吞掉 rc。** 今天吞过四次,其中一次差点让我给正确的
  代码报 bug。用 `PIPESTATUS` 或分开取。

## 不要从名字推断

- **容器名前缀不等于归属。** 我从 `yihou_e2e_flow_*` 推出「m1 在调试」,报给用户
  两次,实际是 run 自己的 agent。**判据是 FATAL 之后 5 秒就出现的容器,不是前缀。**
- **删掉一行之前,先读它上面那段注释。** 那条 `froms` 边看起来是多余的排序边,
  注释说明它保护着下游唯一可信的吞吐数字,而且删错了**不会有任何报错**。

## 环境

- **算「还剩多久」。** `squeue` 的 `%M` 是已用,上限 8 小时。
- **`/shared_nfs` 在登录节点是只读**(`ro,relatime`),计算节点可写。
  从登录节点 `mkdir` 会静默失败——不要把 `2>/dev/null` 加在会失败的命令上。
- **zsh 不对未加引号的变量做词分割**,`--include=*.json` 会被 glob 掉,要加引号。
- **`--stall-after` 是双刃**:默认 20 秒让「升级无接收方」几秒内就报出来;
  我设成 3600 把同一条诊断藏了一小时。**900 秒是折中**——够一次冷启动(实测
  222/232 秒),又不至于把失败藏起来。

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
