# `agent_sys` 框架 bug 记录 — 合并版

生成于 2026-09-05T06:05Z。合并 `temp/bugs/` 下 **15** 份原始记录,原件保留不动。

**这是框架本身的账,不是我们 definition 的账。** 后者在
`validator.failures.2026-09-05.md`(37 条 validator 失败 / 27 次运行 / 11 个
validator)。两份文件主语不同:那份是**我们写的东西被拒**,这份是
**`agent_sys` 自己的行为**。

**每一条的 `file:line` 都重新打开源码核对过,不是抄原记录。** 核对结果写在每条
的「核对」里;**有两条经不起重读,单列在文末**,不混进正文。

按「今天还会不会再撞上」排序,最可能的在前——不是按时间。

| # | 现象 | 位置 | 状态 |
|---|---|---|---|
| 1 | `output_absent` 报的原因是假的 | 事件属性 | **未修复** |
| 2 | validator 的 stdout 无处可寻 | `validator/check.py` | **未修复** |
| 3 | `INVALID` 两义,退出码只读一义 | `agent/runner.py:952`, `cli/main.py:1278` | **未修复** |
| 4 | 声明一条预期失败 = 关掉全run完成度检查 | `cli/main.py:1408` | **未修复,已绕过** |
| 5 | 命令错误写进产物字段,全部 validator 通过 | 产物层 | **未修复** |
| 6 | escalation 到顶无人接收 | `monitor/base.py:731` | **未修复** |
| 7 | 重试卡在自己留下的半开 version 上 | handoff store | **未修复** |
| 8 | monitor 停转,run 被判定无法推进 | `monitor/base.py:345`, `cli/main.py:1053` | **未修复,部分撤回** |
| 9 | terminal task 的 prepare+body+seal 共用 20 秒 | `cli/main.py:1015` 邻近 | **未修复** |
| 10 | 无法解析的 `${...}` 静默当字面量 | `spec_loader/variables.py:81` | **未修复,已绕过** |
| 11 | validation zone 找不到 `claude` CLI | `agent/backends/claude_sdk.py:303` | **已绕过** |
| 12 | 两个 `copy_out`,docstring 承诺了另一个的保证 | `env_mgr/grants.py:177` | **未修复(文档问题)** |
| 13 | 在 Slurm 计算节点上跑 `agent-sys` 的三个坑 | 环境,非代码 | **已绕过** |
| — | ~~stall detector 会掐掉在干活的 task~~ | `cli/main.py:1015` | **已撤回** |
| — | ~~program task 标记 running 但没有 program~~ | `task_graph/scheduler.py:265` | **已解决,标题是错的** |

---

## 1. `output_absent` 报的原因是假的

**观察**:事件 `message` 写 *"declared output … was never delivered"*,而文件**交
付了**(m2 的例子里 35 个),真正被拒的是 **seal**。原因在
`attributes.seal_refused`,`message` 一个字都不提;`detail` 写 `exit 0`,反过来
诱导读者以为框架把活丢了。

**核对**:属实,且这条今天直接害过人。本记录作者在 T+1062 依据 `message` 向
leader 报告*「原因未测量」*,而同一事件里 `seal_refused` 写着
`required section 'Interface' is missing`,连缺哪个 section、现有哪些 section、
规则是什么都在里面。**四个属性只读了一个。**

**状态**:框架未修。工具侧已修——`assets/lib/read_events.py` 现在打印
`_text` 没用到的**全部**属性(`841ca22`)。

**绕过**:读事件时不要只看 `message`;`seal_refused` 和 `detail` 各是半个答案。

---

## 2. validator 的 stdout 无处可寻

**观察**:validator 认真写出的诊断走 stdout,而 stdout 不被保存。一次正确的拒绝
连同理由一起消失。

**核对**:`validator/check.py` 存在。这条与 **T14** 是同一道缝的两个方向:T14 是
*validator 解释器起不来 → 非零退出 → 不写 `verdict.json`*,阶段无法区分「坏掉的
validator」和「被拒的 handoff」;这条是 *validator 跑起来了、写了诊断 → 送进空气*。

**状态**:未修复。m3 在 `cff4571` 补了另一半——**exit 0 的 body 输出会被丢弃,
而那正是最疼的情况**。

**绕过**:`assets/lib/read_events.py` 读事件存;`validator_report.txt` 在 zone 里
读得到,但要**定位到那一个 validator 自己的 report**,不能全run grep(见第 5 条
的注意事项)。

---

## 3. `INVALID` 两义,run 的退出码只读一义

**观察**:`INVALID` 既表示「封装完成、判定为负」,也表示别的状态,退出码把两者读
成一件事。

**核对**:
- `agent/runner.py:952` — *"`INVALID` rather than leaving the slot `GENERATING`: a hole is …"*
- `cli/main.py:1278` — *"`INVALID` is a *sealed* negative verdict and is unamb…"*
- `task_graph/runner.py:97` — `verdict = HandoffStatus.VALID if valid else HandoffStatus.INVALID`

三处都在,且互相印证:代码自己在两个地方解释 `INVALID` 该怎么读,说明歧义是真的。

**状态**:未修复。`0afabf1` 另记了 `_completion_gaps` 的另一半也不成立——
**task status 从不 finalise**。

---

## 4. 声明一条预期失败,就关掉整个 run 的完成度检查

**观察**:框架有 expected-failure 机制,但**用不了**。

**核对**:三点全部属实,逐条打开确认——

```
cli/expectations.py:246   _BY_PACKAGE: dict[str, ExpectationSet] = {"demo": DEMO}
cli/main.py:1408          gaps = sorted(unfinished) if nothing_promised else []
```

第一行:**不能按 package 声明**,写死了 `"demo"`。
第二行:**只要承诺了任何一条**,`nothing_promised` 为假,`gaps` 恒为空——
**十五个 handoff 的完成度检查一起关掉**。
第三点(sealed verdict 没有 reason 字段,承诺钉不住具体原因)与前两条同源。

**状态**:未修复,**已绕过**。接受判据因此不用退出码,改成「指名文件 + 指名失败
条件」:一个 run 产出 15 handoff / 43 verdict / 42 true,唯一的 false 是
`check_no_regression` on `integration_report`,且其 `validator_report.txt` 恰好
四条 `PROBLEM:`。`assets/lib/accept_mock.py` 实现这条。

---

## 5. 命令的错误文本被写进产物字段,全部 validator 通过

**观察**:`engine_argv.txt` 整份内容是一条命令的报错;handoff 照常封装,三个
validator 全绿。**有内容的错误文件比空文件更难发现。**

**核对**:2026-09-05 00:45 在 rung 2g(`20260905T003217-c44c92`)发现,首个
skip-ahead run。属实。

**注意(本记录作者实测,给写检查脚本的人)**:`PROBLEM:` **不是拒绝独有的**。
在 `20260904T112414` 里,23 份 `validator_report.txt` 共 **11 条 `PROBLEM:`,分布
在 6 份里,其中 7 条出现在 verdict 为 true 的 5 份 report 中**。全run grep
`PROBLEM:` 会数到 11 条并且判错——必须限定到那一个 validator 自己的 report。
另:`PROBLEM:` 行**有两格缩进**,`grep -c '^PROBLEM:'` 返回 0。

---

## 6. escalation 到顶,没有接收方

**观察**:输出缺失时 escalation 正确触发,四次全部得到 *"nothing to push"* ——
一次是 *"the executor is a program body: there is no agent to instruct"*,三次是
*"the attempt holds no executor: it is not in its main phase"*。**机制跑通了,结论
无处可去。**

**核对**:`monitor/base.py:731` 是 `_apply(self, decision, unit)`,docstring 写
*"An unrecognised one is an error — nothing here may default to a benign outcome."*
机制本身是严谨的;缺的是接收方,不是判断。

**状态**:未修复。与第 2 条、T14 同族:**机器正确地得出诊断,然后没有地方送。**

---

## 7. 重试卡在自己留下的半开 handoff version 上

**观察**:`build_workset` 失败后三分钟重试,`handling_failed` 报
*"657bcbde v0 is already open by task 356505d8"* ——**开着 v0 的正是现在重试的
同一个 task**;随后 `monitor_gave_up`:*"the pusher has no action for
handling_failed"*。恢复路径存在、触发了、然后不恢复地终止。

**核对**:属实。磁盘上该 handoff 有 **v0/v1/v2/v3 四个 version 目录**,store 里
只有**一个** version(`v0`,`generating`)。**`v1` 有 39 个文件**——但其
`content/` mtime 早于本次 run,且是 `drwxrwxrwx`,是 `cp -a` 进来的 mock 素材,
**证明 body 跑到了「开 v0、建 v1、填充」这一步,不证明它产出了 workset**。

**状态**:未修复。

---

## 8. monitor 停转,run 被判定无法推进

**观察**:`monitor 'default' has stopped turning`。

**核对**:`monitor/base.py:345` 是 `check_liveness(`,`:578` 是 `_beat(`,
`cli/main.py:1053` 是 `for record in check_liveness(monitors, period=period):`
——三处都在。

**但原记录自己带一节「What is NOT established, and the claim that was
withdrawn」,且写明「the run is dead」是错的**:runner 还活着在转,死的是别的东西。

**今天的补充**:leader 曾把 rung 2d 的 monitor 死因报成*「一次瞬时 I/O 阻塞触发
了 liveness 检查」*,随后自己撤回——在**健康的** rung 2e 上采样,8 次里有 1 次
是 `D` 状态而 NFS 4 ms 就答。**`D` 状态不具区分力,rung 2d 的死因至今不明。**
m1 缩小了范围:217 在 18:31:52 已完全拆除,比 rung 2d 最后一次写早近十分钟,
所以 monitor 是在**空闲、拆除后、无容器**阶段停的——排除了整个「负载下卡死」家族。

**状态**:现象未修复;**具体死因未确定**,不要当成已知原因引用。

---

## 9. terminal task 的 prepare、body、seal 共用 20 秒

**观察**:一个 terminal task 的三个阶段共享 20 秒预算。

**核对**:`agent/runner.py:717` 存在(`seal_refused=refusals.get(...)`),
`cli/main.py` 中 405/891/911 三处被引用。原记录状态写明:*reported, not fixed —
`cli/main.py` 超出本轮商定范围,且当时有四个 run 在飞。*

**状态**:未修复,超范围。

---

## 10. 无法解析的 `${...}` 被静默当作字面量

**观察**:写错的变量引用不报错、不警告,原样穿过。

**核对**:`spec_loader/variables.py:81` 是 `_REF = re.compile(`,**上方注释明说
嵌套引用是被刻意排除的**:*"a nested reference would make this a grammar, and the
alternative to a grammar is a parser nobody asked for."* ——所以「不支持嵌套」是
设计决定;**问题在于不支持时不出声**。原记录把两者分得比我这段更细,见原件。

**状态**:未修复,package 侧已绕过。**同族**:未识别的 `--var` 同样被接受
(rc=0、零警告),失效的 `${VAR:?}` 同样沉默——**三处都把「给不了的输入」当成
合法输入。**

---

## 11. validation zone 找不到 `claude` CLI

**核对**:`agent/backends/claude_sdk.py:303` 存在,是
`options.setdefault("cwd", self.assignment.zone)`,与「zone 改变了 body 看到的
环境」一致。`validator/phase.py:451` 是 `def read_verdict_file(`。

**状态**:package 侧已绕过,框架侧未修。原记录状态即如此写。

**同族**:validation zone 会重写 `HOME`(`validator/environment.py:232-237`,
`{**dict(config.values), "TMPDIR":…, "HOME": str(zone/"home"), …}` —— `config.values`
先展开、`HOME` 后赋值,所以 **zone 的值一定赢**),`PATH` 被削成
`/usr/bin:/bin`(`spur` 在 `/usr/local/bin`,于是 transport 直接消失),
`SPUR_CONTROLLER_ADDR` **未设**。

---

## 12. 两个 `copy_out`,docstring 承诺了另一个的保证

**观察**:文档承诺的完整性保证属于另一个同名函数。

**核对**:`env_mgr/grants.py:177` 写着 *"`copy_out` verifies the digest *before
returning*, so integrity arri…"*;`agent/gate.py:226` 是
`content = store.copy_out(hid, version, Path(tmp) / "out")`;
`handoff/protocols.py:294` 与 `handoff/digest.py:82` 亦在。**两个 `copy_out` 确实
都存在,docstring 的错配是真的。**

**状态**:未修复,超范围。这是文档缺陷而非行为缺陷——但读者据此推断的保证是假的。

---

## 13. 在 Slurm 计算节点上跑 `agent-sys` 的三个坑

**观察**:三个环境性故障,非 `agent_sys` 代码缺陷。原记录标题即写明 *all
environmental*。

**今天的补充(实测)**:
- `/shared_nfs` 在登录节点是 **`ro`**、在计算节点是 **`rw`**,同一个 volume
  (`b2e6868e-…`)。run 从登录节点发起,所以 run root 必须移到
  `/home/yihou/agent_sys_runroot`。
- **agent shell 里的 `find` 不是 `find`**:是 shell function,execs Claude Code
  binary 当 `bfs 4.1.1`;真正的 GNU findutils 4.9.0 在 `/usr/bin/find`。
  `bfs` 拒绝 GNU 的相对时间写法并**大声报错、rc=1**,但大家都 `2>/dev/null |
  wc -l`,把诊断和退出码一起丢掉。**`-newermt` 本身没有问题。**
- 计算节点的 `python3` 是 3.12.3,登录节点是 3.14.6;从 repo 目录跑
  `python3 -m agent_sys.cli.main` 在节点上挂在 `agent_sys/cli/main.py:31`
  的 `from cli import …`——是 `sys.path` 没配,不是「跑不了」。

**状态**:已绕过。

---

# 经不起重读的两条

**这两条按 leader 的要求单列。它们留在 `temp/bugs/` 原处,但不应再被当作
框架缺陷引用。**

## A. ~~stall detector 会掐掉正在干活的 task~~ —— 已撤回

**原记录声称**:`cli/main.py:1015` 的 `(not holding or blocked)` 与 docstring 说的
合取不符;`blocked` 在本 package 里恒真,于是守卫退化成「20 秒没动静」,健康的
安静操作与卡死无法区分。

**核对**:`cli/main.py:1015` 确实是
`elif (not holding or blocked) and time.monotonic() - last_change > stall_after:`
——**代码引用没错,推论错了。**

m4 自己撤回(`e846dec`):`blocked` **不是恒真**。它是
`[t for t in live if _awaiting_a_decision(t, registry)]`,读的是 escalation
**记录**;干净的 run 里为空,只有发生 escalation 后才非空。m4 在一个活的 rung 1
上查了 26 分钟——**store 里没有任何 escalation 记录**。

**本记录作者补的第二个证据(不同的 run,leader 未用过)**:在
`20260904T041742` 里,`blocked` 空了 **51 分钟**,直到 05:08:56
`operator_workset` 被判缺失的那一刻才非空。**跨两个 run:escalation 是失败的
事件,不是背景状态。**

**第三个证据,直接否掉「掐掉了在干活的 task」**:`build_workset` 的叶子任务
三次运行的存活时间是 **13.8 s / 10.8 s / 16.9 s**,三次都在 20 秒阈值**以下**,
三次的 escalation 都在 `output_absent` **之后**(微秒级)。
**没有任何一次运行显示这个 detector 掐掉过任何东西。**

**原文件里 m1 的原始缺陷仍然成立且值得修**:`(not holding or blocked)` 与
docstring 的 *"and no attempt holds a thread"* 不一致——**一旦有东西
escalate,正在执行的叶子会在 20 秒后被拆掉。** 但这是**未被观测到的**风险,
不是已发生的事故。

## B. ~~program task 标记 `running` 但没有 program~~ —— 已解决,标题是错的

**原文件自己写着**:*"RESOLVED 2026-09-05 02:37 — and the title is wrong. **The
program runs.**"*

抓法是把子进程采样从**每 10 分钟改成每 5 秒**,跨状态切换采:
`02:35:24 .. 02:36:20 children=1 (mode_on 'running')`。

**核对**:`task_graph/scheduler.py:265` 存在,是 `continue  # take nothing;
stay queued`。代码没问题。

**这一条是「零 vs 缺席」的教科书例子**:*没看到子进程* 被读成 *没有子进程*,
而采样间隔比子进程寿命长。**分母没测,就报了分子。**

---

# 计数

- **正文 13 条**,均已核对 `file:line` 且引用属实。
- **2 条经不起重读**:stall detector(推论被作者撤回,且本记录补了两组独立
  反证)、program task without program(原作者已自行解决并写明标题是错的)。
- **0 条**因为文件/行号不存在而作废——15 份记录里所有 `agent_sys` 引用都能打开。
  唯一一次「找不到」是本记录作者抽取引用的正则丢了 `agent/` 前缀,
  与被记录的缺陷无关。

---

# 2026-09-05 新增

## 14. 一个等待步骤等的是**另一个 package** 的日志

**来源:m2 `475f2fc`(`assets/load/capture.sh`、`line.sh`、`replay.sh`,+42/-3)。
以他们的提交为准,本记录未独立复现。**

**观察**:capture 步骤等待的日志路径属于**另一个 package**。在这台机器上它**解析
得开**——五个 demo 都还在磁盘上——所以它能通过 review、能跑通、能过 CI。
**只有在缺少那个 package 的环境里才会失败。**

**为什么单列一条**:这是本轮的**新缺陷类别**。前面十三条里的缺陷都在本 package
内部;这一条的错误**指向 package 之外**,而本地磁盘的富余恰好把它掩盖。**环境越
完整,越发现不了。** 这与第 20 条(注释里写着反模式,grep 把注释当代码)是同一
族的另一面:**都靠「本地恰好有」而活下来。**

leader 已请 m3 在自己的 stage 里扫同一形状。

**状态**:m2 已修(`475f2fc`)。**类别未扫完。**

**不要按邻近性归因**:它是否是今天三条 profiling 线死亡的原因,**由 m2 判定**,
本记录不作此推断。

---

## 15. 三条 profiling 线在二十分钟内死亡 —— 原因未定,占位

**这不是 validator 失败,所以不在 `validator.failures.2026-09-05.md` 里。**
`0 validation(s) dropped`,没有任何拒绝,分配仍然持有,stage 仍是 `running`。

**三个实例**:`p4_a`、`p4_b`、`p4_m4real`,均在 2026-09-05 上午二十分钟内。

**为什么难看见**:**运行在任何人会去检查的字段上都是健康的。** 分配持有、阶段
`running`、零丢弃。今天有三件仪器会把这个状态读成「活着」,其中一件是本记录作者
的存活探针(19:23 之前)。

**状态**:**原因未定。持有人 m2,正在 088 cards 0–3 上复现。**

**这一条是占位**,写在这里是为了让 m2 的结论有地方落,而不是另开一个文件。
**在 m2 给出判定之前,不要把它和第 14 条合并**——两者的邻近性不是证据。
