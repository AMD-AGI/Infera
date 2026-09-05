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
   *代价:2026-09-05 我为了把 mock 回路压到 4 分钟,把五个阶段全部强制成
   `m*_agent=runner`(program),**正好和这条相反**,而且没说。*

10. **造出来的 handoff,在评估影响不大时可以手工微调**,不必为此重跑真实负载。
    用户 2026-09-05 追加:*「太久的错的语料不是我喂给你的,发现了顺手删掉就可以。」*
    **但「微调」不等于「编造」**:语料 workset 连 `workset.yaml` 都没有,而 evidence
    是它里面的字段、validator 会交叉核对 measured spread——补它就是造测量数字。
    **诚实的做法是拿今天真机产出的同类产物去嫁接**,不是手写。
    *代价:我两次把语料缺口当成不可动的边界,直到用户第三次点破。*

10a. **单独调试某个阶段时,不要求它产出真实的优化/提升。** 用户 2026-09-05:
    *「m4 单独调试时不要求跑真实优化……怎么舍弃优化提升尽快拿到结果。」*
    **可达性是目标,数字不是。** 降级产物必须自己声明降级。

10b. **下游阶段不许等上游跑完。** 用户 2026-09-05:*「m5 根本不需要等 m4 出来
    再说。」* 把上游算子加一两行无关代码就是一份合法的**反向优化**产物——
    空改动测出来的就是 baseline,所以 `speedup: 1.0` 是**正确值而不是猜测**。
    **m1/m2/m3 都真实跑通之后,下游没有任何「外界信息不足」的借口。**
    实例:`/home/yihou/make_reverse_kernel_opt.py`。

## Validator:先全关跑通,再二分打开

> 用户 2026-09-05,原话:*「每个节点真连调试的首个阶段,你可以把 validator 全部
> disable 掉。着重跑通,然后拿着一份结果,逐个修 validator,这时只需要跑一遍 e2e
> 就能拿到更多的信息。validator 打开的过程中,也可以分阶段二分的打开,比如先打开
> 稳定能 pass 的,再逐渐收敛到难 debug 的。过于难调的,单独记录文件上报给我就行,
> 真实跑通前不要把太多精力花在 validator 上。」*

10c. **每个节点第一次真连,validator 全关。** 机制是
    `assets/lib/make_debug_package.py --out <repo>/e2e-flow-noval`,生成一棵副本。
    - `validators: []` **非法**——框架原话 *"A kind with no validator cannot be
      admitted"*,所以注入 `check_nothing`(全 true、一个不读、往 stderr 打
      `validation is DISABLED for this run`、`strength: weak`)。
    - **必须生成在仓库内**:`env_mgr.workspace.cut` 要求包在带
      `extensions.preciousObjects` 的 git 仓库里。`e2e-flow-noval*/` 已 gitignore。
    - 回收用 `--keep a,b`,并且**保留 `check_nothing` 在每个 kind 上**,这样部分
      恢复的树仍然自报降级。
    - **`check_deploy_serves` 不能进这个回路**——它做真实 bring-up + 180 秒压测,
      登录节点上会挂死。先砍它,其余 20 个一起开。

10d. **一个 `-noval` 的绿只证明「走通」,不证明任何正确性。** 而且它**分不清
    「走通」和「越过了若干次静默失败」**——所有 verdict 都是 `check_nothing`。
    唯一诚实的句子是「链子在关闭验证的情况下走完了五个阶段」。

10e. **过于难调的 validator 单独记录上报,不要在跑通前纠缠。**

## 记录

11. **框架的 bug,无论修了没修,统一记进 `bug.record.<date>.md`。**
    (在 `agent_sys/examples/llm_e2e_performance_optimization/` 下。)

12. **调试过程中每一次 validator 失败都要记下来**,便于后续分析。
    (`validator.failures.<date>.md`,同目录。)

## 工作方式

13. **小事别找我确认**,用户不一定一直在。
14. **工作用英文,向用户汇报用中文。**
15. **算力宁可撞墙也别空着。** 一段跑不完的时间照样有价值——它能提前撞出问题并
    记录下来。用户 2026-09-05:*「可以发,这样遇到问题可以及时发现记录。」*

---

# 调试技巧 —— 每一条都是当天赔出来的

> **先读这一条,它决定下面每一条能不能生效。**
>
> **一条规则有三种形态,只有第三种会衰减:**
> 1. **改变命令** —— `tail -2` 而不是 `tail -1`;`NOW=$(date -u)` 然后引用它;
>    绝对路径 + 不用 `&&` 跨过会静默失败的步骤;可用的 `pgrep` 模式。
> 2. **改变产物必须包含什么** —— `runtime.replayed_from` 必须存在,而消费者在它
>    缺失时拒绝;`--node` 不匹配就 rc=2;**一个负控制必须「弄坏 → 放回 → 通过」**;
>    每条验收主张都要点名一个文件和一个会失败的条件;产物里要有出处小节。
>    **这一类不靠记忆,因为「有没有」是可查的。**
> 3. **要求人记得** —— 「读产物不读退出码」「先测分母」。**只有这一类会衰减。**
>
> **推论(readme-cn):当一条教训无法变成命令时,下一个问题不是「写得更醒目一点」,
> 而是「产物里要有什么,才能让这件事变成可查的?」** 这就是「mock 不许断言生产者
> 没有的事实」变成 CONTRACT §5.3 加一张表、「缺席不可见」变成第二遍审计的路径。
>
> *代价:readme-cn 今天第二次撞上「`git commit` 因 pathspec 加倍失败,而 `&& push`
> 照跑」——第一次之后他已经写下了教训。他的精确结论不是「写下来没用」,而是:
> **一条必须在复合命令中途、在时间压力下、在你正想着内容而不是机制的那一刻触发的
> 规则,形状就是错的。** 修法之所以有效,是因为它取消了那个选择的时刻。*

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
- **在多人同时编辑的文件里,永远引用可 grep 的锚点(标题文字、表格首格),不要
  引用行号。** readme-cn 的 L2133 在「他读到」和「对方读到」之间漂了 31 行,其中
  20 行是他自己那次提交造成的——**计算时正确,送达时错误,而且是他自己弄错的。**
  今天这个仓库里行号的保质期是个位数分钟。
  **三条轴,而第三条的失败方式不同**(readme-cn):**时间**——裸行号;**空间**——
  `campaign_config.py:316` 不带 checkout 路径;**分配**——从陈旧清单里取的 T 编号。
  **行号会漂移,标识符会撞车。** 漂移的指针误导一个读者然后自我纠正;**撞车的编号
  把两个发现塞进一个名字下,而且从有人引用它的那一刻起就不再便宜。**
  推论:往只追加的共享文件里追加之前,**在那一刻重读标题清单**,不是每会话一次。
- **消息可以撤回,提交不会。** 一句关于 `--kernel` 的错误说法在 90 分钟里流通了
  三手(m4 提出→自己撤回→我转发→m3 写进代码作为修复理由)。**抵达持久化存储的
  恰好是错的那一个。** 而且它落在最糟的位置:**一个由假理由支撑的正确修复,会被
  下一个认真的人拆掉**——他去核理由,发现不成立,于是移除守卫。
  **更尖的一版(m3):代码被改回来了,日志没有。** `f92e42b` 的 commit message
  至今带着那句错话且无法修订。**写 commit message 时要当它不可更正——因为它就是。**
- **`ast.parse` 只证明语法,对名字解析一无所知。** m1 用它验了两次并称之为
  「verified」,而文件里带着一个 `NameError` 躺在共享树里若干分钟。**没有装
  pyflakes 时用 `symtable`。** 一般形式:**仪器无法以你需要的方式失败,而它的
  输出不会告诉你这一点。**
  **`ast.parse` 和 `symtable` 都看不见实参个数错误。** m1 把十二个 `print(...)`
  机械改成 `_note(notes, ...)`,其中两处把 `file=` 带了过去,而 `_note` 不接受它。
  **两处都在 teardown 的 `except` 块里——所以 `TypeError` 只在「teardown 已经失败」
  时触发,也就是这条 note 最值钱的那一刻。** 这个包在 `NameError` 之后采纳的两个
  仪器对它都是盲的;**抓住它的是「真的去调用那个函数」。**
  **第三种失败方式(checkpoint 09:28):仪器答对了,而你没看。** 他在同一条命令里
  `date -u` 打出 `09:28:36Z`,却手写了 `09:05Z` 进文件——**读取发生了,就在同一行**,
  失败在顺序:他在读取返回之前就把值写死了。**规矩:`NOW=$(date -u …)` 然后引用
  `${NOW}`,永远不要在读取旁边手写那个值。** 前两种是「仪器在回答别的问题」,
  这一种是「仪器在场且未被使用」。
- **错误字符串是真值。** `command -v` 以 `exec: bash: not found` 失败,变量非空,
  于是「这个容器有该程序」。和 `find` 失败成零同形——**只因为答案荒谬才被抓住。**
- **一个「缺了就中止」的值,必须在命令块里,而不只在解释它的散文里。**
  (m5,2026-09-05 收尾时自己给的诊断,今天最值得带走的一条。)
  他们九个配置没跑到阶段 5,其中**四个死于本可预知的东西**——缺 wrapper、
  `aiperf_trace`、`mock_stages` 没配 `m<N>_agent`、stub 入口。
  **而其中两条是他们自己几小时前写进任务 brief 的。文档没有保护住启动行。**
  *同类:`--var transport=spur` 我加进了变量表和一个块,没进那个真被当范本的块,
  m3 照抄后两小时才发作。`--var deploy_entrypoint=scripts/stub/deploy.sh` 在封存
  的 kit 里一直存在,没有任何东西指向它,于是三次四十分钟的等待。*
  **推论:凡是「没有默认值、缺了就 abort」的变量,清点一遍,逐个进标准块。**
- **「多核一点」不是修法;「核那个能杀死它的具体问题」才是。而对一条出处主张,
  那个问题永远是同一个:这份产物是哪一次运行产出的?一次 `store/task` 读取。**
  (readme-cn 给我 2026-09-05 那次撤回的诊断。)
  *代价:我把「kit 记录了错误的节点」放大给全队,二十分钟内进了一个提交。
  **而反证在我自己的记录里躺了几小时**——我当天早上亲眼读过那条命令行
  `pid 2394563 --var node=crsuse2-m2m-217`。它不是「没查」,是**没把已有的事实
  和这条主张连起来**,因为它印证的正是我早已相信的那个限制。*
  **推论:撤回也可能像主张一样错。** readme-cn 在执行我的撤回之前,先自己去
  `store/task` 核了一遍才动手。
- **一个关于「将来不会」的主张,不能建立在对过去的计数上;计数只许可
  「至今没有」。**(readme-cn,2026-09-05,自己赔出来的。)他们把
  `check_deploy_serves` 记成「0/55,**永远**不会有理由」——**五小时后 m1 的
  `01ed8dd` 就推翻了它**:12/74,最近八次判定全带报告。**一个已被修复的缺陷,
  被写成了一条不可修复的属性。** 他们随后把 `check_environment` 条目里的「永远」
  也删了,尽管它的数字仍然是零。
  **推论:改正要落在「读者最先撞上的那一处」**,不只是汇总段——他们自己的发现
  正是「更正到不了快路径」,所以两处都改了。
- **重建是对共享文件的破坏性操作,哪怕它里面每一处改动都对。** checkpoint 重新
  生成失败账本(142+/179−),每一条都是改进,却把 readme-cn 同一小时里的五处
  更正一起生成掉了——**提交本身无法区分「正确的重建」和「错误的合并」**。
  同形状:`git commit -- <path>` 取的是工作树。
  **推论(2026-09-05 修正,原写「一个文件只能有一个 owner」,leader 已推翻):
  需要一个 owner 的是「重建」,不是「追加」。** `todo.md`、`bug.record` 是追加,
  保持共享;账本是重新生成的,归一人。**真正管用的是前置检查而不是归属**——
  写共享文件前先 `git diff`:**前置检查查的是世界,归属查的是协调者的记忆。**
  *代价:归属规则在宣布它的那条消息里就已经是破的——`todo.md` 当时正躺着 m3 的
  48 行未提交改动。*
- **一个「曾经为真」的值,可以经由**指令**抵达,而不只是经由工具或记忆。** 我让
  m4 把 217 上测的 13.3% 写进 287 的产物。工具和记忆都有防御(重测、重读、查
  分母),**协调者的指令没有——它抵达时已自带授权**。owner 必须像核查工具一样
  核查交给他的指令。
- **三个人对一个 flag「意图」的一致,不等于一次对它「效果」的测量。**
  `graph_ceiling.py:169` 的注释记录了三条佐证——m1 的注释、m5 读 `start_worker.sh`、
  两个源文件。**三条都是「这个 flag 应该是什么意思」的阅读,没有一条观察过引擎。**
  而 m4 的 kit 同时传 `--cuda-graph-backend-decode full`,新配置路径在 decode 上
  胜出,**那个 flag 被静默推翻**——于是这个判据恰好在「decode 图真的在跑」的运行上弃权。
  **来源独立不等于方法独立。** 三个人读同一类证据,只是三次同样的阅读。
  **tier 2 形式:一条断言「某 flag 有何效果」的注释,必须引用一次观察(运行、日志),
  不能只引用源码阅读——「它引的是哪一类」是可查的。**
- **一致性检查通过时,先问它比的两个值是从哪来的。**(readme-cn,2026-09-05,
  给 `_agree_or_die` 写的判据,但形状是通用的。)
  > 两个**独立**来源 → 这次通过是有信息的。
  > **一个来源被拆成两份** → 它只告诉你复制成功了。
  三条把一个来源拆成两份的路径,今天各出现一次:**继承**(下游全部
  `env_render --inherit` 同一份 kit)、**铸造**(记录在源头就写错)、**自引用**
  (我把 `premise.run_environment` 设成 workset 自己的环境)。
  **这是「所有比较方都共享的错误看不见」那条规则的极限情形——「所有方」退化成一方。**
  通过仍然返回 true,但它**排除不掉任何一种世界状态**。
- **观察者会写进它所观察的产物里。** m2 用 `worker.log` 的 mtime 当活性信号,
  而他自己每次 `health_generate` 探测都会在里面写一行「Health check failed」——
  **mtime 追踪的是轮询器,不是引擎。** 引擎最后一次真实工作停在 09:47:39,
  而 mtime 一直在动。**修法是 tier 1:这个轮询只读 `up=` 和 `traces=`,不读 mtime。**
  一般形式:**一个会写进产物的探测,使那个产物不能再当活性信号用。**
- **坏掉的仪器,只有在它和你已知的事情矛盾时才会被抓住。** m3 抓住 `find` 的空
  结果,不是因为谨慎,是因为两条命令之前他刚 `ls` 过那个目录。**如果那条命令是
  他跑的第一条,他会信。这种情况至今没有任何防御。** 推论:一个工具给出的
  **第一个**结果,和一个**确证**你已有判断的结果,是最不该信的两种。
- **派活时明说「如果那行是工具的假象,就停下来告诉我」。** 这一句比工具本身值钱:
  它给了人「交回空结果」的许可,四个 owner 因此没有改动任何一个 validator。
- **通过的那一侧往往更危险。** 一个被剥掉阈值的 validator 是**平凡通过**的,
  所以 26 个 PASS 是最弱的证据,不是最强的。
  **同一形状会出现在审计上**:一个只做正向遍历的检查,面对「缺席型」缺陷不是
  沉默,而是**报告文件健全**——十三行准确、零分歧、结果干净,而要紧的四行
  **从来不在被检查的总体里**。
- **一个结果如果印证了你此刻正在支持的东西,你去核它的概率会下降。** 三个实例、
  两个人、同一方向:checkpoint 的「连续七个生产者都选了 8」讨好了他正在审视的
  改动;他的「这个检查在首次应用时抓住了我」讨好了他正在背书的检查;**而我把第二条
  放大给全队和用户,讨好了我刚采纳的机制——我没有核它。**
  **操作准则:印证性的结果要按驳斥性结果的标准去核,尤其当它印证的是你刚写下的东西。**
  **推广(checkpoint 08:20):不只是「印证性的」,是「无人反对的」。** 他往 bug
  记录里写了九个「最后核对」时间戳,**一个都不是读出来的**——07:40 看了一次钟,
  之后全靠外推,最离谱的一个**快 66 分钟、落在未来**。而他是**在同一个文件的
  页眉里采纳「凡会过期的都要标日期」这条约定时**写下它们的。
  他自己的诊断:*「我全程手边就有一只钟,一条 `date -u` 的距离,而我没跑,
  因为没有任何东西在质疑那个数。」* **无人质疑的说法,享有和印证性说法同样的豁免。**
  补上的那一条:**日期本身必须是读出来的;一个编造的时间戳比没有时间戳更糟,
  因为它让「已核对」看起来是可查的。**

## 零和缺席

- **一个「零」到底是「没记录」还是「没发生」,不跑控制组分不出来。**
  踩过三次:`zone: NONE`(成功的任务也是 NONE,该字段根本不存)、
  `ps --ppid` 无子进程(几十分钟后才查,子进程早退了)、
  `ws_handoff_refine` 零写入(目录是只读挂载,再忙也是零)。
  **先测分母,再报分子。**
- **但分母只解决「空间」歧义,解决不了「时间」歧义。** 2026-09-05:节点上零容器,
  可能是「从未启动」也可能是「已经跑完并自行拆除」——**这两者是同一个零**。
  checkpoint 拿另一台节点做对照,正确地排除了「exec 命名空间假象」,但**任何跨
  节点对照都区分不了这两个,因为差别在时间里,不在读数里**。而我把「没有进程」
  推成「body 从未启动」,实际是它在 08:25:48 就自行拆除了,checkpoint 08:27 才看。
  **解决它的是一个带活动时间戳的产物**(`deploy.log` 里的 `started_at`)——
  **要「活动」的证据,不要「状态」的证据。**
- **一次成功的重试会毁掉自己的证据。**「重试之后就好了」和「它本来就没坏」,
  事后是同一个观察。*代价:我 07:05 撞上 spur 控制平面失败,重试通了,记成
  「暂时性」;09:45 同一个错误杀死了验收链——**第一个数据点在我手上两个半小时,
  被我当成了噪声。**「第二个实例把它从偶发变成一种条件」(m2)。*
  **这条是可机械化的(tier 2):重试必须留下被它跨过的那次失败的产物。**
  m1 的 NCCL 重试已经这么做了——它保留 `worker.attempt1-nccl-fail.log`,
  所以那四次故障是可数的;我的 `squeue` 重试什么都没留,所以第一次不可数。
  **但这个先例只覆盖三条通道里的一条(checkpoint):** `78909fc` 改的是
  `readme.md`——**是 brief 不是代码,所以它只到达 `kind: ai` 那条路径**;
  程序路径是空白;**而丢掉第一个实例的那次重试,既不是 ai 任务也不是程序任务,
  是一个操作者在敲命令——没有 brief 到得了那里,也没有代码路径能强制它。**
  「先例已经在包里」不等于「问题解决了」——**真正失败的那个案例,恰好在先例
  够不到的那条通道上。**(这是「brief 无法指示程序」的推广:**它也无法指示操作者。**)
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
- **agent 的 transcript 是一件我们谁都没用过的仪器。** m4 那条线「停滞 64 分钟」,
  被诊断成挂死、被拿去质疑停滞检测器、被我建议去查 `wchan`/`stack`——**而 agent
  早就结束了这一轮并写下了完整的拒绝理由,在 transcript 里躺了 64 分钟**:启动行
  少了 `--var gpu=`,`HIP_VISIBLE_DEVICES` 为空,而包里三处写明「body 不许自己
  改它」,**它认出自己就是那个 body,所以拒绝挑一张看起来空闲的卡**。
  **mtime 和进程表回答「它在动吗」;transcript 回答「它在做什么、为什么停」。**
- **`tail -1` 也会吞掉内容,而这一次更贵。** 我的十分钟巡检一整天都在跑
  `grep -aE 'phase|did NOT finish' $f | tail -1`。四条「死因不明」的运行,**真正的
  死因就写在最后一行的上面一行**——`"the escalation reached the top … Nothing has
  changed for 900 s"`——**而且它含 `phase`,我的 grep 匹配到了它,然后 `tail -1`
  把它扔了。** 日志在 `/home/yihou/` 里可读了一整天;没有任何东西不可恢复,
  **我们只是从没读过最后一行之前的那一行。** 取 `tail -3`,或者分开抓终止行。

## 不要从名字推断

- **容器名前缀不等于归属。** 我从 `yihou_e2e_flow_*` 推出「m1 在调试」,报给用户
  两次,实际是 run 自己的 agent。**判据是 FATAL 之后 5 秒就出现的容器,不是前缀。**
- **删掉一行之前,先读它上面那段注释。** 那条 `froms` 边看起来是多余的排序边,
  注释说明它保护着下游唯一可信的吞吐数字,而且删错了**不会有任何报错**。
  **把它当成候选之前也要读。** m2 花了一次双臂对比去确立「`/stop_profile` 的
  `ReadTimeout` 是症状不是故障」——**而调用点上方十二行的注释早就写着**:
  *「路由的 HTTP 客户端有 30 秒读超时……这里失败不构成『停止失败』的证据——
  下面那个字节数检查才是判据。」* 作者预见到了它,并为此建了那个检查。
  **他的对比确认了一个已文档化的设计决定,而不是发现了什么。**

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

The five stages used to be five *separate* packages, each driven to a real
cluster run on 2026-09-02. They were not a flow: **a handoff only travels inside
one run's graph**, so five packages are five runs and nothing chains. `e2e-flow/`
is the single package that joins them.

**This is a refine of definitions, not a rewrite of bodies.** The ~20k lines of
`.py`/`.sh` assets carried over are the only thing here that has ever produced a
number; they move and adapt, they do not get re-derived.

### 进度(2026-09-05 11:5x,读表得来,不是外推)

| 阶段 | 状态 |
|---|---|
| ① 造 handoff + 重构 validator | 完成 |
| ② 单独并行跑每个 mock | 完成 |
| ③ 串通 mock | **完成** — 3 m 46 s 走完五阶段,15 个 handoff,`run complete`(validator 关);开回 20 个后阶段 1、2 通过 |
| ④ 单独跑每个真实 | m1 ✅ m2 ✅ m3 ✅ / **m4 首次跑起 campaign 未完成** / **m5 从未跑过** |
| ⑤ 串通整个真实 | 最远 = mock 1–2 + 真实 3 完成 + 真实 4 进行中 |

**这张表会过期。改它之前先读运行日志,不要照抄。**

## Context — this environment (rows carry their own measurement date)

| | |
|---|---|
| where I am | login node, **no GPU**, no direct docker daemon |
| GPU holds | **Do not read a hold or a node IP from this file — run `squeue -u $USER` and resolve the IP from the node itself.** Holds cap at 8 h; every row here expires faster than the file is re-read. *2026-09-05: the two jobs this row used to name (`106250`/`crsuse2-m2m-061`, `106253`/`crsuse2-m2m-031`) were both long gone, and m1 took `10.245.159.129` from here and nearly pointed a real deploy at another tenant's host. "measured 2026-09-03" was honest labelling and did not prevent it.* |
| reaching them | `spur exec <jobid> bash -c '...'` — exec namespace; docker talks to the **host** daemon, but the **filesystem identity is you, not root** (measured 2026-09-04: `id -u` → `50112975`, writes land `-rw-r--r-- yihou ubuntu`). Matters because `/home` is `sec=sys` NFS, where a root-squashed write would map to `nobody` and leave a tree nobody can clean up. |
| shared FS | `/shared_nfs` 360 T, **46 T free**, shared by every spur node — this is how "remote" works |
| scratch | `/shared_nfs/yihou/agent_sys/ws_handoff_refine/` — **writable from a compute node only.** `/shared_nfs` is `ro,relatime` on the login node, so this path is not scratch for the leader; see the debugging section. |
| mock inputs | `/shared_nfs/yihou/agent_sys/cheat_for_mock/` — 25 real sealed handoffs, one folder per kind, **sealed 2026-09-02 and therefore older than this round's contract**: zero `environment.yaml` in 442 files, and the `operator_workset` has no `workset.yaml`. Do not delete it and do not hand-write the missing parts — graft today's real artefacts (see rule 10). Patched copy: `/home/yihou/cheat_for_mock.20260905T115337/`, originals kept beside each graft as `content.pre-0905`. |
| fast loop | `python3 -m agent_sys.cli.main show --package <dir> --var …` loads and type-checks every yaml in **< 1 s** |
| **4-minute loop** | **the whole five-stage graph runs on the login node with no GPU in under 4 minutes** — `--package e2e-flow-noval --var mock_stages=all` plus `MOCK_IMAGE_ID=sha256:…` as an **environment variable** (without it `mock_adapt.sh` cannot read the digest, exits 3, and the task is recorded `succeeded` anyway). Measured 2026-09-05: 3 m 46 s, 15 sealed handoffs, `run complete`. Every question that used to cost a GPU hold and forty minutes is asked here now. |

## Key references

- **`mission.md`** (repo root) — the authority. Every requirement below traces to
  a numbered item there.
- `/shared_nfs/yihou/agent_sys/cheat_for_mock/README.md` — **four things that
  will mislead you**, including a `kernel_table` that is a 34-row synthetic seed
  and an `integration_report` carrying a *refused* verdict.
- `/shared_nfs/yihou/agent_sys/debugging/integration/DELIVERY-NOTE-FROM-LEADER.md`
  — why that refusal was the validator working, and why the 5 % / 10 % bars must
  **not** be widened.
- `agent_sys/spec_loader/validate.py:34-56` — the `jsonschema` idiom to copy.
- `e2e-flow/RUN-PLAN.md` — **the canonical launch block. Open it FIRST, then diff
  your line against it.** Everyone who did it in the reverse order paid: it is the
  only place `--timeout 21600` appears, and `--var transport=spur` was missing
  from it until 2026-09-05 and cost m3 a run two hours in.
- `e2e-flow/WHAT-GREEN-ESTABLISHES.md` — which validators have ever refused
  anything, and what a PASS from each one is worth.

## Core principles

1. **Read the artefact, not the exit code.** Every acceptance claim names a file
   to open and a condition that fails.
2. **`items_schema` is not a schema layer.** Measured: for a file/tree item
   `handoff/content.py:184-197` validates the *filename string*, never the
   contents, and the schema is never exported to a body. Mission rule G2 needs
   real schemas under `assets/schemas/`, loaded by producer **and** validator.
3. **Every identifier bound on a shared host is a parameter.** Container names,
   ports, workdir, served model name. `: "${VAR:=…}"`, never `export VAR=`.
4. **Deletion — see standing rule 3 at the top of this file; that is the rule.**
   It is wider than the version that stood here until 2026-09-05: **every path on
   every host**, not just `/shared_nfs`, and `/tmp` is an allowed substring
   alongside `yihou`. `CONTRACT.md` §5.2 agrees.
   *This entry used to carry the pre-widening wording and therefore **permitted
   deletions the current rule forbids** — under `/home`, `/mnt/m2m_nobackup`, and
   inside container mounts. Found by readme-cn auditing this file 2026-09-05.*
   **Container removal is standing rule 1, not this entry**: `docker stop -t 10`,
   never `rm -f`, and "you did not create it" is **not** a reason to leave a
   foreign GPU user running — rule 1 retired that caution.
   Never `agent-sys run --clean` on a shared root — it removes *every* run.
5. **Research → gather → analyse → plan → work.** The repo receives only
   `e2e-flow/` and `todo.md`.
   **Scratch is `/home/yihou/`, not `/shared_nfs`.** `/shared_nfs` is
   `ro,relatime` on the login node (measured), so anything the leader writes goes
   under `/home/yihou/` — which also satisfies the deletion rule's `yihou`
   substring.
6. Bugs in `agent_sys` are recorded under
   `agent_sys/examples/llm_e2e_performance_optimization/temp/bugs/` first, then
   worked around; fixed only when the evidence is unambiguous.
   **`temp/` is gitignored while all its records are tracked** — a narrow
   `git add -f <one path>` is house style, not an override.
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
