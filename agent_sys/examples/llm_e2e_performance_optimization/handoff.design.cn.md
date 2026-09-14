# 新 handoff 设计报告

写给人看的。每一条断言后面都跟着一个可以打开的文件；凡是我没能在文件里核实的，
下面会写"未核实"而不是圆过去。

---

## 1. 为什么旧的五个包不是一条流水线

五个 stage 今天是五个**独立的 task package**（`deploy-demo`、`profiling-demo`、
`analyze-demo`、`kernel-opt-demo`、`integration-demo`），2026-09-02 各自在集群上真跑
过。它们加起来不是一条流水线，原因只有一条，而且是机制性的：

> **一个 handoff 只在一次 run 的 graph 内部流动。**

所以五个包就是五次 `agent-sys run`，中间什么都不接。上一个包的产物落到它自己的 run
目录里，下一个包启动时对它一无所知。要让 stage 1 的输出真的成为 stage 2 的输入，唯一
的办法是让它们在**同一张图**里，这就是 `e2e-flow/` 这个新包存在的全部理由。

不接的代价不是抽象的。五个包**并排写**（side-by-side authoring）而不是串起来写，
于是同一个概念在不同包里长成了不同的形状——这就是第 3 节那三处接缝。它们在五个包各自
运行时**一个都不会暴露**，因为每个包只看得见自己那一半。

规模：旧的五个包一共 **26 个 kind、24 个 validator**（`handoff.analysis.md` 的清单表：
1 + 7 + 6 + 2 + 10 个 kind，2 + 5 + 6 + 3 + 8 个 validator）。新包是 **15 个 kind、
21 个 validator**。删掉的每一项都对应 `mission.md` 的一个编号条目。

---

## 2. 新的 kind 列表

### 2.1 命名规则

`${mode}.${result_type}`——**点号左边是配置，不是角色名**。

这条规则针对的是一个具体的坏味道：旧包里叫 `baseline` 和 `profiled`。那是在讲一个
故事里的**角色**（"基准的那个"、"被 profile 的那个"），而两个 handoff 的真实区别是
**引擎的配置**：

- `profiling_mode_off` = profiler 脱离 + CUDA graph **打开**，这一组数字才有意义；
- `profiling_mode_on` = profiler 挂载 + CUDA graph **关闭**，因为一次 graph launch
  会把 profiler 存在的意义（看见那些 kernel）藏起来。

同样地，stage 5 的两臂是 `stock.` 和 `patched.`。名字里写配置，读者就不需要去别处查
"baseline 到底指哪一个"。

### 2.2 十五个 kind

全部列在 `main.yaml` 的 `handoffs:` 里——权限自上而下继承，**没列在那里的 kind，任何
子图都无法获得授权**，所以那一处清单是唯一的词汇表。每个 kind 的**定义**写在它生产者
所在的 `steps/*.yaml` 里，`main.yaml` 只引用。

| # | kind | content_type | producer | consumers |
|---|---|---|---|---|
| 1 | `deploy_kit` | `code` | m1 `deploy_and_prove` | m2 `run_profiling_mode_off`、`run_profiling_mode_on`；m3 `identify`、`build_workset`；m4 `optimize_kernel`；m5 `apply_patch`、`integrate_and_verify`、`packup` |
| 2 | `profiling_mode_off.bench_result` | `reproducible` | m2 `run_profiling_mode_off` | m2 `merge_profiling_evidence` |
| 3 | `profiling_mode_on.bench_result` | `reproducible` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` |
| 4 | `profiling_mode_on.profile_result` | `reproducible` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` |
| 5 | `profiling_mode_on.kernel_table` | `structured_text` | m2 `run_profiling_mode_on` | m2 `merge_profiling_evidence` |
| 6 | `profiling_evidence` | `reproducible` | m2 `merge_profiling_evidence` | m3 `rank`、`build_workset`；m5 `integrate_and_verify`、`packup` |
| 7 | `kernel_worklist` | `structured_text` | m3 `rank` | m3 `identify` |
| 8 | `operator_identity` | `structured_text` | m3 `identify` | m3 `build_workset` |
| 9 | `operator_workset` | `code` | m3 `build_workset` | m4 `optimize_kernel`；m5 `apply_patch`、`packup` |
| 10 | `kernel_optimization` | `code` | m4 `optimize_kernel` | m5 `apply_patch`、`integrate_and_verify`、`packup` |
| 11 | `patch_overlay` | `reproducible` | m5 `apply_patch` | m5 `integrate_and_verify`、`packup` |
| 12 | `stock.measurement` | `reproducible` | m5 `integrate_and_verify` | m5 `packup` |
| 13 | `patched.measurement` | `reproducible` | m5 `integrate_and_verify` | m5 `packup` |
| 14 | `integration_report` | `structured_text` | m5 `integrate_and_verify` | m5 `packup` |
| 15 | `e2e_packup` | `code` | m5 `packup` | 无——`main.task.outputs`，整条流程唯一的对外产物 |

两个读起来会误会的地方：

- **`deploy_kit` 一路传到 m5，不只是因为它带 environment 记录**，还因为 m3 和 m4 必须
  **在 m1 拉起来的那个容器里跑东西**（workset 自己的测试、KernelForge）。
- **全流程没有任何 `serve_*` task，也没有 `deployment_*` handoff。** M2.5 / M5.2 禁止
  "agent A 去部署、agent B 去使用"，所以需要服务的 task 自己在 STEPS 里拉起、用完拆掉。
  单这一条相对五个 demo 就删掉了 5 个 task 和 8 个 handoff kind。

---

## 3. 三处接缝，各自是怎么消掉的

三处都记在 `handoff.analysis.md` 的"Three seams that will bite when the flow is
stitched"。它们的共同点是：**在五个包各自运行时都不是 bug**，因为解析是按文件来的，
每个包只看见自己那一份。**它们只在拼起来的那一刻才成为问题。**

### 接缝一：`kernel_table` 一个名字、两个 content_type

- stage 2 把它封成 **`reproducible`**（五个标准 item），实测是一张**真实的 124 行**
  kernel 表；
- stage 3 消费的是一个 **`structured_text`**（`text.json` + CSV），实测是一份
  **34 行的合成种子**。

也就是说 **stage 2 的产物按原样根本喂不进 stage 3 的消费者**。今天只有"每个包有各自
的 root"这一件事挡着它们相撞。更糟的是这两份东西在 mock 语料里同名，读者会以为是同一
件东西的两个版本，其实是两件不同的东西，而合成种子那一份更容易误导人。

**怎么消掉的**：只留一个 kind——`profiling_mode_on.kernel_table`，`structured_text`，
由 m2 生产、m3 消费。合成种子（旧的 `seed_table`）**整个删掉**（M3.2）：它当初存在的
唯一理由是"stage 2 的真实输出拿不到"，现在拿得到了。

配套的是 M3.5：**`check_kernel_table` 只定义一次**，放在 `steps/common.yaml`，m2 当
生产侧、m3 当输入侧共用。旧的两个包各带一份拷贝，`args` 不同、`min_rows` 不同。

### 接缝二："workset" 横跨两个 kind

- stage 3 的 `operator_workset`：`reproducible`、**多算子**，要求 10 个文件（含
  `invocation_spec.json`、`forge_task.yaml`）；
- stage 4 的 `workset`：`code`、**单算子**，要求 7 个文件（含 `integration.md`）。

必需文件清单不一样，content_type 不一样，粒度不一样。3 → 4 之间需要一次**显式的格式
转换**，而那次转换在旧包里就是 `publish_workset` 这个 task。

**怎么消掉的**：M3.7 / M4.1 把两个 kind **合并成一个**——m3 的 `operator_workset`，
`content_type: code`。于是 `publish_workset` 这个 task 也没有存在意义了，一并删掉
（M4.2）：输入直接就是 m3 的产物，图这一层没有东西需要转换。

再配上 M4.4：**workset 相关的 validator 全部归到 m3 的输出上**，m4 一个都不声明，
只引用 kind 并继承 `check_workset_shape` 和 `check_workset_runs`。

### 接缝三：两个 validator 共用一个名字

原文其实点了**两组**，不只一组：**`check_service_live` 和 `check_packup_shape` 各自
是两个共用一个名字的 validator**。两个 `check_service_live` 差在
`expect_cuda_graph`；两个 `check_packup_shape` 的行数下限差了大约 **2 倍**。

"解析是按文件来的，所以其实没坏"——但**读者会以为只有一个**，而实际有两个。

**怎么消掉的**：`check_service_live` 直接**不存在了**（M2.8.1）。它评判的是
`deployment_*` handoff，而那类 handoff 随着"不许拆分部署与使用"一起被删掉了——**它评
判的对象没有了，validator 自然也就没有了。** `check_packup_shape` 在新包里只有一个
定义，在 m5。

顺带记下 `handoff.analysis.md` 记录但**没有修**的那个缺口：`check_service_live` 证明
的是一个部署**活着**，不是它与另一臂**可比**。stage 5 那次 −21% 的假回归就是这个缺口
的代价。新包里这个缺口仍然开着，编号 `todo.md` T7。

---

## 4. `environment` 记录如何贯穿全程

### 4.1 它是被"提升"的，不是新发明的

**mission M1.2.1 要的那份记录今天已经躺在 sealed handoff 里了**，只是没有 schema：
`content/items/env/deployment.json` **和 `context.json`**。CONTRACT §2.1 做的事情是
取这两者的**并集**，按 mission 的要求拆成两半，再给它一个 schema
（`assets/schemas/environment.schema.json`）。

拆成的两半是：

- **`fixed`**（M1.2.1.1，可固化环境）：`node`、`node_ip`、`gpu_arch`、`gpu_count`、
  `image`、`image_id`、`dockerfile`、`rocm`、`model_name`、`model_path`、`tp_size`、
  `scripts`。
  `image_id` 和 `image` 都要，因为**一个浮动的 tag 不是一次复现**——一个写着
  `:latest` 的 kit 记录的是"那天那台机器上碰巧是什么"。
- **`runtime`**（M1.2.1.2，哪台机器的哪个 docker container）：`slurm_jobid`、
  `container`、`ports`、`endpoint`、`transport`、`started_at`。

### 4.2 一份文档、一个 schema、三个位置

G5 的原话是*"整个流程的 handoff 都需要传递 env"*。十五个 kind **全部**带这份文档，
但落点随 content_type 变：

| content_type | `environment.yaml` 的位置 |
|---|---|
| `reproducible` | `items/env/environment.yaml`（`env` 本来就是必需 item） |
| `code` | `items/codes/environment.yaml` |
| `structured_text` | `items/env/environment.yaml`，`env` 要在 kind 的 `items_schema` 里声明 |

**是同一份文档、同一个 schema。** validator 只需要挑对目录，不需要知道自己在看哪种
content type——`check_environment` 就是按找到的 content type 派发的，所以某个 kind 换了
类型不会导致它悄悄不再被检查。

### 4.3 `fixed` 严格比对，`runtime` 只报告不判定

`check_environment` 跨输入比对 `[node, gpu_arch, image_id, model_path]` 四个字段——这
四个是"两个 handoff 到底可不可比"的最低条件，不一致就意味着它们描述的是两台机器，跨着
它们算出来的每个数字都作废。

`runtime` **只报告，从不判定**，理由具体：**m5 的两臂按构造就有不同的容器**。一个容器
一生只持有一个状态，这正是两臂设计存在的理由（CONTRACT §5，mission G5.1 授权的例外）。
一个 `strong` validator 如果在这里失败，就会恰好停在契约规定必须不同的那个 stage 上。

### 4.4 一条被推翻的继承规则

旧包的 validator 继承了一条"handoff 里不许有绝对主机路径"的规则，理由是*"封存时会因为
一条路径拒绝整个交付"*。**m3 对着框架实测：这个前提是假的。**
`handoff/store.py:447` 写着 `# locality.check — NOT CALLED`，`:494` 给了原因——那个形状
启发式把一行 HTTP access log 读成了文件系统路径，在一个真实 kit 上**误报率 97%**。

这件事在这里是硬性的，不只是措辞问题：`environment.schema.json` **要求**
`model_path`，而它天然就是 `/shared_nfs/yihou/models/...` 这样的绝对路径。把那条规则
原样搬过来的 validator **会拒掉本包里每一个合规的 handoff**——m3 就是这么在 fixture 上
发现的。

结论：规则按它自己的道理保留（可移植性——一个写死某台主机目录的脚本换台机器就跑不了），
但**范围缩到可执行与生成内容（`.py`/`.sh`/`.json`/`.jsonl`），跳过 environment 记录**，
并且**不要再用"封存会拒绝"当理由**。

---

## 5. Schema 层

### 5.1 它不是框架已有能力的重复

这一点必须讲清楚，否则整个 `assets/schemas/` 看起来像是在重造轮子。

框架的 `items_schema` **不校验文件内容**。`handoff/content.py:184-197` 构造的是

```python
instance = {
    k: (item.value if item.kind == "data" else str(item.path.name))
    for k, item in content.items.items()
}
```

——对于一个 file 或 tree item，被拿去校验的值是 **`item.path.name`，也就是文件名这个
字符串**。文件里有什么，从头到尾没有被读过。

它是封存边界上的一道**准入检查**（`store.py:448,501`），**从不导出给任何 body**，而且
旧的五个 demo 里**没有任何一个 validator import 过 `jsonschema`**——全部是手写的
（`analyze-demo/…/check_workset_shape.validator/check.py:96`）。

所以 G2 要的那一层（*"所有结构化的文档，尽量有自己的 json schema，该 schema 同时暴露给
producer & validator"*）**框架没有提供**，本包自己带。

### 5.2 布局

`assets/schemas/` 下 8 个 JSON Schema + 1 个 layout：`environment`、`bench_result`、
`kernel_table`、`kernel_worklist`、`operator_identity`、`workset`、
`kernel_optimization`、`integration_report`，外加 `deploy_kit.layout.yaml`
（M1.1 要的文件/目录布局规格，不是 JSON Schema）。加载器是
`assets/lib/schema.py`，198 行，producer 和 validator **import 的是同一个文件**。

`jsonschema>=4.18` 本来就是 agent_sys 的声明依赖（`agent_sys/pyproject.toml:37`），
写法照抄 `agent_sys/spec_loader/validate.py:34-56`：`Draft202012Validator` 加一个
`referencing` registry，让 schema 之间可以 `$ref`。

### 5.3 两个环境变量都必须写，以及为什么

解析规则只有一行，**没有回退到私有拷贝这条路**：

```
<AGENT_SYS_TASK_PACKAGE | AGENT_SYS_DEMO_PACKAGE>/assets/schemas/<name>.schema.json
```

**两个变量都要写，永远。** 理由在 `schema.py` 的 docstring 里，是一句实测：

> **一个 validator 的 *input* 阶段拿到的是 GLOBAL 环境行，永远没有
> `AGENT_SYS_TASK_PACKAGE`；只有 PRODUCER 行才导出它。**

所以一个只读其中一个名字的 body，**在手工测试里是好的，在 phase 里是坏的**——这已经
代价过一次运行。

（准确说，`package_root()` 之后还有第三条路：body 在 zone 之外被跑起来时——比如开发者
在 shell 里直接调——用这个文件自身的位置解析，因为这个文件本身就在包里。那是给人用的
便利，不是给 phase 用的回退路径。）

### 5.4 与之配套的第二条环境规则

同一节还记着一条同源的坑：**要跑
`"${AGENT_SYS_DEMO_PYTHON:-python3}"`，绝不要裸 `python3`。**
`cli/main.py:668` 导出的是这次 run 自己在用的解释器。一个 validation zone 拿到的是
policy 派生的 `PATH`，其上 `python3` 解析到 `/usr/bin/python3`——那个解释器**没有
`referencing`**，因而 import 不了 `assets/lib/schema.py`。

实测：body 在**写出 `verdict.json` 之前**就 `ModuleNotFoundError` 死掉，phase 报的是
*"nothing was decided"* 而不是一个判定——**一个起不来的 validator，看起来和一个从没被
问过的 validator 一模一样。** m5 把自己的 leaf 逐个跑过图之后发现，21 个 validator 里
**12 个有这个毛病，横跨全部五个模块**。

### 5.5 `structured_text` 自带一份 schema 拷贝

M3.6 还要求 `structured_text` 的 handoff **把 schema 拷一份带在身上**
（`items/schema`），生产时从 `assets/schemas/` 拷；validator 检查这份拷贝与包里那份
**逐字节相同**（CONTRACT §3.4）。这样产物既是自描述的，又**可证明不是一个私有分叉**
——只有前者的话，一件产物完全可以自描述地描述着另一样东西。

---

## 6. 我拒绝写进去的句子

lead 给我的 brief 里有两处，我在文件里核不到，或者核到的与其表述不同。按要求列出来，
**没有按原话写**：

1. **"它今天已经躺在 sealed handoff 的 `content/items/env/deployment.json` 里"**——
   只对了一半。`CONTRACT.md` §2.1 的原文是 `deployment.json` **和 `context.json`**，
   并且明确说这份契约取的是**两者的并集**（*"takes their union"*）。只点 `deployment.json`
   会让读者以为提升的是一个文件的重新包装，而实际是两个文件的合并再拆分。我按并集写。

2. **"两个同名的 `check_service_live`"**——`handoff.analysis.md` 那条接缝点的是**两组**
   而不是一组：*"`check_service_live` and `check_packup_shape` are each two validators
   sharing a name"*。两个 `check_packup_shape` 的行数下限差约 2 倍。少写一组会让这处
   接缝看起来比实际小。我两组都写了。

另外两处是精度问题，不是错误，我按文件写而没按 brief 写：

3. brief 说 **"两条 fallback 路径"**。`schema.py` 里是两个**环境变量**，之后还有第三条
   基于文件自身位置的解析路径。我把第三条标成"给人用的便利，不是给 phase 用的回退"，
   因为把它算成第三条 fallback 会让人以为 phase 里也有兜底，而那正是这一节要防的误解。

4. brief 说 CONTRACT 包含 **§5.2.0**——**存在**，我核过了
   （`CONTRACT.md:1076`，标题是 "ON EVERY NODE WE HOLD, KILL EVERY GPU USER THAT IS
   NOT OURS"）。§8a 也存在（`:1278`）。但这两节讲的是集群纪律和多人共用一个 worktree
   的提交规则，**不是 handoff 设计**，所以本报告没有展开它们——不是因为核不到。
