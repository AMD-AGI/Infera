# `e2e-flow` 包 bug 记录 — 2026-09-06

生成于 **2026-09-06T06:56:50Z**(`NOW=$(date -u +%FT%TZ)` 实读后引用,不是手写的)。
集群:`smci355-ccs-aus-n04-25`,8×MI355X SPX,slurm 29184。

> 本文件是**追加**的,不是重新生成的。第 1 条由 m2 写入;后来的人请在末尾追加,
> 不要重排已有条目。

---

## 1. 代码提供了一个旋钮,而包从不接线 —— 一个成因,至今 11 个实例

**这是一类,不是一串意外。** 分开记会让它们看起来是 11 件事,而第一轮里
**一个成因被当成三次不同的意外发现过三遍**,所以它一直是三个条目。

### 形状

一个 body 读 `${E2E_FOO:-default}`(或 `os.environ.get("E2E_FOO", default)`),
而 **`shared.yaml` 和 `steps/*.yaml` 都没有声明 `E2E_FOO`**。
包声明变量只有这两个地方,所以**没有任何 `--var` 到得了它**,
那个 default 就是它这辈子唯一能取的值。

**为什么它不像 bug:** 代码读起来完全像一个可配置项——有默认值、有 `:-`、
甚至有注释解释该怎么调。**缺的是接线,而缺席是不可见的**:
`grep E2E_FOO` 在 `assets/` 里有命中,一切看起来正常。

**和「有默认值的出处字段」(2026-09-05 第 x 条)同族,但更窄:**
那一条是「没人想起来传」,这一条是**传不进去**。

### 两个有后果的实例

**(a) `E2E_EVAL_THINKING` —— 本轮就会咬人,而且它报的是一个数字,不是一个错误。**

```
assets/accept/measure.sh:193   THINKING='${E2E_EVAL_THINKING:---thinking-mode glm-45}'
assets/accept/measure.sh:265   "thinking_mode": "${E2E_EVAL_THINKING:---thinking-mode glm-45}"
```

`lm_eval.sh` 自己的注释写明了后果:*「Without the matching
chat_template_kwargs a healthy deployment scores 0.00.」*
**本集群服务的是 Qwen3,不是 GLM。** 部署是健康的,eval 会跑完,每一个请求都成功,
`content` 全空,分数落地。**没有任何东西会失败。**
`lm_eval.sh:47` 接受 `none` 并把它变成空 flag,所以修法存在——只是够不着。
→ **已交给 m35 在 `m5_integration.yaml` 里暴露为 `--var eval_thinking`,
并保留当前 `glm-45` 行为作为默认**(只暴露、不挪默认,是唯一不会破坏第一个集群
复现的改法)。

**(b) `E2E_SGLANG_SRC` / `E2E_AITER_SRC` —— m3 的 `identify` 找不到源码,而它不会拒绝。**

```
assets/identify.task/identify.py:696
    repos = [r for r in (os.environ.get("E2E_SGLANG_SRC"),
                         os.environ.get("E2E_AITER_SRC")) if r]
:699    finder = kernel_finder(magpie_root, [...], repos, timeout)
```

两个都没接线 → `repos` 恒为空列表。同一行上方的 `E2E_MAGPIE_ROOT` **是**
一个 `--var`(`magpie_root`),但它的默认 `/shared_nfs/chaox/Magpie`
**在本集群不存在**,而 `os.environ.get(...) or "/nonexistent"` 会把缺失变成一个
不存在的路径而不是一次中止。
**于是 `identify` 在没有任何源码可查的情况下产出 `operator_identity`,
而 `min_resolve_ratio` 默认 `0.0`,所以它不会因为「什么都没解析出来」而拒绝。**
→ **未上报为待修**,因为本轮 m3 尚未真跑到这里;记在这里,
**这样它在 m3 的 worklist 空掉的那一刻是一次查表,而不是一次重新发现。**

**(c) `E2E_STOCK_VS_M2_TOLERANCE` —— 这一类里最难看见的一个:旋钮**存在**、
**能传**、**传了完全没反应,而且没有任何观察量会不同。**(m35,2026-09-06)

前两个实例是「变量读不到」。这一个是**读得到的那个变量被另一个读不到的变量压过去**:

```
steps/m5_integration.yaml:412        stock_vs_m2_tolerance: '${stock_vs_m2_tolerance:-0.10}'   ← 只在 validator 的 args 里
check_no_regression/check.py:445     tolerance = block.get("tolerance")        ← 优先取生产者写的
check_no_regression/check.py:446     if tolerance is None: tolerance = args.get("stock_vs_m2_tolerance", 0.10)
compare.py:310-311                   default=float(os.environ.get("E2E_STOCK_VS_M2_TOLERANCE", 0.10))
integrate_and_verify.task/readme.md:284-292   规范调用**不传** --stock-vs-m2-tolerance
```

`E2E_STOCK_VS_M2_TOLERANCE` 在本节的 sweep 里就在不可达一侧(已核,不是我另数的)。
于是生产者永远写 `tolerance: 0.10`,validator 永远用生产者的 `0.10`,
而 `--var` 自己的默认**也是** `0.10`。

> **三个 0.10 相等,所以调这个 `--var` 什么都不会变,而操作者会得出
> 「这个测量很顽固」而不是「这个旋钮没接线」。**
> 和 `gpu_devices` 第一个集群那次同形:**错的答案和对的答案是同一个数字。**

再补一刀:`:446` 那个回退**在实践中够不到**——三条会省略 `tolerance` 的生产者路径
(`compare.py:142,148,156`)全都是 `ok: null` 路径,在 `:445` 之前就 return 了。

→ **不修,也不放宽。** PRE-REGISTER 在任何结果到达之前把 10% 定死了,那条约束成立。
记在这里的价值是:**如果将来真要动这个门槛,必须动在生产者那一侧——那里改动是可见的**,
而不是动一个静默无效的 `--var`。

**这个子形状 sweep 抓不到,值得单列:** 工具能找出 `E2E_STOCK_VS_M2_TOLERANCE`
读得到、没声明;它**看不出旁边那个已声明、看起来正常的 `--var` 因此是失效的**。
判据要多一步:**一个已声明的 `--var`,如果消费者在用它之前先读一个生产者写的同名字段,
那么真正的接线点在生产者那一侧,而不是在声明它的地方。**

### 无后果的,记录但不花钱

| 变量 | 读处 | 为什么不动它 |
|---|---|---|
| `E2E_RESOLVE_TIMEOUT_S` | `identify.task/entry.sh:22` | `identify` 是秒级的容器探针,1800 s 是天花板,没人会撞到 |
| `E2E_RUN_TAG` | `serve/mix_up.sh:103` | 默认回落到 `$CTR`,label 仍然可归属 |
| `E2E_PROBE_SCRATCH` | `lib/controls/packup_probe.py:81` | 工具自己的 scratch,默认 `tempfile.gettempdir()` |
| `E2E_SWEEP_SCRATCH` | `lib/interpreter_sweep.py:100` | 同上 |
| `E2E_PACKUP_MOCK` | `lib/mock_m5.sh:326` | 默认指向第一个集群的 `/shared_nfs/...`,本集群不存在——**但只在 m5 走 mock packup 时才到达,本轮不走**。**收窄一句(m35,已读代码):到达时它不是「降级」,是 `mock_m5.sh:332` 的 `exit 1`,而 `packup.task/entry.sh:13` 的 `rc != 3` 分支把它原样抛出——`packup.py` 根本不会被调用。`E2E_MOCK_STAGES` 默认是 `all`,所以「本轮不走」是**发车行的选择**,不是默认行为:`mock_stages` 必须显式排除 m5。** |

### 三个阈值:接线缺失是**有意**的,不要「修」它

```
assets/compare.py:307  --max-throughput-regression  default=env E2E_MAX_THROUGHPUT_REGRESSION or 0.05
assets/compare.py:309  --max-latency-regression     default=env E2E_MAX_TTFT_REGRESSION       or 0.10
assets/compare.py:311  --stock-vs-m2-tolerance      default=env E2E_STOCK_VS_M2_TOLERANCE     or 0.10
```

`compare.py:300-305` 明写 *「These are not to be widened」*,并记着上一轮把它们
放宽到 0.35/0.30 是错的。**所以这三个 env 名字没接线是特性而不是缺陷**,
不要把它们加进 `steps/*.yaml`。

**但其中一个有一个值得核的裂缝(我只核到一半,标明边界):**
m5 的 agent 调 `compare.py` 时**一个阈值 flag 都没传**——

```
assets/integrate_and_verify.task/readme.md:285-291
  python3 .../compare.py --stock … --out … （没有 --stock-vs-m2-tolerance）
```

而 `m5_integration.yaml:412` 把 `stock_vs_m2_tolerance: '${stock_vs_m2_tolerance:-0.10}'`
喂给 **`check_no_regression` 这个 validator**,`:409-411` 的注释说
*「The tolerance stays a `--var` because a tolerance is a judgement」*。
→ **`--var stock_vs_m2_tolerance` 会移动 validator 的杆,却不会移动
`compare.py` 写进报告里的那个声明值。** 两者一松一紧就可能产生一次
「读起来像生产者缺陷」的拒绝。
**我没有核到底:** `compare.py:303` 和 `m5_integration.yaml:409` 两处注释都说
`check_no_regression` 会拒绝「声明的杆比自己松」的报告,并点名了
`check.py:249-267`——**我读了那一段,它是逐行重算 pooled mean 的分支,不是比杆的
分支。两条注释不是一次阅读**,所以这条留作**未验证**,而不是结论。

### 找出其余实例的方法(可复跑,不靠记忆)

```sh
python3 /data/yihou/e2e_verify_20260906/m2/tools/sweep_unwired_env.py \
        agent_sys/examples/llm_e2e_performance_optimization/e2e-flow
```

判据分两步,**第二步是必须的**:

1. `assets/` 里读到的 `E2E_*`,减去 `shared.yaml` + `steps/*.yaml` 里声明的;
2. **再减去 `assets/` 自己在运行时供给的**——`E2E_OUTPUT_*` 由框架导出、
   `E2E_KIT_GPU_DEVICES` 由 `load/line.sh:179` export、`E2E_ARM`/`E2E_MODE`
   由调用方 `on "E2E_ARM=… bash x.sh"` 内联传入。
   **只做第一步会把 24 个正常变量报成缺陷**,那样这个工具本身就成了它要抓的东西。

本次:读到 95 个,声明 58 个,未声明 37 个,**运行时供给 24 个,不可达 13 个**。

**13 里有 2 个是工具的假阳性,我数进去了又减出来,写在这里免得下一个人重新发现:**
`E2E_PORT_`(来自 `.md` 里的 `$E2E_PORT_*` 通配写法)和
`E2E_X`(来自 `check_agent_env.py:68` 一句**讲解这些模式本身**的注释)。
**真实数目是 11。** —— 这正是「grep 的命中不是证据,必须读那一处在说什么」:
两次命中都长得像模像样。

**工具本身跑过已知答案再用:** `E2E_EVAL_THINKING` 和 `E2E_RESOLVE_TIMEOUT_S`
必须出现在不可达一侧(出现了),`E2E_AIPERF_TRACE` 必须两侧都不出现(已声明,
没出现),`E2E_KIT_GPU_DEVICES` 必须出现在被过滤一侧(出现了)。
**四个已知答案先跑,再看新答案。**

**追加(m35,2026-09-06 07:0x):这把 sweep 现在也是「修好了没有」的量具,而且是别人写的量具。**
我在 `m5_integration.yaml` 里给 `E2E_EVAL_THINKING` 接了线(默认逐字保持
`--thinking-mode glm-45`),然后对**修改前的包副本**和当前包各跑一次这把 sweep:

```
before (HEAD 的 m5_integration.yaml)   UNREACHABLE 13   含 E2E_EVAL_THINKING
after  (接线之后)                       UNREACHABLE 12   不含
```

差一行,就是那一行。**上面的「11」是减掉两个假阳性之后的数,那次统计发生在这次接线之前;
接线之后同样口径是 10。我没有去改标题里的数字**——那是另一个人的统计,
而重新生成一份共享文件里的账,是这份文件自己记过的破坏性操作。

**另外两条方法上的补充:**

1. **我自己第一次搜这个旋钮时用的是小写模式,而环境变量名是大写的,所以我差点漏掉。**
   这把 sweep 用的是 `E2E_[A-Z_]+` 所以不受影响,但**手查时判据必须是大小写不敏感**:
   `--var` 是小写、env 是大写、两者在不同文件里,任何单一大小写的 grep 都会漏掉一半。
2. **不命中才是证据,命中什么都不是。** 我核 `pip install` 那次,六个命中里五个是
   precondition 的 `err:` **字符串**,一个是注释——**真正让「没有」变成证据的,
   是 `find . -type f` 减去被 grep 的后缀之后返回空**,也就是覆盖面本身可查。
   这和本节「工具本身跑过已知答案再用」是同一条,只是从另一头说。

---

## 2. 一个变量在一条路径上失效、在另一条上生效 —— `gpu_devices`

第一个集群把 `--var gpu_devices` 记成「失效」,连吃五次发车。**在本包里它一半一半,
所以两个方向的概括都是错的。**

**m5 的 serve 路径:失效,而且发车行够不到。不是一行,是一处缺席——
所以 `grep GPUS` 找不到它:**

```
serve/mix_up.sh:106      docker run … --device=/dev/kfd --device=/dev/dri   （没有 -e GPUS）
serve/mix_up.sh:157-165  docker exec -d "$CTR" env MY_IP=… TP="$TP" … bash /mix_worker.sh
                         ^ 一个封闭的显式 env 列表，GPUS 不在里面
serve/mix_worker.sh:26   GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"    → 永远取默认
serve/mix_worker.sh:100  HIP_VISIBLE_DEVICES="$GPUS"
```

链条里唯一的假设是「除非镜像自己设了 `GPUS`」,**这一步是核过的而不是带过的**:
`docker inspect -f '{{json .Config.Env}}'` 显示引擎镜像**没有**定义 `GPUS`、
`HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`。
→ **m5 的一条臂永远吃 `0..TP-1`。本机 TP=8 时那就是全部八张卡,
和任何人会要的完全一样——错的答案和对的答案又是同一组数字。**
可带走的后果不是这个 flag,而是:**本节点上 m5 的臂无法与任何人共存。**

**m2 的 load 路径:接通的。** `load/line.sh` 根本不走 `mix_up.sh`,
它跑的是 **kit 自己的** `scripts/deploy.sh`,并在 `:169-184` 用一个有文档的
三分支优先级显式映射这个变量,**而且把选中的分支按名字打进日志**:

```
cards: <list> (from --var gpu_devices)                  分支 1
cards: <list> (the set the deploy_kit records taking)   分支 2  ← 默认，且是被论证过的那个
cards: not named here; the kit's own default applies    分支 3  ← 出现即是发现
```

剩下的一半诚实地开着:kit 自己的 `deploy.sh` 认不认 `E2E_KIT_GPU_DEVICES`,
是 **m1 的 kit** 的性质,而它此刻还不存在。**未验证,不是已验证。**
`line.sh:154` 记着 kit 的默认是硬编码字面量 `0,1,2,3`,而且*「nothing in the kit
reads a card」*——所以分支 3 要当成发现来读。

---

## 3. 同一个变量,两个不同的默认值 —— `trace_end_ms`

```
shared.yaml:149              E2E_TRACE_END_MS: '${trace_end_ms:-180000}'
steps/m5_integration.yaml:126 E2E_TRACE_END_MS: '${trace_end_ms:-60000}'
```

**一个变量名,两个默认,差三倍,没有任何东西会警告。** m2 的 replay 窗口和 m5 的
bench 窗口因此可以在同一次运行里静默地不一致。
→ 处置:**两处都显式传 `--var trace_end_ms=60000`**,让哪个默认都不做决定。
待定的一点已路由给 m35:`stock_vs_m2` 比较的量是否对窗口长度敏感——
如果敏感,两者必须等于 **m2 实际跑的那个值**,那就是 m5 迁就 m2 而不是反过来。

---

## 4. 一条**写进方法学的缓解措施**,在我们实际敲命令的那个 shell 里是坏的 —— `PIPESTATUS`

追加于 **2026-09-06T07:05:12Z**(`date -u` 实读)。m35 发现,leader 复现,我在本机第三次复现并
多测出两条。

> **这不是第 1 条那一类。** 第 1 条是「代码提供了旋钮而包不接线」。
> 这一条是「**我们反复教给彼此的那条修法,在我们打字的那个 shell 里返回空**」。
> 它必须进仓库而不是只留在消息里,因为它已经写进 `mission.verify.e2e.md` 的方法
> 段落和 m35 的检查清单——**消息可以撤回,文件会一直教下去。**

### 现象(本机实测,`/bin/zsh` 5.8.1)

```
$ false | head -1
PIPESTATUS[0]=''      pipestatus[1]='1'
```

zsh 数组**从 1 开始**,而且活的那个是**小写** `pipestatus`;大写 `PIPESTATUS`
在 zsh 里根本不存在。

### 危险的地方不是「空」,是**空会被读成成功**

一个谨慎的人会加 `:-0` 兜底。**那正是让它变静默的那一步:**

```
$ false | head -1
$ [ "${PIPESTATUS[0]:-0}" = 0 ] && echo "succeeded"
succeeded          ← 命令退出码是 1
```

**于是「`| head`/`| tail` 会吞掉退出码」的那条缓解措施,自己带着被吞掉的退出码。**

### 我多测出来的两条(leader 的报告里没有,而且第一条更容易咬人)

**(a) `pipestatus` 会被**紧接着的下一条命令**清掉,而且清成 `0`。**

```
false | head -1
st=${pipestatus[1]}                 → st=1     （紧接着取，对）

false | head -1
echo "...任何一条命令..."
echo "${pipestatus[1]}"             → 0        ← 那是 echo 的状态，不是管道的
```

**清成 0,也就是清成「成功」。** 所以
`cmd | tail; echo done; echo "rc=${pipestatus[1]}"` 是静默错的,
而 `cmd | tail; echo "rc=${pipestatus[1]}"` 是对的——**差别只在中间有没有别的命令**,
这是一个没人会在 review 里看出来的区别。

**(b) 有一个不依赖数组、也不依赖取值时机的写法,两个 shell 都对:**

```
setopt pipefail   # 或 set -o pipefail，bash/zsh 都接受
false | head -1
echo $?           → 1
```

### 修法,按优先级

1. **最好:不要把你需要状态的那条命令放进管道。** 重定向到文件,再 `$?`,
   要看输出就之后 `tail` 那个文件。(`m2_show.sh` 现在就是这么写的,
   而且注释里写明了为什么——免得有人「简化」回去。)
2. `set -o pipefail` + `$?`:两个 shell 都对,不依赖数组下标或取值时机。
   代价是它只说「管道里有东西失败了」,不说是哪一段。
3. 非要取数组:交互式/zsh 用 **`${pipestatus[1]}`**,
   `bash -c` 里或 `#!/usr/bin/env bash` 脚本里用 `${PIPESTATUS[0]}`,
   **而且必须紧接着管道取,中间不能有任何命令。**

### 与本文件其它条目同族

和「`sh` 跑 `set -o pipefail` 报 Illegal option」「`bash -n` 只验语法」
「`dash -n` 不可能在非法选项上失败」是同一族:**仪器在回答另一个问题,
而它的输出格式完好。** 区别是这一条**藏在缓解措施里**,所以它专门咬那些
已经知道原始陷阱、并且照做了的人。

### 我自己的用例审计(2026-09-06,逐条查过,不是假定)

| 用例 | 受影响? | 结论是否依赖它 |
|---|---|---|
| 第一次 `show`:`… \| tail -30; echo "SHOW_RC=${PIPESTATUS[0]}"` | **是**,打出空 | 否——当场发现是空,改用 bash 脚本 |
| `aiperf validate` 负控制:`… \| tail -6; echo "rc=${PIPESTATUS[0]}"` | **是**,打出空 | 否——判据是输出文本 `Validation failed` / `passed` |
| `m2_show.sh` 里的 `rc=$?` | 否——**没有管道**,重定向到文件 | 文档里所有 `rc=0`/`rc=1` 都出自这里,成立 |
| 镜像内 `aiperf validate …; echo "nc_rc=$?"` | 否——在 `bash -c` 里,取值处没有管道 | `nc_rc=1` / `real_rc=0` 成立 |
| `docker images \| grep -i aiperf; echo "rc=$?"` | 否——grep 是管道**最后**一段,`$?` 就是它 | 成立 |
| `check_trace_buildable.py` / `check_gsm8k.py` / sweep | 否——判据是输出内容与断言,不是退出码 | 成立 |

**没有任何一份文档里的结论建立在一次 `PIPESTATUS` 读数上。**
两处受影响的都只是打印,而且都当场看见了空值。

---

## 5. 方法:**一次 grep 是关于「文件集」的主张,不亚于关于「模式」的主张——而没说出口的那一半永远是文件集**

(m35,2026-09-06T07:18:23Z。这不是一个缺陷,是今天已经决定过四个答案的一条判据,所以单列。)

今天四次「搜了、没命中、差点当成证据」,**四次的模式都是对的,错的都是范围**:

| 谁 | 搜了什么 | 漏掉的是 | 如果就此收手会得出 |
|---|---|---|---|
| m35 | `stock_vs_m2_tolerance`(小写) | env 名是**大写**的 `E2E_STOCK_VS_M2_TOLERANCE` | 「这个旋钮是接好的」 |
| m35 | `E2E_MAGPIE_ROOT` in `steps/*.yaml` | **`shared.yaml`**(在包根,不在 `steps/`) | 「magpie 没接线」——**错的缺陷,错的修法** |
| m2 | 声明处 | 注释里的声明式写法算不算声明 | 24 个正常变量报成缺陷 |
| m1 | 一个含 `{` 的模式 | `{` 在 regex 里是区间量词 | —— |

**为什么它专门骗人:** 一条 grep 的输出会**如实报告它搜过的那部分**,
所以「没命中」看起来像一个关于世界的结论,实际上是一个关于**你选的那些文件**的结论。
模式写错通常会当场报错或返回明显荒谬的东西;**文件集选错什么征兆都没有。**

### 可机械化的形式

> **报告一次「搜索没有命中」之前,先说出文件集,并且证明它覆盖了应该覆盖的范围。**
> 我那次 `pip install` 的审计之所以能用,不是因为 grep,是因为
> `find . -type f` 减去被 grep 的后缀之后**返回空**——覆盖面本身变成可查的。

三条具体的:

1. **大小写:** `--var` 是小写、env 是大写、两者在不同文件里。
   任何单一大小写的手工 grep 都会漏掉一半。判据用 `-i`。
2. **包的声明只有两个地方,而其中一个不在 `steps/` 里:**
   `shared.yaml`(包根)+ `steps/*.yaml`。**只搜后者会把「已声明但指向不存在的路径」
   误报成「未声明」。** 这两个是不同的缺陷、不同的修法——
   **前者一个 `--var` 就修好,后者 `--var` 修不了,需要那个路径真的存在。**
3. **命中不是证据,不命中才是**(本文件 §1 已有,这里是它的另一半):
   命中之后必须读那一处在说什么。我核 `pip install` 那次六个命中里
   五个是 precondition 的 `err:` **字符串**、一个是注释。

### 和本文件其它条目的关系

§1「找出其余实例的方法」里那句 **「工具本身跑过已知答案再用」** 是同一条从工具那头说的;
这一条是从**手工检索**那头说的。**两头都要,因为今天四次里三次是手敲的 grep,不是工具。**

### 5a. 同一族的第二半:**`2>/dev/null` 会把一个「好的失败」变成一个「安静的零」**

(m35,2026-09-06T07:25:08Z。这一条是当场撞上的,不是复盘出来的,而且差三十秒就变成一条错的上报。)

本机 `find` 是 **`bfs 4.1.1`**,不是 GNU findutils。它**不支持相对时间戳**:

```
$ find <dir> -type f -newermt '-5 minutes'
bfs: error: Invalid timestamp.
Supported timestamp formats are ISO 8601-like, e.g. 2026-09-06T07:23:26Z
```

**注意它做对了什么:它报错,而不是返回空集。** 这正是本文件反复要求的那种工具行为
(「优先选坏输入报错的工具,而不是坏输入返回空的」)。

**而我把它关掉了:**

```sh
find "$R" -type f -newermt '-5 minutes' 2>/dev/null | wc -l    ->  0
```

树里最新的文件是 **73 秒前**写的。我读成「五分钟内零写入」,差一步就把
「这条运行安静了」报给 leader——**而那正是这个工具本来要防止的、方向相反的错误结论。**

**同一分钟里同一个坏谓词还产生了一个「看起来合理的非零」:**
`2>&1 | wc -l` 把错误信息的 **10 行数成了 10 个文件**。
**所以一个坏谓词能同时伪造零和非零,取决于你怎么接它的 stderr。**

**本机可用的写法(拿两个已知年龄的文件对过):**

| 写法 | 本机 |
|---|---|
| `-newermt '-5 minutes'` | **报错** |
| `-newermt '2026-09-06T07:18:00Z'`(ISO 8601) | 可用 |
| `-newer <参照文件>` | 可用 |
| `-printf '%T+ %p' \| sort \| tail` | 可用,而且**带日期**,见 2026-09-05 那条 `--time-style` |

> **判据:凡是它的「零」会被拿去做判断的命令,不许加 `2>/dev/null`。**
> 停滞检查正是这种命令——**它的零就是结论本身。**

*落实方式不是记住:`watch_for_verdict.sh` 的文件头里写明了这条,并且整个脚本没有
一处 `2>/dev/null`,理由写在旁边。*

### 5b. 一条相邻的:**把「谁」的问题问给人,把「什么」的问题下给系统 —— 两种错向的误投递**

(m35 与 leader,2026-09-06,相隔二十分钟,方向相反。)

- **m35:** 「这份共享文件现在还有谁在编辑?」——**问了一个人**,
  而答案是 `git status --porcelain` 一条命令。等了一轮。
- **leader:** 「停下来封存」——**下给了一个人**,
  而 `kind: ai` 的 agent 只能由 run 内部的 monitor 通过 `instruct()` 驱动
  (`monitor/base.py:766` 是唯一的生产调用点;`agent_sys/cli/` 里没有任何
  `instruct`),`E2E_INSTRUCTION` 在发车时就固定了(`shared.yaml:118`,
  而 `m1_deploy.yaml:96` 恰好记着一次「指令写进去了、没到达」)。
  收信人只能等,或者拆掉两次已证明的 bring-up。

**共同形状:一个关于「什么」的问题被路由给了「谁」,或者反过来。**
和 §5 同族——§5 是「这条主张其实是关于文件集的」,这一条是
**「这条主张其实是关于世界的,而我问的是人」**。

> **判据两条,都便宜:**
> 1. 开口问人之前,先问「有没有一条命令能答」——**读世界比读记忆准**;
> 2. 下达一条指令之前,先问「收信人有没有这个控制点」——
>    **读产物告诉你正在发生什么,它不告诉你你能做什么,那是第二个问题。**

### 4b. 同族第二例:`find` 的**好**失败模式,被我们自己用 `2>/dev/null` 关掉了

追加于 **2026-09-06T07:26:17Z**(`date -u` 实读)。**m35 发现**,leader 转达并自纠了转达口径,
我在本机复现。

本机的 `find` 是 **`bfs`**,不是 GNU findutils。相对时间字符串它**不接受**:

```
$ find <dir> -type f -newermt '-5 minutes'
bfs: error: Invalid timestamp.
Supported timestamp formats are ISO 8601-like, e.g. 2026-09-06T07:25:32Z
   rc=1
```

> **要点不是「bfs 是个陷阱」,恰恰相反:它响亮地失败了。
> 是 `2>/dev/null` 把一个好的失败模式变成了一个安静的零。**

这条和第 4 条同族——**在我们实际敲命令的环境里失效的东西**——但方向相反:
第 4 条是缓解措施本身坏了;**这一条是工具是对的,而我们把它的正确行为关掉了。**
写错教训的代价很具体:「避开 bfs」是错的,「**不要给一条可能失败的命令加
`2>/dev/null`**」才是对的,而后者本来就写在 CLAUDE.md 里。

**代价:** m35 差三十秒就要根据那个零上报「run tree 已经安静了」。
**最新的文件当时 73 秒。**

**更难看的变体(m35 同一分钟内抓到):** `2>&1 | wc -l` 把错误信息的
**10 行数成了 10 个文件**。
**同一个坏谓词,既能产生一个像样的零,也能产生一个像样的非零——所以那个结果
两个方向都不可信。**

**可用的写法(对已知年龄的两个文件做过已知答案测试):**

| 写法 | 本机实测 |
|---|---|
| `-newermt` + **ISO 8601** 时间戳 | 对(返回 2,正确) |
| `-newer <参照文件>` | 对 |
| `find … -printf '%T+ %p\n' \| sort \| tail` | 对,而且无歧义 |
| `-mmin -N` | 对 |
| **任何相对时间字符串** | **rc=1,报错** |

**我自己的审计(同一类,不只是 `find`):**

- 我早先跑过 `find /apps/data -maxdepth 3 -iname '*trace*' 2>/dev/null`——
  **同一形状,同一处压制**。它返回空,而我没有从它得出任何结论
  (`/shared_nfs` 为空那条来自 `ls`),所以代价为零。
  **廉价的实例会教出关于昂贵实例的错误结论**,所以它记在这里。
- **我确实靠一个零做过判断:**「真实 `mock_root` 仍然是空的」。
  已用**不压制 stderr** 的方式重取:`find … -mindepth 1` → 无输出、rc=0;
  同一条命令在一个已知有 3 个条目的目录上返回 3。**那个零是真的。**
- **`graft_kit.sh` 里有一处同族缺陷,已修:**
  `[ -n "$(ls -A "$DEST" 2>&1)" ]` —— 这是「**错误字符串是真值**」那一条:
  `ls` 失败时诊断信息落到 stdout,字符串非空,于是判定「目录非空」。
  它**恰好失败得安全**(拒绝覆盖而不是覆盖),**而这正是这种写法能通过 review
  的原因**。改成 `find "$DEST" -mindepth 1 -print -quit`,错误回到 stderr,
  不可读的目录变成一次可见的失败而不是一个数据。两个控制重跑过:
  空目标 rc=0,非空目标 rc=1。

---

## 6. **一个崩溃不是一次拒绝,而框架分不出来** —— `jsonschema` 导入失败杀死本轮第一次真实验证

(m1 定位,m35 测量爆炸半径,2026-09-06T07:44:33Z。**这是 `mission.verify.e2e.md` 缺陷 #2 的
逐字复现,连诊断信息躺在 `attributes.detail` 里没人读这一点都一样。**)

### 现象

```
07:29:59Z  deploy_and_prove  output_validating -> validation_unreached, escalated x4
attributes.message:
  check_deploy_kit: exited 1 and wrote no verdict.json; nothing was decided.
  lib/schema.py:169  from jsonschema import Draft202012Validator
  ImportError: cannot import name 'Draft202012Validator' from 'jsonschema'
               (/usr/lib/python3/dist-packages/jsonschema/__init__.py)
```

validator 的 `HOME` 指向自己的 zone,user-site 因此解析到一个空目录,
回落到系统的旧 `jsonschema`。同一个解释器在普通 shell 里拿到的是
`~/.local/.../jsonschema 4.26.0`,是好的。

### 爆炸半径:**按 validator 是 8/22,按 kind 是 15/15**

**导入在 `def validate()` 内部,不在模块顶层**(`schema.py:169` 是函数第一句)。
所以 `import schema_lib` 无害,**只有真的调用 `validate` 才炸**。
而调用点全都写着 `except schema_lib.SchemaError` —— **`ImportError` 不是
`SchemaError`**,于是它穿过去,body exit 1,没有 verdict。

| 会崩(8) | 不会崩(14) |
|---|---|
| `check_deploy_kit` `check_environment` `check_bench_report` `check_bench_result` `check_kernel_table` `check_no_regression` `check_optimization_shape` **`check_workset_shape`** | `check_acceptance` `check_command_parses` `check_deploy_serves` `check_identity_resolved` `check_measurement_order` `check_nothing` `check_overlay_applies` `check_packup_shape` `check_patch_live` `check_profiling_evidence` `check_speedup_substantiated` `check_trace_coverage` `check_worklist_shape` `check_workset_runs` |

**`check_workset_shape` 我第一遍数漏了**,因为我搜的是共享库的调用点
`schema_lib.validate(`,而它在 `check.py:667` **有自己的一份**
`from jsonschema import Draft202012Validator`。**又是 §5:模式对,范围错。**
抓到它靠的是补问了一句「整个包里还有没有别处 import jsonschema」。

> **但真正的数字是 15/15:`check_environment` 挂在每一个 kind 上,
> 所以每个 kind 都至少有一个会崩的 validator,没有任何 kind 能产出 verdict。
> 本轮不存在「部分绿」这个选项,也不存在绕过它的降级配置。**

### 这一条真正的教训:**verdict.json 表达不了「崩了」和「拒绝了」的区别**

handoff 被记成 `invalid`,而 `invalid` 读起来就是一次拒绝。
**`check_packup_shape.validator/check.py:184-188` 早就知道这件事**——它 catch
`Exception` 并把 `"THIS VALIDATOR DID NOT RUN: <type>: <msg>"` 写进 reasons,
注释原话 *"verdict.json cannot express the difference (todo.md T29)"*。
**另外七个调用 `schema_lib.validate` 的 validator 没有这个守卫**,
所以它们是安静地崩,而不是说出来。

→ **可机械化的改法(不是记忆):把 `check_packup_shape` 那个 try/except 模式
铺到每一个 validator body 上。** 它把第三种结局从「没有 verdict」变成
「一个自报『我没有运行』的 verdict」,而后者是可数的。

### 两条本可以省掉一小时的东西,都在包里

1. **同样的崩溃已经被记录过,而且就记在造成它的那一行上方。**
   `check_workset_shape.validator/check.py:655-661` 的 docstring:
   *"crashed on the import, wrote no `verdict.json`, and the handoff was
   recorded `invalid` — **a missing dependency reported as a judgement about
   the artefact**."* —— 又一次「答案已经被取回来了,只是没有被读」。
2. **修法的先例也在包里,而且比 `PYTHONPATH` 好。**
   `deploy_and_prove.task/mock_adapt.sh:103-112` **探测**一个能 import 的解释器:
   ```sh
   for candidate in "${AGENT_SYS_DEMO_PYTHON:-}" python3 /usr/bin/python3; do
     if "$candidate" -c 'import jsonschema, yaml' >/dev/null 2>&1; then PY="$candidate"; break; fi
   done
   ```
   **它在坏输入上给一句话而不是一个 traceback。** 硬编码 `PYTHONPATH` 在环境
   再变一次时会安静地解析到错的 site-packages;探测会说出来。

### 它对今天所有读数的影响 —— 包括我自己那个「绿」

07:36 我报告 zone 是绿的:**1 个 zone,37 个文件,没有空的**。那个结论**是对的**。
**而运行照样死在验证上。**

> **「validator 有没有被递到东西」和「validator 有没有活到去看它」是两个问题,
> 而只有第一个有工具。** 我的检查回答了它的问题,而它的问题不是决定结局的那个。

**推论,已写进 PRE-REGISTER:materials 检查是「归因一次拒绝」的前置条件,
永远不是「验证发生过」的证据。读任何一张板子之前,先把 `verdict.json` 的**个数**
和 kind 声明的 validator 个数对一下——缺失的那个就是第三种结局,而它是安静的。**

## An output-phase validator gets no PATH, lands on /usr/bin/python3, and HOME remapping hides the only modern jsonschema

Recorded 2026-09-06T07:47:09Z by m1, on smci355-ccs-aus-n04-25. Run
`/data/yihou/agent_sys_runroot/runs/20260906T064218-15c264/`.

**Symptom.** `check_deploy_kit` **exited 1 and wrote no verdict.json** — a crash,
not a refusal. The escalation had no recipient, so the task sat at
`output_validating` for the full 900 s with a log that read healthy, and the run
ended `deploy_kit: invalid`. **The explaining line was not the last line**; the
reason existed only in `store/event`'s `attributes.message`:

```
ImportError: cannot import name 'Draft202012Validator' from 'jsonschema'
             (/usr/lib/python3/dist-packages/jsonschema/__init__.py)
```

**Mechanism — four steps, and each was verified rather than inferred.**

1. An OUTPUT phase takes spec §8.2's PRODUCER row, which is
   `Prepared.environment` filled from `harness_env()` — the `env` allow-list in
   `~/.claude/settings.json`. The package's own `env:` block does **not** reach a
   validator (`validator/phase.py:324`; `check_deploy_serves/check.py:30` says so
   in prose).
2. That allow-list cannot carry PATH: `env_mgr/harness.py:43`
   `_RESERVED = ("CLAUDE_CONFIG_DIR","CLAUDE_CODE_TMPDIR","TMPDIR","PATH")`.
3. With no PATH inherited, `#!/usr/bin/env python3` falls back to POSIX's
   built-in default PATH and resolves **/usr/bin/python3 (3.10)** — not the run's
   interpreter, which here was miniconda 3.13 and carries jsonschema 4.26.0.
4. `validator/environment.py:235` remaps `HOME` into the zone. That is correct
   isolation — its docstring is explicit that it never inherits `os.environ` —
   and it drops `~/.local/lib/python3.10/site-packages` off `sys.path`. What
   survives is the system's **jsonschema 3.2.0 (2019)**, which predates
   Draft 2020-12.

**Controlled reproduction**, matching the traceback exactly:

```sh
env -i HOME=/tmp/fakezonehome_yihou TMPDIR=/tmp sh -c \
  'python3 -c "import jsonschema; print(jsonschema.__file__)"'
# -> /usr/lib/python3/dist-packages/jsonschema/__init__.py   Draft202012: False

env -i HOME=/tmp/fakezonehome_yihou TMPDIR=/tmp \
    PYTHONPATH=/home/yihou/.local/lib/python3.10/site-packages sh -c \
  'python3 -c "from jsonschema import Draft202012Validator; print(\"IMPORT OK\")"'
# -> IMPORT OK
```

**This is a host provisioning gap, not a package defect**, and the package says
so without knowing it: `assets/lib/schema.py:169`'s docstring claims `validate`
"behaves identically under the run's interpreter and under a bare
`/usr/bin/python3`". That holds wherever the system jsonschema is >= 4. **On this
cluster it is 3.2.0, and the claim is false** — so a docstring that was accurate
where it was written is now the thing that stops the next reader looking here.

**Blast radius: every validator that validates a schema, in every module.** Not
m1's alone. Nothing about it depends on which stage is real.

**Only one injection point exists**, because PATH is reserved: the `PYTHONPATH`
key of the settings allow-list. Note `_block` lets the **live** value win, so
adding the key is not enough on its own — the launch line must carry the full
PYTHONPATH or the key changes nothing.

### 6a. 附:**包里一条注释记着第一个集群量到的 `jsonschema` 版本,在本机差了一个大版本**

(leader 发现,m35 复测,2026-09-06T07:47:15Z。)

```
assets/lib/interpreter_sweep.py:103
  #: … `/usr/bin/python3` is jsonschema 4.10.3 with no `referencing`
```

本机实测:

```
/usr/bin/python3,我的 HOME      -> 4.26.0   ~/.local/lib/python3.10/site-packages
/usr/bin/python3,空 HOME        -> 3.2.0    /usr/lib/python3/dist-packages
```

**3.2.0,不是 4.10.3,而且方向正好要命:4.10.3 是有 `Draft202012Validator` 的。**
一个照着这条注释判断的人会得出「本机不可能出这个问题」。

> **这就是 RUN-PLAN 那条「读机制,不要读数值」——只是这次它出现在我们自己的
> assets 注释里,而不是出现在一份显然属于别的集群的文档里。**
> 一个测量值写进注释的那一刻就开始过期,而注释不带日期也不带主机名。

**可机械化的一半:凡是往代码注释里写一个测过的数值,连同「在哪台机器、什么时候」
一起写。** 没有这两样,下一个读者无法判断它还成不成立,
而**注释比文档更容易被当成当前事实**——它就在代码旁边。

*(空 HOME 的复现只要一条命令,不占 GPU:*
`HOME=/tmp/yihou_emptyhome /usr/bin/python3 -c "import jsonschema; print(jsonschema.__version__)"`*)*
