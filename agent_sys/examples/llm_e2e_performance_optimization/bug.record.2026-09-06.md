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
