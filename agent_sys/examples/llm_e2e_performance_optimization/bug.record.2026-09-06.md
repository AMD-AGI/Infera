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

### 有后果的第三个:**`E2E_KIT_PORT_BASE` —— 而它使 RUN-PLAN 自己开的药方无法执行**

(m35,2026-09-06T14:15:16Z。)

```
shared.yaml:92    # …own base -- a caller may pass `E2E_KIT_PORT_BASE` -- and records what it bound
                  ^ 只出现在一条注释里
读它的:  assets/check_deploy_serves.validator/check.py
          assets/check_deploy_serves.validator/stub_kit/stub_env.sh
          assets/deploy_and_prove.task/mock_adapt.sh
          以及 kit 自己的 scripts/env.sh
声明它的:**没有**。`shared.yaml` 和 `steps/*.yaml` 里都没有,也没有 `kit_port_base` 这个 `--var`。
```

**RUN-PLAN.md 的规范发车块自己写着**:*「`port_router` 移动一个端口而把整个 band 留在原地;
真正要动的是 `E2E_KIT_PORT_BASE`」*。**而 `E2E_KIT_PORT_BASE` 从发车行里够不着。**

> **一个开不出来的药方是药方本身的缺陷。** 2026-09-06 14:08 两条链同时起来,
> 两边都知道应该分开端口段,**两边都做不到**——只能改 `port_router`,而那正是
> 文档说不管用的那个。

**和 §1 其余实例同一形状,但后果不同:** 别的实例是「一个可配置项其实配不了」,
这一个是**「文档指定的规避手段配不了」**——所以撞车不是疏忽,是不可避免。

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

## 7. 方法:**一次 grep 是关于「文件集」的主张,不亚于关于「模式」的主张——而没说出口的那一半永远是文件集**

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

### 7a. 同一族的第二半:**`2>/dev/null` 会把一个「好的失败」变成一个「安静的零」**

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

### 7b. 一条相邻的:**把「谁」的问题问给人,把「什么」的问题下给系统 —— 两种错向的误投递**

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
和 §7 同族——§7 是「这条主张其实是关于文件集的」,这一条是
**「这条主张其实是关于世界的,而我问的是人」**。

> **判据两条,都便宜:**
> 1. 开口问人之前,先问「有没有一条命令能答」——**读世界比读记忆准**;
> 2. 下达一条指令之前,先问「收信人有没有这个控制点」——
>    **读产物告诉你正在发生什么,它不告诉你你能做什么,那是第二个问题。**

### 7d. 归纳:**今天四次代价,底下是同一个形状——一个看起来像标识符的值,其实是一个类别**

(leader 归纳,m35 记录,2026-09-06T12:34:07Z。四个实例分别由三个人在四个不同的场合撞上,
**而没有一个人在撞上的时候认出前一个**。)

| 值 | 看起来是 | 其实是 | 代价 |
|---|---|---|---|
| 容器名前缀 `yihou_` | 归属 | **所有人共用的类别** | 同一个容器先后判给两个人,两人都说不是自己的 |
| `min_launchers_in_top_n` | 你要传的旗标 | **validator 私有的 args 字段名** | 照抄会得到一个不存在的 `--var`,静默无效,四十分钟后同样拒绝 |
| `m2_profiling` / `run_profiling_mode_*` | 「模块 2」这一个东西 | **store 里三个不同的 closure** | 我差点把一份错的反例发到另一个集群的共享文件里 |
| `--demo-root` | 这次运行的标识 | **六次运行共用的父目录** | 「按 run id 找编排进程」这条路根本走不通,而它看起来显然可行 |

**共同的失败方式:相等性检查会通过,而它检查的不是你以为的那个东西。**
名字对上了,类别没对上——**而类别不对是不会报错的。**

> **判据(便宜,且在动手之前):对任何要拿来做判断的名字,问一句
> 「这个值在这个系统里唯一吗,还是一群东西共享它?」**
> 共享的 → 它只能用来**排除**,不能用来**确认**。
> 前缀能说明「不是我们的」不成立,不能说明「是我们的」;
> `--demo-root` 能排除别的根,不能锁定这次运行。

**和 §7 的关系:** §7 说的是**检索的范围**没说出口,7c 说的是**名字的来源**没说出口,
这一条是它们的共同上位——**一个名字的作用域没说出口**。
三条都不会报错,都返回一个格式良好的答案。

**一条正面的做法(m35,已用在 `make_gate_result.sh` 里):**
当一个关联只能是「关联」而不是「同一性」时,**把关联的方式写进产物本身**——
`supplied explicitly` 对 `inferred from --demo-root (an association, not an identity)`。
**六个月后的读者重新推不出这个区别,也不会想到要问。**

### 7c. **一条拒绝告诉你「把 X 设成 0」时,X 是它读的 args 字段名,不是你要敲的 `--var` 名**

(m2 发现,m35 复核两半,2026-09-06T12:22:29Z。这是 §7 那一类里最贵的一种,因为**照做会静默失败**。)

```
拒绝文本:   set min_launchers_in_top_n to 0
steps/common.yaml:162
    min_launchers_in_top_n: '${kernel_table_min_launchers:-10}'
             ^ validator 的 args 字段              ^ 真正的 --var
```

**照字面抄下来敲 `--var min_launchers_in_top_n=0`,会发生什么:**

我做了差分,不是推测——同一条 `show`,加与不加那个假 `--var`:

```
不加:  rc=2
加上:  rc=2
diff:  完全一致(字节级)
```

> **一个不存在的 `--var` 不产生任何信号:不报错、不警告、输出一个字节都不变。**
> 于是它 **load 干净、发车干净,四十分钟后为同一个原因再拒绝一次。**

*(leader 转述时说它「会被列进 supplied 然后忽略」。我没能确认这半句——
这次调用在打印那一行之前就因为别的缺失变量退出了。**能确认的是更强的那一半:
输出字节级相同**,所以在这个配置下它连「被列出」都没有。不确定的部分标出来,
不当成已确认。)*

### 为什么这属于 §7

§7 说的是**检索的范围**没说出口;这一条是**名字的来源**没说出口。
两边都是「看起来像同一个东西的两个东西」:

| | 看起来是什么 | 实际是什么 |
|---|---|---|
| §7 | 一次搜索关于「世界」 | 关于**你选的文件集** |
| 7c | 拒绝里的名字是**你要传的旗标** | 是**它自己读的字段** |

**而两者的失败方式相同:安静。** 错的文件集返回一个格式良好的空;
错的 `--var` 返回一个格式良好的 run。

### 判据(便宜,而且在发车前)

> **把拒绝里的名字拿去 `grep -n` `steps/` 和 `shared.yaml`。**
> 命中的那一行会长成 `<args 字段>: '${<真正的 var>:-<默认>}'`——
> **冒号左边是拒绝告诉你的,`${}` 里面才是你要敲的。**
> 两个名字**故意不同**,因为一个是 validator 的私有词汇,一个是操作者接口。

*(同族先例已在本文件:`min_requests` 与 `integration_min_requests` 被**故意**拆开,
理由是「一个 `--var` 在移动两个 owner 的评分线」。所以「args 名 ≠ var 名」不是疏忽,
是设计——而拒绝文本没有说这件事。)*

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

## 8. **一个崩溃不是一次拒绝,而框架分不出来** —— `jsonschema` 导入失败杀死本轮第一次真实验证

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
`from jsonschema import Draft202012Validator`。**又是 §7:模式对,范围错。**
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

*(数字见 §8:**按 validator 是 8/22**——只有真的调用 schema 校验的那八个会崩,
`schema.py:169` 的 import 在 `def validate()` 内部所以不到达就不炸;
**按 kind 是 15/15**,因为 `check_environment` 挂在每一个 kind 上。
决定「有没有部分绿」的是后一个数,而它只在 §8 里。—— m35 附注)

**Only one injection point exists**, because PATH is reserved: the `PYTHONPATH`
key of the settings allow-list. Note `_block` lets the **live** value win, so
adding the key is not enough on its own — the launch line must carry the full
PYTHONPATH or the key changes nothing.

### 8a. 附:**包里一条注释记着第一个集群量到的 `jsonschema` 版本,在本机差了一个大版本**

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

## Two module owners on one node share a port band AND a work_root by default, and the first to arrive kills the second

Recorded 2026-09-06T08:23:35Z by m1.

**What happened.** m1's run `20260906T075853-e6f882` (launched 07:58:53) died at
08:12:09 with exit **143 = SIGTERM**. The exit code says nothing; the reason was
in the stage's own log, `/data/yihou/e2e_flow/deploy_main.log`:

```
[deploy 08:11:26] tag=qwen3-32b-mix ctr=yihou_dk_qwen3-32b-mix ports=8101..8106
deploy.sh: ABORT: etcd port 8103 is already in use. Move the band with
  E2E_KIT_PORT_BASE=<free base>; do not wait for it.
```

**Ownership of the port established by config, not by name or adjacency:**
`docker inspect yihou_e2e_chain_main_etcd --format '{{json .Args}}'` shows etcd
configured on **8103/8104**. That run is m2's — orchestrator `573170`,
`/data/yihou/e2e_verify_20260906/m2/launch_chain.py`, started 08:05:04.

**Two defaults collide, and neither owner chose either of them.**

| | m1 | m2 |
|---|---|---|
| port band | `8101..8106` (kit default) | `8101..8106` (same default) |
| `work_root` | `/data/yihou/e2e_flow` | `/data/yihou/e2e_flow` — byte-identical |
| `validate_work_root` | `/data/yihou/e2e_flow/validate` | same |

**A preflight cannot fix the port half.** m1 checked 8101/8140/8160/5557/8801 at
07:58 and all were free; m2 launched at 08:05; m1's deploy reached the band at
08:11. **The reading was correct when it was taken and wrong when it was used** —
the same shape as reading GPU occupancy at plan time instead of at bring-up time.
The only real defence is a band that is not the default:
**`E2E_KIT_PORT_BASE` moves the whole band; `port_router` moves one port and
leaves the rest behind.**

**The work_root half is worse because it is silent.** Nothing aborts when two
runs write into one work root; they simply interleave. `deploy_kit/` under the
shared root was created 08:15:46 by m2's run while m1 still believed the tree was
its own. No corruption observed, and that is luck rather than a property.

**Compounding: m2's line is `mock_stages=none`**, a fully real chain, so its m5
reaches `mix_up.sh:81` -> `reset_gpus.sh`, a **node-level** kill of every KFD
process that protects only `slurmstepd`. Under CLAUDE.md rule 5 that makes the
node exclusive for the duration, so the two lines could not have coexisted even
with disjoint ports.

**What to change, and it belongs in the launch block rather than here** — this
is a launch-line fact, and a collision is invisible until the moment another
tenant takes your number:

- every concurrent line needs its **own `E2E_KIT_PORT_BASE`**, spaced well clear
  (the kit claims a RANGE, ~6 ports, not one);
- every concurrent line needs its **own `work_root` and `validate_work_root`**,
  and the owner's name is the obvious discriminator;
- and a line that will reach a real m5 needs the node to itself regardless.

---

## 5. `WAITING_RESOURCE` 和 `WAITING_HANDOFF` 看起来是同一种保护的两种口味,而只有一种是保护

追加于 **2026-09-06T08:34:54Z**(`date -u` 实读)。我在真链跑起来之后自查发现,**在它造成代价之前**。

### 事实

```
grep -n 'resources: {gpu' steps/*.yaml
  m2_profiling.yaml:432    resources: {gpu: 8}
  m2_profiling.yaml:462    resources: {gpu: 8}
  （m1_deploy / m3_analysis / m4_kernel_opt / m5_integration：一条都没有）
```

**只有 m2 的两个叶子申领 GPU 租约。其余四个阶段一个都不申领。**

### 为什么这很容易读反

`m2_profiling.yaml:267-272` 明写了这个设计意图,而且写得很好:

> *「第二个之所以等,记录成它本来的样子:一个**资源**,不是一个依赖。
> `WAITING_RESOURCE` 是和 `WAITING_HANDOFF` 不同的状态,正是为了让这个区别
> 活到运行记录里。」*

读到这一段,很自然会推出「框架用租约把 GPU 工作串起来了」。**推不出来。**
那段话只描述 **m2 内部两条臂**之间的关系。

> **跨阶段的分离完全来自 handoff 依赖链** —— m3 拿不到 m2 的产物就不能开始,
> 所以 m2 的引擎先被拆掉。**那是真实的顺序,但它是「依赖顺序」而不是「租约顺序」。**

**差别在哪里会咬人:一个从上一个阶段泄漏出来的容器,不会被任何东西排除在外。**
租约会排除它;依赖不会——依赖只保证「上一个任务报告完成了」,
不保证「上一个任务留下的进程没了」。而本文件 CLAUDE.md 一侧已经记着:
**agent 会活过编排、并在容器被停一分钟后重建它**。

### 同一天的具体实例:`measure_gpu`

我按「m1 占 0-3,所以 4-7 空闲」把 `--var measure_gpu` 从 `0` 改成 `4`。
真链起来之后测到:

```
08:20  cards: 0 0 0 0 75 75 75 75      引擎自己选了 4-7
08:21  cards: 75 75 75 75 75 75 75 75  m1 的 selftest 引擎（0-3）与主引擎（4-7）并存
                                        —— leader 实测,整个节点都在用
```

**两个前提都被同一个事实证伪:kit 在 bring-up 时自己选卡。**
`--var gpu_devices` 被省略(`none` = 自由选择),所以

> **发车行记录的是一个关于卡的**意图**;只有跑起来的部署记录**事实**,
> 而那个事实在发车时不可知。**

这不是谁不小心。这和本轮 `gpu_devices` 失效那条是同一个形状,只是从反方向来的:
那条是「flag 说了不算」,这条是「事实在说话之前还不存在」。

### 判据(可查,不靠记忆)

- 断言「某两个阶段不会同时用 GPU」之前,**先看它们申领了租约没有**:
  `grep -n 'resources: {gpu' steps/*.yaml`。没有租约 = 只有依赖顺序在保护你。
- 依赖顺序**只**在「上游没有泄漏容器」时成立。所以一个依赖 `measure_gpu` 的
  阶段开跑前,应当检查**那张卡上有没有容器**,而不是检查上游任务是否 succeeded。

---

## 6. 真实的消费者,以消费者从不使用的方式调用,就是另一件仪器

追加于 **2026-09-06T08:48:18Z**(`date -u` 实读)。**代价:一次真链在 m2 死掉,35 分钟。缺陷是我的。**

### 发生了什么

```
run_profiling_mode_off -> exit 1
AIPerf: ConfigurationError: hash_id 0 requested at 477 tokens but was already
        materialized at 512 tokens. A hash_id must map to a single fixed block size.
```

起机干净、路由健康、`ready after 10s`、拆除干净。**引擎一切正常,负载从未开始。**

**缺陷:** 我的 trace 生成器把 block 0 当成所有会话共享的 system prompt,
同时采用「最后一块是部分块」的布局。ISL=477 的记录只有一块,那块**既是共享的
block 0 又是部分块**,于是 477 token;ISL≥512 的记录里 block 0 是完整的 512。
**每一条记录单独看都合法,整个文件不合法。**

### 为什么我的验证抓不到——这才是要记的部分

我跑了**真实的** `PromptGenerator._build_token_sequence`,覆盖全部 3108 种布局,
带两个已知答案的对照,然后报告「已验证」。它有一行:

```python
check_trace_buildable.py:19    g._cache.clear()
```

**我是故意清缓存的**,为了让每种布局互相隔离、可归因、可复现。
**而那个共享缓存正是 AIPerf 的 `dataset_manager` 用来强制跨记录一致性的东西。
所以我的对照恰好消灭了唯一可能失败的性质。**
逐条验证了 3108 次,文件的那**一个**全局性质一次都没验。

`aiperf validate mooncake-trace` 通过 7761 行,原因相同:它逐行验证。

> **两件仪器,都是真的,都对唯一要紧的性质失明。**

**本轮一整天的教训是「用真实的消费者,不要用它的 validator」。这一条是它的下一层:
真实的消费者,以消费者从不使用的配置去调用,就是另一件仪器。**
而那个隔离是我出于好理由加的——可复现、可归因——**它同时把全局性质从测试里
拿掉了,而没有任何东西提示这件事。**

### 判据

- 一个性质是**逐项的**还是**全局的**,决定了对照该不该隔离。**隔离对前者是对的,
  对后者是致命的**,而同一个 harness 两者都能跑,不会告诉你在跑哪一种。
- 所以:**按消费者的调用方式调用它**——一个 `PromptGenerator`、不清缓存、
  全文件、按顺序。任何偏离都要问一句「这个偏离拿掉了什么性质」。

### 修法与验证(break → fix → pass,跑过的)

共享链只提供**完整**块;每条记录的部分尾块拿一个**没人复用的新 id**。
前缀复用(值得回放的那个性质)不受影响。

`tools/check_trace_wholefile.py` **从不清缓存**:

| | 旧文件 | v2 |
|---|---|---|
| 每个 hash_id 全文件唯一尺寸 | **6839 处冲突**,第一条 `hash_id 0: 512 then 477` —— **AIPerf 的原话** | 干净 |
| 真实 builder,一个共享缓存,7761 条按序 | 失败 | 全部通过 |
| `aiperf validate` | 通过(逐行,对此失明) | 通过 |

**新检查在旧文件上复现了 AIPerf 自己的错误信息**,所以它测的是真正杀死运行的
那件事,而不是它旁边的东西。**旧文件保留,不删**:它是证据,也是这个检查的负控制。

**代价侧的测量:** ISL/OSL 完全不变(同一 seed,只有 hash_ids 变了);
distinct hash_ids 4947 → 11525;**理论前缀命中率 0.81 → 0.57**。
命中率下降是尾块不再共享的直接结果。

## Where the `ABORT: etcd port 8103` evidence lives, and why three greps missed it

Recorded 2026-09-06T10:02:06Z by m1, at the leader's request — they could not verify the claim
and were right to label it unverified rather than repeat it.

**The string is not in the run tree and not in the workspace.** It is in the
**shared work_root**, which is a third place nobody greps:

```
/data/yihou/e2e_flow/deploy_main.log        418 bytes, mtime 2026-09-06 08:11:27
```

```
[deploy 08:11:26] tag=qwen3-32b-mix ctr=yihou_dk_qwen3-32b-mix ports=8101..8106
[deploy 08:11:26] gpus=0,1,2,3 tp=4 graph_max_bs=64
[deploy 08:11:26] work_root=/data/yihou/e2e_flow/qwen3-32b-mix
    LISTEN 0      4096               *:8103             *:*
deploy.sh: ABORT: etcd port 8103 is already in use. Move the band with
  E2E_KIT_PORT_BASE=<free base>; do not wait for it.
```

> **A stage writes its own diagnosis into `work_root`, not into the run tree.**
> So `grep -r` over the run directory — the obvious place, and the place the
> stall/verdict evidence does live — returns nothing, and the natural reading of
> that nothing is "the claim is unsupported".

**And the shared work_root makes that evidence perishable**: the same path is
written by whichever run holds it, so a second owner's stage can overwrite the
first's diagnosis with no error. Copied to
`/data/yihou/e2e_verify_20260906/m1/EVIDENCE-deploy_main-abort-08-11-26.log`
with a read timestamp in the file, because `cp -a` preserves the SOURCE mtime
and therefore records nothing about when the copy was taken.

### The sequence, with all three accounts, because only two are load-bearing

| time | what | established by |
|---|---|---|
| 08:11:26 | m1's deploy **hard-aborts** on etcd 8103, held by the other line | the log above, first-hand |
| 08:12:08 | leader SIGTERMs m1's run, **for a card-contention reason that was wrong** | leader's own account |
| ~08:20 | the chain's *selftest* engine takes cards 0-3 — the four m1 declared | leader's `rocm-smi` + `docker inspect` |

**The collision was real and it was via ports, not cards, and it had already
happened one minute before the stop.** The card contention the stop was ordered
for did not exist at 08:12 and did exist by 08:20, through a container that had
not been created when the decision was made.

**What generalises is not the timing.** The leader gave two instructions that
hour whose premise was *which cards a line will occupy*, and both premises were
falsified by one fact:

> **The launch line records an INTENTION about cards. Only the running
> deployment records the FACT** — the kit picks its own devices at bring-up
> (`worker_entry.sh:93`, from `E2E_KIT_GPU_DEVICES`), so the answer is not in
> `--var gpu_devices`, not in `mix_worker.sh:26`'s default (that is m5's serve
> path, not m1's kit path), and not in what an earlier run did.

Three people predicted the cards from the launch line and all three were reading
the wrong file. **This is the same lesson as the `gpu_devices`-is-inert entry
arriving from the opposite direction**: there the flag was inert and the default
happened to be right; here the flag is absent and the software chooses well.
**Both times, reasoning about cards from the launch line was wrong.**


---

## 9. 900 秒 stall 是自终止的,不留孤儿 agent、不留容器 —— 实测,退掉一条一直在用的假设

追加于 **2026-09-06T14:10:10Z**(`date -u` 实读)。**这不是 bug,是一条被证伪的操作假设**,
记在这里是因为它改变 leader 在每一次 stall 时该做什么,而消息不会留下来。

### 此前的做法与它的依据

一整天我们在每次「escalation 无接收方」时都**提前杀掉运行**,理由是
「不要为 stall 白付 15 分钟」,而背后还有一条更贵的担心:CLAUDE.md 记着
**「杀掉编排进程不会杀掉它的 agent,而 agent 会在容器被停一分钟后重建它」**。
于是没有人敢让 stall 自己烧完——**那条路径从来没有人看过。**

### 实测(运行 `20260906T130845-298750`,2026-09-06 ~14:05Z)

让它烧完之后:

```
main is waiting on a decision no one will make — the escalation reached the top and
this entry point installs a sink that records and does not answer
(validation_failed: the task is terminal and there is nothing to push).
Nothing has changed for 900 s; still in a phase: m2_profiling:running, main:running,
run_profiling_mode_on:output_validating

run complete; this package promises no failure, and the run did NOT finish: …
```

编排进程**自己退出了**。我去杀它的时候它已经不在了。随后的核对:

| 检查 | 结果 |
|---|---|
| 任何 cwd 在 run 目录下的 `claude/versions` 进程 | **无** |
| 我们的容器 | **0** |
| 八张卡 | `0 0 0 0 0 0 0 0` |
| 端口 8101-8106 / 811x / 812x / 814x / 5557 / 8801 | **全部释放** |

> **900 秒 stall 在本集群是自终止的,而且它自己收拾干净:没有孤儿 agent,
> 没有遗留容器,卡和端口都回到基线。**

### 推论(改变行为的那一条)

- **不必再为了「怕留下孤儿」而抢在 stall 之前杀。** 提前杀仍然可以省 15 分钟,
  那是个**时间**判断,不再是**安全**判断——两者此前被混在一起。
- **CLAUDE.md 那条「agent 活过编排」的教训仍然成立**,但它描述的是
  **人为杀掉编排**的情形,不是 stall 自己走完的情形。**两条路径的收尾不一样,
  而我们把前者的风险套在了后者身上。**
- 想省那 15 分钟仍然要按记录的顺序拆:**先 agent(按 `readlink /proc/<pid>/cwd`
  判别,不要用会匹配自己命令行的模式),再容器 `docker stop -t 10`,再编排,再核对。**

**与本文件其它条目同族的地方:这是一条「不去看就一直成立」的假设。**
它不是被推理推翻的,是被**让它跑完一次**推翻的——代价 15 分钟,而我们为了
回避这 15 分钟付了一整天的谨慎。


---

## 9. **吵闹的失败是良性的那个;安静的失败没有探测器** —— 两条链共用 `work_root`

(checkpoint 发现,m35 记录,2026-09-06T14:15:16Z。2026-09-06 14:08,两条链同时在跑了约五分钟。)

两条链共享了**两样**东西,而它们的失败方式完全相反:

| 共享的 | 会怎样 | 谁会知道 |
|---|---|---|
| 端口 8101/8102/8103 | 第二个 bring-up **中止,并且报出是哪个端口** | **立刻,而且指名道姓** |
| `work_root=/data/yihou/e2e_flow` | **什么都不会发生** | **没有人** |

> **端口撞车会自己喊出来。共用 work_root 不会。**
> 两条 run 往同一棵树里写,**事后没有任何东西能把它们分开**——
> 损害是靠**读产物**发现的,不是靠一次失败发现的。

**这就是为什么那五分钟里,唯一响的警报是良性的那个。** 真正危险的一半
**没有探测器**,而且**至今没有**:没有任何检查在发车时比对两条 run 的 `work_root`,
`preflight.json` 也不记录「这棵树里还有别人」。

**推论,而且它比这次事故大:** 我们一整天都在按「哪个失败会喊」来排优先级。
**一个不会喊的共享状态,不会因为它安静就更安全——它只是把发现的时刻,
从失败那一刻推到了某个人读产物的那一刻。**

## 10. 方法:**我回答了一个比被问的更窄的问题,然后把答案当成一般答案交了出去**

(m35,2026-09-06T14:15:16Z。这句话是 leader 从我一次错误里摘出来的,而它覆盖今天至少三次。)

| 被问的 | 我实际回答的 | 后果 |
|---|---|---|
| 「有没有第二条链?」 | 「**前一条**还活着吗?」——它死了,于是我报「没有撞车」 | 撞车已经发生了五分钟 |
| 「这个 grep 证明了什么?」 | 「在**我选的文件集**里没有命中」 | §7 |
| 「空的 `v0` 是缺陷吗?」 | 「**这一个** zone 的版本目录是空的」 | 一整晚的回溯筛查 |

**三次都不是搜错、读错或算错。** 三次都是:**问题的范围比我作答的范围宽,
而我交付的那句话没有带上自己的范围。**

> **判据(便宜,而且在开口之前):把要说的那句话写下来,然后问
> 「它的主语,是不是原问题的主语?」**
> 「前一条链死了」的主语是**那一条链**;「没有撞车」的主语是**整台机器**。
> **两句话都为真,只有一句回答了问题。**

**和 §7 的关系:** §7 说的是**检索**的范围没说出口,这一条说的是**结论**的范围
没说出口——而结论那一句是会被转述的那一句。

---

## 11. 一个**被守卫**的字段仍然可以整体错 —— `jobid` 指向一个已经结束的 hold

> **编号说明:本文件此时已经有两个 `## 9.`** —— 我把自己的 7 改成 9 去避开 m35 的 7,
> 而同一时间另一个人也写了 9。**在并发追加的文件里,顺序编号本身就是竞态。**
> 引用请用标题文字(可 grep),不要用编号,也不要用行号。**我没有给任何人重新编号**
> ——上一次重新编号正是这次冲突的来源。

追加于 2026-09-06T14:1xZ。**这一条是我自己的缺陷,而且它是 `produced_by.commit` 那条的
更尖形态。**

### 事实

2026-09-06 14:00:01,hold `29184` 结束;14:00:34,`29313` 拿下同一台节点(24 小时)。
**我 14:08 发的那条链仍然带着 `--var jobid=29184`。** m35 的带 `29313`。

`jobid` 会被封进每一份产物:`environment.yaml` 的 `runtime.slurm_jobid`。
**所以那条链产出的每一个 handoff 都会声称自己产生于一个当时已经结束的作业。**

### 为什么守卫抓不到它 —— 这才是要记的部分

`slurm_jobid` **是被守卫的三个字段之一**(另外两个是 `fixed.node`、`runtime.transport`)。
守卫在 `measure_in_container.sh:118-126`:

```sh
_agree_or_die() {  # name  ambient  from_record
  if [ -n "$2" ] && [ -n "$3" ] && [ "$2" != "$3" ]; then   # 只在“不一致”时拒绝
    ... exit 1
  fi
```

**它比较的是「环境里的值」和「记录里的值」,而两者都来自同一个 `--var jobid=`。**
一个陈旧的 jobid 在两边**完全一致**,于是守卫通过。

> **守卫检的是「两个来源是否吵架」,不是「这个值是否为真」。
> 一个字段可以被守卫,并且整体地、一致地错。**

这正是 CLAUDE.md 里「一致性检查通过时,先问它比的两个值是从哪来的」那条的实例:
**一个来源被拆成两份,通过只告诉你复制成功了。**

### 为什么 `show` 也抓不到

`29184` 是一个**格式合法的字符串**。`show` 只检查「有没有值」和 yaml 能否类型检查。
**一个指向死作业的 jobid 和一个指向活作业的 jobid,对 `show` 完全一样。**

### 与 `produced_by.commit` 的关系

同族,但更糟:

| | `produced_by.commit` | `jobid` |
|---|---|---|
| 值是否可知 | 是(树就是 git 仓库) | 是(`squeue -u $USER` 一行) |
| 缺省时写什么 | 字面量 `unknown` —— **自报无知** | 上一次的真值 —— **看起来完全正常** |
| 有守卫吗 | 无 | **有,而且通过** |

**`unknown` 至少在喊「没人告诉我」。一个陈旧的 jobid 什么都不喊。**

### 修法(机制,不是记忆)

发车前断言 jobid 属于一个**正在运行**的作业,而不是断言它非空:

```sh
squeue -h -j "$JOBID" -o '%T' | grep -qx RUNNING || { echo "jobid $JOBID is not RUNNING"; exit 1; }
```

**这属于「发车行从产物里恢复不出来」那一类的反面:它可以在发车前用一条命令证伪,
而事后从任何产物里都看不出来。**

---

## 12. 方法:**两个检查、一种失效方式,就是一个检查** —— 而且这一条是可测试的

追加于 2026-09-06T14:2xZ。**这一条解释了我今天全部四次「按相邻推归属」的错误,
一次性。** 前面几条把它们当成四次独立的不小心记着;它们不是。

### 今天最干净的实例

我要确定「我刚发的那条链的 run 目录是哪个」。我用了**两个**检查:

```
ls -l /data/yihou/agent_sys_runroot/latest      -> 20260906T140831-026b96
find …/runs -maxdepth 1 -newermt '3 minutes ago' -> 20260906T140831-026b96
```

**两个答案一致,所以我信了。它们都错了,而且错得一模一样。**
`latest` 指向**最新**的 run;`-newermt` 也按**新旧**筛选。
m35 的目录比我的晚 **16 秒**创建,于是两个检查都选中了它。

> **两个来源不等于两个方法;而「同一个方法执行两次」在结果一致时,
> 看起来正好像交叉验证。**

这是 CLAUDE.md 里「来源独立不等于方法独立」那条的极限形态:
**不是两个人读同一类证据,是一个人跑同一个判据两遍。**

### 可测试的形式(这才是值得带走的)

**问:如果答案是另一个,这个检查会返回什么?**

| 检查 | 若「我的 run」不是最新的,它会返回什么 |
|---|---|
| `latest` | 仍然返回最新的那个 —— **答案不变** |
| `find -newermt` | 仍然返回最新的那个 —— **答案不变** |

**两个都对「正确答案是什么」不敏感 —— 它们只对「什么最新」敏感。
所以它们是一个检查。**

对照,真正独立的判据:

| 检查 | 若答案不同,它会返回什么 |
|---|---|
| `/proc/<pid>/cmdline` | 返回**那个进程的**参数 —— 随答案变 |
| `readlink /proc/<pid>/cwd` | 返回**那个进程所在的**目录 —— 随答案变 |

### 由此产生的规矩(tier 2,零成本)

> **一条运行的身份来自 `/proc/<pid>/cmdline`,或它的 agent 的
> `readlink /proc/<pid>/cwd`。`latest`、mtime、「最新的目录」一律不作为身份判据 ——
> 它们是关于时钟的事实,不是关于归属的事实。**

**并且**:在报告「两个检查一致」之前,先对每个检查问那个问句。
**如果两个都会在反事实下给出同一个答案,那就只有一个检查,而它没有被验证过。**

### 为什么它值得单独一条

今天四次归属错误分别被记成:卡号相邻、时间相邻、前缀相同、目录最新。
**四个不同的「相邻」,一个成因:我用了一个对正确答案不敏感的判据,
然后用第二个同样不敏感的判据去「确认」它。**

---

## 13. 一道守卫的**两条等待分支在本机上是不可达代码**,因为它的判别器恒为真

**observed_at 2026-09-06T14:51:45Z(`date -u` 读出,不是手写);发现者 m35,
运行 `20260906T140831-026b96`。**

`_off` 臂在 preflight `exit 1`,程序体,`escalated` 的理由是
*"nothing to push: the executor is a program body: there is no agent to instruct"*
—— 没有接收方,任务停在 `running`,900 秒 stall 计时开始。

### 现象

```
PREFLIGHT ABORT: a foreign container holds a GPU on this node and a card this
kit was told to take reads busy.
  instrument 1, container label 'deploy_kit_owner=yihou' : NO container carries it
  instrument 2, running containers holding /dev/kfd : rc_26_7_902 xiaoming-dev
  instrument 3, KFD processes (SUPPLEMENT ONLY) : none visible
  cards read at 2026-09-06T14:43:22Z: 0:112326901760 1:115476824064
                                      2:117001453568 3:115174834176
```

> **注意:上面这段是逐字的。`waited: 0s` 这一行本文其余部分提到过,
> 它 —— 不 —— 在这段里。** `waited:` 只出现在 `:256` 的**超时**中止文本中,
> 而这次走的是 `:211` 的**立即**中止,它不打印等待时间。
> **「等了 0 秒」是我从两件事推出来的,不是读出来的:** `:211` 在轮询循环
> `sleep` 之前返回;以及 preflight 读卡 `14:43:22Z`、事件落盘 `14:43:23`。
> *这一条留在这里,因为我第一版把它写进了引用块里 —— 那正是本文件反复记录的
> 「把推断混进逐字引用」。*

### 为什么它是错的 —— 一条命令的反证

```
rc_26_7_902    devices: /dev/dri /dev/kfd /dev/infiniband     up 2 days
xiaoming-dev   devices: /dev/kfd /dev/dri /dev/infiniband     up 3 days
cards at 14:49:43Z (六分钟后，两个容器仍在运行):  0 0 0 0 0 0 0 0
```

**被点名的两个「陌生人」此刻仍然活着,而八张卡全零。它们从来没有持有过显存。**
14:43:22 那 112–117 GB 是我们自己的 `check_deploy_serves` 引擎在拆除中放显存
——它 14:40:00 起机,14:44 消失。

### 机制:判别器测的是「设备映射」,不是「占用」

```
preflight.sh:121  _gpu_containers()
    docker inspect "$n" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}'
      | grep -q '/dev/kfd'
preflight.sh:132  _foreign_gpu_containers()   = 全部 GPU 容器 减去 带我们 label 的
preflight.sh:211  "Instrument 2 decided: a stranger. Abort immediately."
```

**本机上任何开发容器都会永久映射 `/dev/kfd`,所以 `_foreign_gpu_containers`
恒为非空。** 于是:

> **只要有任何一张卡读到 busy,不论持有者是谁,这个脚本就走 `:211` 中止。
> 指令里要求的两条等待分支 —— 「是我们的前驱 → 等 `DK_GPU_WAIT_S`」和
> 「什么都看不见 → 也等」—— 都要求 instrument 2 为空才能到达。它永远不为空。
> 两条分支在本机是不可达代码。**(它们没执行过是从 `:211` 的位置和
> 14:43:22→14:43:23 的一秒间隔推的,不是从中止文本里读的 —— 见上面那条注意。)

所以「守卫倒向中止」这句话对它还太客气:**它只有一条活分支。**

### 它自己的收尾文字说对了,然后在下一行扔掉

```
preflight.sh:259  "something is holding VRAM that docker cannot name from here"
preflight.sh:260  "If instrument 2 named a container, it is a stranger."
```

### 同一句话在指令里被正确地用过一次,又被错误地用了一次

给 m1 的指令写着 *"KFD process inspection … must never be the deciding test:
/proc cannot see into other containers"* —— **这句话逐字适用于
`HostConfig.Devices`,它看见的是映射不是使用。** instrument 3 被正确降级了,
instrument 2 有同样的缺陷,却被立为决定者。

### 修法(一处)

容器只有在**有显存可归属于它**时才算持有 GPU;归属不出来时诚实的标签是
`unknown`,而 `unknown` 必须路由到那条**已经存在、已经有预算**的等待,不是中止。

### 第二个独立成因,不要用第一个把它盖住

**没有任何东西把 `check_deploy_serves` 的拆除和 m2 的 preflight 排序。**
14:40:00 验证器自己起机,14:43:22 m2 读卡 —— 相隔三分钟,而拆除还没完成。
这是把 busy 读数摆到守卫面前的那件事;守卫的缺陷决定了它必然被读成中止。
**两个都要修:第一个让 busy 不再等于中止,第二个让 busy 不再出现。**

### 与 CLAUDE.md 顶部那节的关系

这正是今晨写下的那条:*「一道守卫的价值,不看它挡住了上一次,看它在
『它判不出来』时往哪边倒。」* 这道守卫是为了修上一次运行的
「等错了对象」而收窄的,**而收窄的代价落在这一次。**
它同时是 `VRAMSUM=272 KFD=0` 那条的镜像:那次是 `/proc` 跨不过命名空间
所以**拒绝不足**;这次是 `HostConfig.Devices` 跨得过命名空间但答的是另一个问题,
所以**拒绝过度**。**两次都是判别器回答了一个相邻的问题。**

---

## 14. ~~引擎自称 ready 之后一秒失去 detokenizer~~ —— **这一条的中心主张是错的，见开头的撤回**

> # 【撤回 2026-09-06T16:13:28Z，作者 m35，撤回的是我自己】
>
> **下面把 detokenizer 那两行当成故障签名。它不是签名，它是每一次起机都有的噪声。**
> 我在 run 4 里据此加了一次重试，**那次重试连杀两个健康的起机**，
> 两次都在第 130 秒、对着 2400 秒的预算、冷启动中途中止，两次都没有 `router.log`
> ——router 根本没被起到。
>
> **判据是一张表,分母写明:**
>
> ```
> 21 / 21 —— 100%
> 19 个不同的臂（find -name 'worker*.log' 于 e2e_flow / e2e_flow3 / e2e_flow4，
>              重复副本已合并）+ run 4 的两次尝试
> 每一个都有 "fired up and ready to roll!" ×1 和 detokenizer 失败 ×3–5
> 其中包括每一次成功的起机：c2a（真答过一次）、chain3（15:21:42 PASS）、
> serves-d8ff1deb（check_deploy_serves 通过）
> ```
>
> **我只采样了失败的那些。** 我打开的四份日志,每一份都是因为出了事才打开的——
> **这个样本在结构上不可能给我反例**,而这句话我今天写进过 CLAUDE.md 两次。
>
> **更糟的是 kit 自己在同一份日志里说过它是良性的,而它在三十秒后照我说的中止了:**
> ```
> 16:05:45  not ready yet (60s of 2400s). A cold start on this model is minutes of
>           weight load and JIT kernel builds; the repeated health-check failure
>           in the log is that, not a hang.
> 16:07:15  worker log carries the detokenizer signature after 130s
> 16:07:15  attempt 1 hit the known detokenizer signature; retrying exactly once
> ```
>
> **这正是我今晨写进 CLAUDE.md 顶部那节的形状**——「一个修复正确地针对眼前那次
> 失败,而对下一次是错的」——而我在引用那条规矩之后几小时内亲手造了一个实例。
> 我自己那三个问题本来就能拦住它:
> * *哪一类输入从通过被推到拒绝?* —— **本机上每一次健康的冷启动。**
>   我当时回答「没有,它只是给一条本来就会死的路径加一次尝试」。**那是假的,而我没查。**
> * *它判不出来时倒向哪边?* —— **倒向中止**,在第 130 秒。我说它倒向「花时间」。
>
> **【run 3 的死因已经查明,2026-09-06T17:00:58Z,发现者 checkpoint,他们同时撤回了自己
> 15:48 那份基于 `tail -4` 的报告——那四行读到的是恢复,不是故障。】**
>
> `worker.log` 272-286 的完整段落:
> ```
> 15:31:19  The server is fired up and ready to roll!
> 15:31:19  [aiter] Process-3 start build [mha_batch_prefill_bf16_...]
>           Process-1/2/4 waiting for baton release
> 15:31:40  Health check failed ... detokenizer ... last_heartbeat 15:31:19
> 15:31:47  Health check failed ...
>           [aiter] Process-3 finish build, cost 32.6s
> 15:31:53  SGLang ready on port 8141 · "GET /health" 200 OK
> ```
> **引擎卡在一次 32.6 秒的 aiter JIT 编译里。detokenizer 心跳正好停在那个窗口,
> 然后自己恢复了。** kit 按 worker 的 `/health` 起 router;router 起进了那个窗口、
> ConnectError、退出。**worker 15:31:53 回来了,而 8140 上已经没有人在听。**
>
> > **没有任何东西坏掉。是有东西慢,而一个超时把「慢」变成了「失败」。**
>
> **这同时解释了 detokenizer 那两行为什么普遍存在:它是一次 JIT 编译的影子。**
> 21/21 说的是「它到处都有」,checkpoint 这份说的是「它为什么在那儿」。
>
> **未修,而且刻意不修:** router 的就绪门槛太短是**一次观测**。leader 提过
> 放宽它,随即自己撤回——理由值得抄:*「21/21 表明 JIT 停顿在全部 21 条臂里都发生,
> 而只有一条臂的 router 死于它。所以这是十九分之一,不是抛硬币;
> 而我正要在两个我批准的守卫各杀死一次运行的当天,凭一次观测批准第三个守卫。」*
> **第二次出现时再修,那时它有两次观测。记在这里就是为了让第二次是「认出来」
> 而不是「重新发现」。**

> **~~run 3 那条臂的真实死因仍然未知。~~**（上面已查明）它的 router 连不上 8142 的 etcd,
> 而 detokenizer 那两行在**每一次**起机里都有,包括同一天成功的那些。
> **下面保留原文,因为「我当时是怎么推出来的」比结论有用。**

### 原文（结论已撤回，推理过程保留）

#### 原 14 标题: 引擎自称 ready 之后一秒失去 detokenizer —— 而**修好的守卫这一次根本没被调用**

**observed_at 2026-09-06T15:46:57Z（`date -u` 读出）；发现者 m35，运行 `20260906T150155-79bca5`。**

上一条(§13)的修法落地了、被验过了、**而这次运行死在别的地方**。记下来是因为
「我们刚修的那个」是最容易被当成死因的东西。

### 时间线（全部来自那条臂自己的日志，不是退出码）

```
15:29:15  臂 preflight：六个端口全空、卡全空、ok: true、waited 0s
15:30:37  worker: Application startup complete
15:31:19  worker: The server is fired up and ready to roll!
15:31:20  worker: Health check failed — no response from detokenizer for 20 s,
                  last_heartbeat 15:31:19            （共 4 次）
          router: httpx.ConnectError: All connection attempts failed —— 启动即死
15:31:54  拆除；check_deploy_serves REFUSED；deploy_kit 判 invalid
```

**kit 是在 worker 的 `/health` 通过之后才起 router 的**（`deploy.sh:181-198`,
`curl -sf .../health` 循环 + `die`）。所以 worker 答了 `/health`,
**而它在自称 ready 之后一秒失去了 detokenizer**，router 起来时对面已经没人。

### 三个嫌疑人，各被一次测量排除（不是被论证排除）

| 嫌疑 | 反证 |
|---|---|
| §13 修好的 preflight | **根本没被调用**。臂 preflight `ok: true`，`classify()` 未进入，卡 283 MiB |
| 端口 / `E2E_KIT_PORT_BASE` | **run 2 与 run 3 用的是同一个 8140-8145**，两份 `deployment.json` 一致，`ss` 在 15:29:15 报六个全空 |
| 我改的 `work_root`/`container` | 机制不经过它们；同镜像同卡，**m1 自己在 `/data/yihou/e2e_flow3` 上 15:21:42 起机成功并真答了一次** |

### 这条线在一小时前就出现过，而我把它记成了「记录，不解读」

run 2 的 m1 `deploy.log` 里写着 *"the health check keeps failing. That is not a
hang."* 我当时明确说不解读它。**它不是模板散文，是本机一种已知的引擎行为——
run 2 扛过去了，run 3 没扛过去。**

### 【更正 2026-09-06T15:50:04Z：下面这段的**归因是错的**，原文保留】

> **我把它写成「run 3 的 kit 引入的回退」。不是。** leader **自己跑的**已知答案
> 扫描(我先前在这里记成「checkpoint 的」,是误记;**checkpoint 主动指出不是他们
> 跑的,leader 也来更正**——一个被错误署名的检查,是没有人会去重跑的检查):
> `etcd.log` 在 `serves-6e2f6dbb`、`serves-e3e9d27d`、
> `serves-ec31ab7d` 都缺,**而且 run 3 自己那次成功的 m1 部署
> `e2e_flow3/deploy/chain3` 也没有。**
> **两条代码路径,只有一条留 etcd 日志——这是长期存在的缺口,不是这次重写造成的。**
> 我把它挂到了我当时正在看的那个变更上。**修法不变,类别变了:
> 照我原来的写法,下一个人会去 diff 两份 kit,什么也找不到。**
> *今天第五次「把相邻当成因果」,这次是我的:run 3 重写了脚本、日志缺失,
> 于是我把两件事连了起来。*

### kit 的一处**日志缺口**，而它让我连续推错两次

run 2 的那条臂有 `etcd.log`；**run 3 的 kit 在任何路径上都不写它**（这一句仍然成立，
错的是把它称为「回退」——见上方更正） —— `grep -n 'etcd.log' scripts/*.sh`
返回空。于是臂的 log 目录只有 `router.log` 和 `worker.log`。

我先把这个缺席读成「etcd 从没起来」，再读成「etcd 通过了它的门」，
**两次都错：这个文件的缺席在任何方向上都不构成证据。**

> **一个前一版产物有、这一版没有的日志,不是「少了个文件」,是「一整类问题
> 从此不可归因」。而它是 agent 重写脚本时顺手丢掉的,没有任何东西报错。**

### 下一轮要改的三件，没有一件是 preflight

1. **恢复无条件的 `docker logs <etcd> > etcd.log`。**
2. **kit 已有的重试瞄准了错的签名。** run 2 对 NCCL/HIP 签名重试；run 3 直接
   `die` 并让人重试。**两者都不覆盖「自称 ready 之后失去 detokenizer」**——
   而那正是我们两次观察到的那一个。一次到达 `fired up and ready to roll!`
   之后一秒内失去 detokenizer 的起机，是可重试的，指令要说出来，
   并且保留第一次尝试的日志（§「重试必须留下被它跨过的那次失败的产物」）。
3. §13 的 preflight 修法和 `E2E_KIT_PORT_BASE` 原样带进下一轮。

---

## 方法:一次错误归因,和一个真实存在但与它无关的机制 —— 两个东西,必须分开记

*2026-09-06 18:4x,m2。这一条记两件事。它们在同一分钟里被提出来,长得像同一件,
而其中一个解释不了另一个。把它们并成一条,会给一个没做出这次拯救的修复记功——
**一个被记了假功的修复会被过度信任**,而下一次它不在场时没有人会察觉。*

### 机制 A —— 真正发生的那个:一个从 grep 里取来的值,没人问过它是哪个阶段产出的

我报告「产物少报了等待时间」,依据是 `gpu_wait_seconds: 0`。**错的。**
两条臂都记着 `10`:

```
/data/yihou/e2e_flow/pmoff/results/preflight.json   gpu_wait_seconds: 10
/data/yihou/e2e_flow/pmon/results/preflight.json    gpu_wait_seconds: 10
```

我 grep 的是**运行树**,命中的是 m1 自己封存的那份 preflight——**在那一份里 `0` 是对的**。
两个值都是真的,分属两个阶段,而我的 grep 从不报告是谁产出的。

> **判据不是「多查一点」,是「一个值在被使用之前,先说出哪个阶段产出了它」。**

**而这里最要紧的一句是否定的:分开 `work_root` 不能防住它。**
它和路径遮蔽无关——**同一棵树里的两份合法产物,一份属于上游,一份属于本阶段。**
无论目录怎么分,一次不带阶段限定的 grep 都会两份都命中。

### 机制 B —— 真实存在,但今天没有造成任何事

两次运行的文件确实共存于同一个 `pmoff/` 和 `pmon/` 下,时间戳自己说得很清楚:

```
pmoff/preflight.json          13:40:23      pmoff/results/preflight.json   18:25:25
pmon/preflight.json           13:44:29      pmon/results/preflight.json    18:30:42
```

**它不会中止,不会报警,只在事后翻检时才被发现。** 它是一个真实的隐患,
应当被修——**但它不是上面那次误读的原因。** 上面那次误读发生在运行树里,
不在这两个目录里。

### 为什么必须分开记

第一版归因是「路径遮蔽,所以分开 `work_root` 就解决了」。
听起来完整,而且它指向一个我们**本来就想做**的改动——这正是它危险的地方:

> **一个被记了假功的修复,会在它其实不覆盖的那一类问题上被信任。**
> 下一次同样的误读发生时,`work_root` 已经分开了,于是没有人会想到去看 grep。

这和本仓库已有的那条是同一族:**「一个由假理由支撑的正确修复,会被下一个认真的人拆掉」**——
方向相反,代价相同。分开 `work_root` 是对的,理由不是这个。

---

## 两个 900 秒的计时器,谁也不知道对方存在 —— 失败自己喂饱了本该杀死它的探测器

*2026-09-06 19:0x,m2,实测两次爆发确认,不是推断。*

**现象:一条运行在引擎已经不能生成任何 token 之后,又活了三十分钟,而且本来会
再活四个小时。** 它对每一个看 phase 行、看进程表、看运行树 mtime 的巡检都是健康的。

### 事实

```
18:35:24   引擎最后一行   #running-req: 31, #queue-req: 0   —— 此后十三分钟无任何输出
18:48:08   GET  /v1/models  200 / 1.3 ms      GET /health  200
           POST /v1/chat/completions max_tokens=4   10 秒无响应
18:48:43   GPU 0/2/3 use 100 %，mem R/W 0 %，bandwidth 0；GPU 1 use 0 %
18:50:24   aiperf TimeoutError × 32   export 669 → 701
19:05:25   aiperf TimeoutError × 32   export 701 → 733
```

**两次爆发相隔十五分钟,一秒不差,每次正好 32 条**——那是 AIPerf 的并发数,
也正是引擎冻结时 `#running-req: 31` 那一批。**它们不是完成的,是超时掉的。**

### 机制

- AIPerf 的**请求超时是 900 秒**;
- `agent_sys` 的 **stall 超时也是 900 秒**;
- 每一批超时爆发都会往运行树里写 `aiperf.log`,**落在 stall 截止之后几秒**。

> **失败以恰好足够的活动、恰好错误的间隔,喂饱了本该杀死它的那个探测器。**

上一次 store 事件是 **18:30:29**,900 秒到期应是 **18:45:29**,没有触发;
**18:50:24 的爆发把它重置了**,下一轮 19:05:25 再重置一次。剩余 529 条 ÷ 32
≈ 16 轮 × 15 分钟 ≈ **四小时**,期间四张 MI355X 以 100 % GFX、零显存带宽自旋。

**这两个 900 几乎肯定是各自独立选出来的。** 本仓库已经记过
「`--stall-after` 是双刃」,而那一条讲的是**设得太长会藏起诊断**;
这一条是新的形状:**设得等于下游负载生成器的超时,会让探测器永不触发。**

### 判据(便宜,而且是事后可查的)

一条「看起来卡住但没被 stall 杀掉」的运行,查两件事:

1. **运行树的写入是否周期性、间隔是否等于某个已知超时。**
   `stat -c '%y %n'` 排序一遍即可——**规整的周期是机器的签名,不是工作的签名。**
2. **产物是不是只增长「失败记录」。** `profile_export.jsonl` 从 669 到 701 到 733,
   每次正好 +32,而这三十二条全是超时。**一个只增长失败的产物,增长本身不是活性。**

### 别的读者会撞上的两个陷阱

- **`/health` 返回 200 由 router 应答,与引擎能否生成无关。** 这里 `/v1/models`
  和 `/health` 都是 1 毫秒返回,而一次 `max_tokens=4` 的生成十秒没有任何字节。
  **唯一有信息量的存活探测是真发一次生成。**
- **`ss` 显示 132 条 ESTABLISHED,看起来像「在干活」。** 一条通向卡死服务端的
  连接,和一条通向繁忙服务端的连接,在 `ss` 里长得一模一样——**这正是这里的全部困难。**

### 未定性的部分(明写,不要被下一个读者读成已解)

**引擎为什么停,没有答案。** 三个 TP rank 100 % GFX、零显存带宽、一个 rank 0 %,
是自旋而不是计算——**我只报这个签名,不报成因。**
能定案的测量是**在卡死状态下取引擎 Python 线程栈**,我没取,拆除之后不可再取。

> **下次 `_on` 以同样方式卡住时,第一件事是取那个栈,而且必须在拆除之前取。**

另有一条**时间上的巧合,不是机制**:`mixed_stacks/` 创建于 **18:35:21** 且为空,
引擎最后一行是 **18:35:24**,相隔三秒,随后 `/stop_profile` 超时。
**记下来是为了下次有第二个样本时能比对,不是为了现在下结论。**

---

## 一条拒绝信息,点名了一个守卫**并不读**的变量 —— 关掉 stack window 要三个 `--var`,而没有一条信息说全

*2026-09-06 19:1x,m2。这一条是在「照着拒绝信息去修」之前读代码才发现的;
照那句话做会原地再吃一次同样的拒绝。*

### 三个名字,一个决定

```
--var stack_window_s=0            steps/m2_profiling.yaml:367  E2E_STACK_WINDOW_S
--var stack_ranks=0               steps/m2_profiling.yaml:133  expect_stack_ranks
--var kernel_table_min_launchers=0  steps/common.yaml:162      min_launchers_in_top_n
```

`check_trace_coverage` 的 5b 规则是这样开关的:

```python
# check_trace_coverage.validator/check.py:223
want_ranks = int(args.get("expect_stack_ranks", 0) or 0)
if want_ranks <= 0:
    return True
```

**它读 `expect_stack_ranks`,而没有任何地方从 `stack_window_s` 推出它。**
于是 `--var stack_window_s=0` 单独设,产出的是一轮**故意不采 stack window
然后因为没有 stack window 被拒绝**的运行:`stacks_manifest.json is missing`,
外加 `check_kernel_table` 规则 9 再拒一次。

### 缺陷本身

**每条拒绝信息点名的子集都不一样,而且没有一条点全:**

| 出处 | 它说要设什么 | 问题 |
|---|---|---|
| `check_trace_coverage.validator/check.py:233` | `--var stack_window_s=0` | **正是这个守卫不读的那一个** |
| `check_kernel_table.validator/check.py:272` | `stack_window_s=0` 和 `min_launchers_in_top_n` 设 0 | 后者不是 `--var` 名,发车行上叫 `kernel_table_min_launchers` |
| `steps/m2_profiling.yaml:131` 的**注释** | 三者的耦合关系 | 只存在于其中一个变量旁边的注释里 |

> **本仓库已有「一个只守住复合值里某一个字段的修复,会把兄弟字段留在原地」。
> 这一条多一个拐点:指引不只是不全,它指错了字段。**
> **一条叫你去设一个守卫根本不读的变量的信息,代价是一次发车,
> 而发车之后的失败和你本来想避开的那一个一模一样。**

### 修法(建议,未实施)

不是把三个变量并成一个——那会改变语义(采不采、期望几个 rank、表里要几个
launcher 是三个可以独立取值的决定)。**修的是信息:每条拒绝把三个 `--var` 名
一次列全,并且用发车行上的名字,不用 yaml 内部的键名。**

*另附:`expect_stack_ranks` 本身今天由 m35 先提出,这里新的是「它和另外两个是
同一个决定的三个名字」,以及「三条信息里有两条指错」。*

---

## ~~`with_stack=1` 的第二个 profiler 窗口冻死引擎~~ —— **标题的因果是错的,见下方更正;判据(引擎日志里少了四行)成立**

> **更正,2026-09-06 19:1x,由我自己在写完本条之后发现,更正内容比原条目重要。**
>
> **同一份配置在两个半小时前成功过。** 运行 `20260906T154908-d9c7af`:
> ```
> 16:37:40..16:38:15  窗口 1  with_stack=0   4 个文件 ~38 MB   CAPTURE_OK
> 16:38:25..16:38:58  窗口 2  with_stack=1   4 个文件 ~67 MB   CAPTURE_OK
> 16:42:19..16:42:25  check_trace_coverage / check_kernel_table /
>                     check_command_parses / check_bench_result  全部 passed
>                     "stack window: 2 rank(s), 5,613,901 python_function events"
> ```
> **同一台机、同一镜像、同一 TP4、同一份 1262 条的 schedule、同样 3 秒的
> `with_stack=1`、两窗之间同样 10 秒间隔。一次通过,一次冻死。**
>
> **分母(2026-09-06 19:2x 补测,原写「两次尝试」是我漏数了一次):**
> ```
> 运行                      测量窗(with_stack=0)   stack 窗(with_stack=1)
> 20260906T113811-fdb0bd   CAPTURE_OK             ABORT: no aiperf load in flight
> 20260906T130845-298750   CAPTURE_OK             （未尝试）
> 20260906T154908-d9c7af   CAPTURE_OK             CAPTURE_OK   4 文件 271 MB
> 20260906T180412-6b7c19   CAPTURE_OK             CAPTURE_FAIL: no trace files
>
> 测量窗   4 次尝试，4 次干净
> stack 窗 3 次尝试，1 次干净
> ```
> **那次 ABORT 是另一种失败(负载已经跑完,窗会开在空闲引擎上),在故障成因上
> 与冻结无关——但它仍然是一次「没拿到 stack window」的尝试,把它剔出去
> 会美化这个数。**
> **不对称本身才是发现,而且它在这么小的样本上依然成立:无栈窗 4/4,
> 有栈窗 1/3,两者在同一台引擎上前后相隔几分钟。**
>
> **所以这不是配置的性质,是间歇性的。** 「`with_stack=1` 是那个变量」这个说法,
> 被一个 16:38 的干净通过否掉了。
>
> **而 `/stop_profile` 的 `ReadTimeout` 在成功那次也发生了**,随后
> `stable at 271429048 bytes` 加四个文件——**所以这个 stack window 上的
> `ReadTimeout` 是常态、不携带信息**;包里的注释早就说了判据是字节数,
> 而这里差别正在字节数:271 MB 对 0。
>
> **本条余下的时间线、缺失的四行 `Stop profiling`、生成探测、两个 900 秒计时器,
> 都是测量,仍然成立。**
>
> *我犯的一般性错误:在一棵运行树里诊断了一小时,一次也没问过
> 「这个阶段在这台机器上成功过没有」。判据是四条命令。*



*2026-09-06,m2。运行 `20260906T180412-6b7c19`,`run_profiling_mode_on`。
本条经历两次自我更正,两版都写在末尾,因为**错的那两版各自都很像对的**。*

### 时间线(全部一手,来自被冻引擎自己的日志与两份 capture 日志)

```
18:34:36   窗口 1 开始  with_stack=0  mixed
18:35:09   "Profiling done" × 4 （TP0-3）   /stop_profile 200
                                            4 个文件，各约 38 MB   CAPTURE_OK
18:35:21   窗口 2 开始  with_stack=1  mixed_stacks  3 秒
           "Profiling starts" × 4 （TP0-3）  /start_profile 200
18:35:24   引擎最后一行（Prefill batch，#running-req: 31；数条 200 OK）
           —— 此后十三分钟以上，任何 rank 没有再写过一个字
           "Stop profiling" 全日志共 4 次，全部属于窗口 1；窗口 2 一次也没有
           /stop_profile → ReadTimeout      0 个文件
```

**引擎从此不再生成 token**:19:48 探测,`/v1/models` 与 `/health` 均 1 毫秒返回 200,
而一次 `max_tokens=4` 的生成十秒无任何字节。**卡:三个 TP rank GFX 100 %、
显存带宽 0,一个 rank 0 %。**

### 判据为什么是「少了四行」

`profiler_manager.py:_stop_profile` 在调用 `self.torch_profiler.stop()` **之前**
先写 `"Stop profiling..."`。四个 rank 一行也没写。

> **所以引擎不是卡在 stop 里,它在 stop 请求到达之前就已经不再推进调度循环了——
> 窗口开了约三秒之后。`ReadTimeout` 是后果,不是原因。**

日志里 `RuntimeError` / `Traceback` / `INTERNAL ASSERT` / `stack.empty` **全为 0**,
所以**不是** SGLang 自己文档里那个 `!stack.empty()` 断言(镜像内
`docs/developer_guide/benchmark_and_profiling.mdx:374`,指向 vllm#18240 与
pytorch#101632)——那一个会抛异常并留下痕迹。**我们这一次是静默停摆。**

### 两条被排除的、看起来很像的机制(留着,免得下一个人重新发现并误判)

1. **`_stop_profile` 里的 `torch.distributed.barrier(self.dp_tp_cpu_group)`**
   被两个**逐 rank**条件包着(`profile_in_progress`、`torch_profiler is not None`),
   而 `_start_profile` 对 `start()` 的 `RuntimeError` 是**只让该 rank**
   `torch_profiler = None` 并返回 `success=False`。
   **任何 rank 在这两个条件上分叉,就会跳过 barrier,把其余 rank 永久留在里面。**
   这是真实存在的死锁形状,**但今晚不是它**——四个 rank 连 `"Stop profiling"` 都没写。
2. **第二次会话本身**:`_start_profile:119` 对 `profile_in_progress` 是**干净的提前返回**
   (`"Profiling is already in progress"`),不是挂起;而窗口 1 的 stop 在返回 200 之前
   已把 `profile_in_progress` 置回 False。

### 缓解(建议,未验证)

**关掉 stack window 需要三个 `--var`**,见「一条拒绝信息点名了一个守卫并不读的变量」那条。
**注意 `SGLANG_PROFILE_WITH_STACK=False` 在这条路径上不起作用**:
`tokenizer_control_mixin.py:387` 的逻辑是请求里的 `with_stack` 覆盖环境默认,
而我们的 capture 每次都显式传。

**未验证的核心问题:是 `with_stack=1`,还是「第二个会话」?** 两个变量同时变了。
分离它们的探针已写好并做过守卫控制:
`/data/yihou/e2e_verify_20260906/m2/probe_two_windows.sh`(`ARM=nostack` 先跑)。
**若复现,在停任何容器之前先跑
`/data/yihou/e2e_verify_20260906/m2/capture_engine_stack.sh`**——
py-spy 在镜像里(`/opt/venv/bin/py-spy`),需要 `--pid=host --privileged --user 0`
(`--cap-add SYS_PTRACE` 不够,本机 `yama/ptrace_scope=1`),
仪器已在一个植入了已知帧名的进程上验证过。

### 便宜的早期探测(应进 capture.sh,尚未实施)

> **任何 stack window 关闭之后,在引擎日志里数 `Stop profiling` 行:
> 少于 rank 数就是这个故障。** 它在窗口结束后数秒可见——
> 比 `ReadTimeout` 早,比 stall 计时器早几个小时。

### 我错了两次,两次的形状都值得记

1. 先报「stack window 开启三秒后引擎停了」——**只是时间上的相邻**,我明说了不是机制,
   这一步是对的。
2. 再报「0 个文件说明没 rank 走过 `stop()`,所以挂在 `stop()` 里面」——
   **前提对,结论多走了一步。** 0 个文件同样符合「从没到达 `stop()`」,
   而日志正是这么说的。
   > **那一行判据在我一小时前亲手拷下来的文件里,我拿它 grep 了别的东西。**
   又一次「答案已经被取回来了,只是没有被读」。

---

## 一个字段有两个消费者,而它们的要求互斥 —— `baseline` 同时喂 m3 的 `--impl` 和 m5 的整文件覆盖

*2026-09-06 20:4x,m2,应 leader 之请在 m4 跑之前从代码和 run 7 的真实 workset 读出来。
结论:**`forge_mock=1` 的产物会被 `apply_patch` 拒绝**,而且拒绝**只花几秒**。*

### 先更正一个代价上的前提

**`apply_patch` 不起任何服务。** `assets/apply_patch.task/` 只有三个文件
(`apply.py` / `entry.sh` / `readme.md`),grep `mix_up|deploy.sh|start_worker` 无命中。
它做的是 `docker create <image> true` + `docker cp` 把 stock 文件取出来、打补丁、
做检查、在节点上摆好 overlay、写 `mounts.json`。
**起机在 `integrate_and_verify`,是后面另一个 closure。**
> **所以这里的拒绝以秒计,永远到不了一条 m5 臂。**

### `apply.py` 的检查顺序

```
690  sha_stock = sha256(从镜像 docker cp 出来的文件)
691  change=="modify" 且 sha_stock != base_sha256          -> 拒绝
793  patched 编译                                          -> 拒绝
801  dropped, lost_reexports = surface_regressions(stock, patched)
809  dropped 且 substitution=="call_site_fragment" 且 overlay_files -> 硬停
828  dropped                                               -> 拒绝「drops N public name(s)」
853  sha_patched == sha_stock                              -> 拒绝「changed nothing」
```

### 会命中的是 828,而 workset 自己就写着

run 7 的 `operator_workset`(handoff `9037ab8a`),五个算子全部
`substitution: module_symbol` + `apply_mode: overlay_files`,
`module_symbols` 分别 **6 / 55 / 32 / 32 / 20** 个。
而 `30_run_forge.sh` 在 `KFO_MOCK=1` 下 seed 的那个 `baseline`,五个全部只有
**472–2163 字符**。gemm 那个全文如下:

```python
"""The incumbent: aiter's assembly a16w16 GEMM, `gemm_a16w16_asm`. ..."""
import aiter
def run(A, B, out, **kwargs):
    return aiter.gemm_a16w16_asm(A, B, out)
```

**模块级名字只有一个 `run`。** 被它覆盖的
`/sgl-workspace/aiter/aiter/ops/gemm_op_a16w16.py` 实测 **2414 字节、6 个模块级名字**,
`base_sha256` 与 workset 记录的 `3201806d…` **逐位相符**(所以 691 不会先拦)。
**六个进去、一个出来,828 拒绝。**

workset 自己的 invariant 里已经写了这句:
> *"The five other module-level names in this file — `_gemm_a16w16_asm`,
> `_SEMA_SHAPE`, `ASM_SPLITK_MAX_GRID`, `_get_semaphore_workspace_keyed`,
> `get_semaphore_workspace` — must survive an overlay."*

**「换个算子」不是出路:五个形状相同。** 这与另一个集群的记录独立复现
(「五个 baseline 全是 harness 形状,于是『换个算子』这条逃生路当场消失」),
而这一次的 workset 是今晚由另一个 agent 现建的。

**注意 853 那条不会命中**:mock 会写四行 `# MOCK RUN` 头,字节与 stock 不同。
**所以拒绝不是因为「负载等于 baseline」,是因为负载根本不是那个模块。**
一份真正的空改动(stock 模块 + 一行注释)828 和 853 都能过——
**用户「反向优化、`speedup: 1.0` 是正确值」的决定,机制上是支持的;
挡住的是种子的形状,不是空改动本身。**

### 根因:一个字段,两个互斥的消费者

`optimized_kernel.py` 同时要满足:

| 消费者 | 要求 |
|---|---|
| m3 的 `--impl`(`harness/_common.py:278-290`) | 自包含、在全新命名空间里 exec、导出顶层 `run(**inputs)` |
| `apply_patch` 的 `overlay_files` | 保住被覆盖模块的**全部**模块级公开名 |

**一个文件只有一种形状能同时满足:stock 模块原样 + 末尾追加
`def run(*a, **k): return <public_symbol>(*a, **k)`。**
`30_run_forge.sh:66-79` 正是这么描述另一个集群的
`mock_adapt.py:_read_seed()` 的——「让整份源码逐字通过,并追加 `def run`」。
**今晚真实的 `build_workset` 把一份手写的 15 行 wrapper 记成了 `baseline`。**

> **所以缺口在「`build_workset` 往 `baseline` 里放什么」,不在 mock,也不在
> `apply_patch`。** `30_run_forge.sh` 自己拒绝掩盖镜像方向的同一问题——
> *"if the workset's own baseline does not meet m3's `--impl` contract, that is a
> finding about the workset"*——**反方向适用同一条原则。**

### 附带的正面事实

那个 agent 从镜像里量出的 `base_sha256` 是对的。
**workset 的出处是可靠的;错的只是它选了哪份源码叫 `baseline` 这一个字段。**

---

## `output_validating` 这个 phase 标签,在验证失败之后**永远**停在那里 —— 它准确,而且意思和字面完全不同

*2026-09-06 22:3x,m2。两次运行死在这个标签下,leader 因此两次得出「验证阶段超过了
1200 秒」,并据此要改 stall 阈值。**实测:两次验证都在一秒内完成,而且都是拒绝。***

```
run 7  build_workset     20:18:48.674/.810/.998  三个 verdict，最后一个 FALSE
                         20:18:49.008  validation_failed -> 根任务（无 agent）
                         20:38:48      stall 触发   = 拒绝之后干等 1200 秒
run 8  optimize_kernel   22:10:51      三个 verdict，两个 FALSE
                         22:30:51      stall 触发   = 拒绝之后干等 1200 秒
```

**输出验证用了不到一秒。** 那二十分钟是**拒绝之后**在等一个没有接收方的升级——
根任务 `main` 是非叶子、不声明 agent,所以 `validation_failed` 投不出去。

> **任务在验证失败后没有下一个状态可去,于是 phase 停在 `output_validating`。
> 这个标签不是陈旧的,它是准确的——而它的意思是「验证结束了,结论是不通过」,
> 不是「正在验证」。**

**这比一个陈旧标签坏得多**:陈旧标签有机会被时间戳戳穿,而这个标签的时间戳是对的。
唯一能分开两者的是**verdict 文件的 mtime**——写完了就是验证结束了。

```sh
find <run> -name 'verdict*.json' -printf '%TT %p\n' | sort | tail -3
```

**判据:phase 说 `output_validating` 而 verdict 文件已经写完 → 它不在验证,
它在等一个不会到来的决定。** 两者相差的那段时间,就是 stall 阈值本身。

### 由此产生的两个错误结论,都被这个标签支撑着

1. **「验证阶段超过了 1200 秒」** —— 没有。那个被担心「要跑几十分钟」的
   `check_workset_runs`,在 run 8 上实测 **7 秒**(`reverify_shapes` 默认 1)。
2. **「看门狗杀掉了一个成功」** —— 没有。run 7 的 `build_workset` 输出验证
   **拒绝**了(mount 缺陷),run 8 的 `optimize_kernel` 拒绝了两次(缺 `--var gpu`)。
   **两次都不是超时杀死运行,是运行已经结束而超时只是收尸。**

**没有任何 stall 取值会改变这两次的结局。** 记下来,是因为下一个人看到
「两次都死在 `output_validating`」时,最自然的动作就是去调那个阈值——
而那个动作既不会有害,也不会有用,并且会让阈值被记上一笔它没做的功。

*(同族:本文件「读产物不读退出码」与「一条线看起来不动时按这个顺序读」。
那两条讲的是 phase 行**信息不足**;这一条讲的是 phase 行**信息充足而含义相反**。)*

---

## 一份 12.5 KB 的 README,因为「`## Boundary` 下面直接就是 `### 子标题`」被判为空 —— 运行 10 死于此

*2026-09-07 04:3x,m2。用框架自己的解析器在真实文件上跑出来的,不是推断。*

### 死因链

```
23:24:27  phase_done   INPUT/OUTPUT_VALIDATING finished
23:58:13  output_absent   exit_status=finished, detail=success
          message:     declared output 879b05db… was never delivered
          seal_refused: …/v1/content/README.md: required section 'Boundary' is empty
23:58:13  push_attempted  "continue, do it until finished"
23:58:13  handling_failed AgentNotListening: instruct(...) has no loop to deliver it.
                          The agent is finished and `mainloop` has returned.
```

**body 成功了,产物做出来了,封存被拒;框架的补救是叫 agent 继续,而 agent 已经退出。**
运行随后空转 4 小时 33 分,直到被人拆掉。

### README 一点都不空

`agent_sys/handoff/readme.py:sections()`,在那份真实 README 上跑出来:

```
total bytes 12552
  'Purpose'    body_len 1537
  'Interface'  body_len 1947
  'Boundary'   body_len 0     <- EMPTY
  然后是它的八个 ### 子节：1019 / 665 / 1826 / 571 / 546 / 910 / 185 / 1711 / 497
```

**`Boundary` 底下有约 8 KB 的内容,而它自己的 body 是 0。**

### 机制

`sections()` 遇到 `heading_open` 且 `token.level == 0` 就**开一个新节**。
markdown-it 的 `level` 是**嵌套深度**(blockquote / list),**不是标题层级**——
所以文档根部的 `###` 和 `##` 一样是 `level == 0`。

> **于是每一个根部标题都会终止上一节。`## Boundary` 紧跟着 `### …`,
> 它自己的 body 就只有中间那个空行。**

**docstring 明确讨论过 `level == 0`**(为了让 blockquote / list 里的标题不算数),
**却完全没有提到标题层级。所以「子标题算兄弟节」看起来是没被考虑,而不是被决定的。**
这一条我只报机制,不报意图。

### 生产者侧的规避(brief 级,可达 `kind: ai`)

**在 `## Boundary` 和第一个 `###` 之间写一句散文。** 一句就够。
`Purpose` 和 `Interface` 之所以通过,正是因为它们有自己层级的正文。

### 这一条同时是「更长的 stall 更安全」的反例

运行 10 用的是 `--stall-after 5400`(我按 leader 的要求设的)。
**它 23:58:13 就死了,04:31 才被发现——盲区 4.5 小时。**
> **阈值调长,对「安静地死掉」这一类失败是纯损失:它不救任何东西,只延长盲窗。**
今晚三种死法各不相同,而阈值对三种都不起作用:
1. 运行 7 / 8:**验证拒绝 + 升级无接收方** —— 计时器只是收尸。
2. 运行 6:**引擎卡死** —— AIPerf 每 900 秒的超时爆发**反复喂饱**探测器。
3. 运行 10:**封存被拒 + agent 已退出** —— 死得很安静,越长越晚被发现。

*(第 3 种与本文件已有的「`escalated` vs `handling_failed`」那张表同族:
`handling_failed` = ai 体、agent 已完成、`mainloop` 已返回。
新的是**触发它的原因**——不是 validator 拒绝,是 `seal_refused`。)*

---

## 15. m2 的 replay 路径从来没有人走通过 —— 两个形状,一条路径,而只有没有语料的集群才够得着

**observed_at 2026-09-07T05:47:46Z（`date -u` 读出);发现者 m35,运行 `20260907T043102-cdd3f7`
与 `20260906T170354-1048af`。两个缺陷都在把 run 4 的 m1+m2 物化成 `mock_root` 时撞到。**

### 缺陷一:目录名两套,而 `deploy_kit` 恰好掩盖了它

```
replay_root.py:152-156   kind -> STAGE 目录，然后写 <stage>/<KIND>/content
sweep_inputs.py:63-66    mock.sh 读            <stage>/<ALIAS>/content
    profiling_mode_off.bench_result  -> stage2-profiling/aiperf_baseline
    profiling_mode_on.bench_result   -> stage2-profiling/aiperf_profiled
    profiling_mode_on.profile_result -> stage2-profiling/torch_trace
    profiling_mode_on.kernel_table   -> stage2-profiling/kernel_table
```

现象:`exit 1: mock: no sealed content at .../stage2-profiling/aiperf_baseline/content`。

> **`deploy_kit` 能过,只因为它的 kind 名等于别名的最后一段。
> 第一个 kind 通过,给了「机制是通的」这个印象,而它是四个里唯一同名的那个。**

我在自己的物化副本里加了四个符号链接(内容一字未动,出处一次 `ls -l` 可见),
其中两个立刻生效:`aiperf_profiled` 17 个文件、`torch_trace` 18 个文件。

### 缺陷二,而它终结了这条路:`mock.sh` 无条件地 reshape `kernel_table`

```
m2_reshape: the record is missing at .../kernel_table/content/items/result/text.json
```

`m2_reshape.py` 的 docstring 自己写明了它为什么存在:

```
sealed stage2-profiling/kernel_table   content_type: reproducible
    items/result/text.json ...
this package's profiling_mode_on.kernel_table   content_type: structured_text
    items/text.json ...
```

**它是为了把「语料的形状」转成「本包的形状」。而一个从本包真实运行物化出来的
replay root,已经是目标形状了** —— run 4 的产物是 `items/text.json`。
于是 reshape 这一步不只是多余,**它是致命的**:它去找一个在真实产物里
不存在、也不应该存在的路径。

> **一般形式:`replay_root.py` 物化的是「真实生产者的输出」,`mock.sh` 期待的是
> 「语料的布局」。两个形状共用一条路径,而这个不一致只有在
> **「m2 被 replay」** 时才可达 —— 另一个集群上 m2 永远是从语料 mock 的,
> 所以那里永远不会走到这一步。**

**这就是「有些事实从一次运行内部看不见」的又一实例,只是换成了空间:
一个只有在语料不存在的集群上才够得着的缺陷。**

### 修法(未做,而且不该由我在活跃期做)

要么给 `mock.sh` / `m2_reshape` 一个「源已经是本包形状」的开关,
要么让 `replay_root.py` 按别名写目录并跳过需要 reshape 的 kind。
**两者都会改动一条活跃的 mock 路径,而现在有一条真实链在跑;
所以只记录,不动手。**

### 代价与判断

两次尝试各赔了一次 stall(约 7 分钟)加上我的时间。**第二次之后我停止了 replay
这条路,改为「让 m2 真跑」** —— m2 真跑约 13 分钟且已知可行,
而 replay 每修一层就露出下一层。
*随后发现另有一条全真链 `20260907T045327-13a18e` 已经在跑、而且更远,
于是我停掉了自己的运行,没有发车。* **端口都是 8101,不查就会撞。**

---

## 我加的「baseline 必须逐字」要求,和 `check_workset_shape` 的既有规则相撞 —— 一个算子因此被排除

*2026-09-07 06:47,m2。运行 `20260907T045327-13a18e`。**这是我自己那条 brief 修改
(`e248c76f`)的代价,由生产者自己写在 `Boundary` 里,而我是去读了才知道的。***

### 生产者的原话

> **One operator was excluded: `attention_ck_tile_kentry`, rank 2, 8.03% of the
> profiled window** — the largest single thing missing from this artefact. Two
> independent blockers, either sufficient. **First, its `baseline` would have to
> be `aiter/ops/mha.py` byte for byte, 138 KB, and that file trips two of
> `check_workset_shape`'s own rules once embedded in a Definition JSON: eight
> absolute paths, and two of the seven template markers rule 7 scans for.
> Trimming the file would break the whole-file-overlay property the verbatim
> requirement exists for.** Second, `_mha_batch_prefill` is batch prefill against
> a paged KV cache … `_common.py:build_inputs` is protected and fills every
> integer tensor with `torch.randint(0, 8, dims)`, which would send the kernel
> reading outside the cache. **That is not a failing gate, it is an illegal
> memory access that takes the process down.**

### 两个 blocker,只有第一个是我造成的

1. **逐字要求 × `check_workset_shape` 的绝对路径 / 模板标记规则。**
   一个真实的大文件嵌进 Definition JSON 之后,必然带着自己的绝对路径和模板标记。
   **而「裁掉它们」正好破坏这条要求存在的理由。**
   → **这两条规则是互斥的,而我加要求时没有检查另一条。**
2. **`build_inputs` 用 `torch.randint(0, 8, dims)` 填所有整数张量**,对分页 KV
   的 batch prefill 就是越界读。**这一条与我的修改无关**,是 harness 的既有限制,
   需要 per-operator 的输入构造器,属于包级改动。

**任一条单独成立就足以排除这个算子。所以「排除」不是我的修改独自造成的——
但第一条确实是,而且如果只有第二条,记录会长得不一样。**

### 由此得到的、比这个实例更一般的教训

> **给一条 brief 加要求之前,先看这条要求会把哪些既有检查从「可满足」推到
> 「不可满足」。** 我加的是「必须逐字」,而包里已经有两条规则在惩罚
> 真实源码必然带有的东西(绝对路径、模板标记)。**新旧两条规则各自都对,
> 合起来把一整类算子排除在外。**

**修法不明显,而且不该由我在停机前决定。** 可能的方向:
`check_workset_shape` 对 `definitions/*.json` 的 `baseline` 字段豁免那两条规则
(它是嵌入的源码,不是这份产物自己的路径),或者 `baseline` 改为按引用而不是
按值携带。**两者都会动到别人的文件和已冻结的契约,记在这里等重启后讨论。**

### 附:一个未定性的东西,现在写下而不是事后回忆

生产者在 `baseline` 顶部加了一个 **`__package__` 赋值**,理由是它自己测的:

> `_common.py:_exec_source` 用 `exec(compile(...), {})` 跑这个字符串,空命名空间
> 没有 `__name__`,模块自己的相对导入(`from ..jit.core import compile_ops`)会
> 抛 `KeyError("'__name__' not in globals")` —— **measured in the deployment
> image on 2026-09-07**。

**这是对的,而且补上了我那条 brief 修改的一个缺口**(「逐字复制真实模块」会
破坏 `--impl` harness,我没预见到)。

**未定性的部分:这一行现在也在会被 overlay 到真实模块上的负载里。**
它只增加名字、不删除名字,所以 `apply.py:828` 不受影响;
**但 `integrate_and_verify` 是真的 import 那个结果的,而一个在 import 时给自己
赋 `__package__` 的模块并不常见。我不知道它在那个位置有没有害。**
> **如果 m5 死在一个 import 上,第一个要看的就是这一行。**
