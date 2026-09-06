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
