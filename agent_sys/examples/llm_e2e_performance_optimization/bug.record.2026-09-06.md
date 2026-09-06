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

### 无后果的,记录但不花钱

| 变量 | 读处 | 为什么不动它 |
|---|---|---|
| `E2E_RESOLVE_TIMEOUT_S` | `identify.task/entry.sh:22` | `identify` 是秒级的容器探针,1800 s 是天花板,没人会撞到 |
| `E2E_RUN_TAG` | `serve/mix_up.sh:103` | 默认回落到 `$CTR`,label 仍然可归属 |
| `E2E_PROBE_SCRATCH` | `lib/controls/packup_probe.py:81` | 工具自己的 scratch,默认 `tempfile.gettempdir()` |
| `E2E_SWEEP_SCRATCH` | `lib/interpreter_sweep.py:100` | 同上 |
| `E2E_PACKUP_MOCK` | `lib/mock_m5.sh:326` | 默认指向第一个集群的 `/shared_nfs/...`,本集群不存在——**但只在 m5 走 mock packup 时才到达,本轮不走** |

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
