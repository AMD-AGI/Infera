# 上游状态：decode radix cache + 投机解码（MTP），以及 decode HiCache

日期：2026-09-24。调研对象：[sgl-project/sglang](https://github.com/sgl-project/sglang) `main` @ [`4142235c2b`](https://github.com/sgl-project/sglang/commit/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f)（2026-09-24 13:35 UTC）。本地基线：`a9fb1c3238`（2026-09-17）。基线源码取自本机 `/tmp/full519`，其 `pd_disaggregation_hook.py` 与 GitHub 上 `a9fb1c3238` 的同名文件 md5 一致。任务说明中的 `/tmp/sgl-src/srt` 在本机不存在。

调研方式：只读访问。raw 文件、PR diff、PR 页面和 GitHub search API；没有创建任何 PR、issue 或评论。GitHub REST API 在调研中途因共享 IP 配额耗尽返回 403，之后改用网页抓取，因此部分 PR 的评审细节只能看到页面时间线上的内容。

## 2026-09-28 更新

查询方式：PR 网页和 main 上的 raw 文件。REST API 仍因共享 IP 配额耗尽不可用。

- **main：** `pd_disaggregation_hook.py` 中的检查仍为 `speculative_algorithm not in (None, "DSPARK")`，EAGLE 和 NEXTN 仍会被拒绝。
- **[#40857](https://github.com/sgl-project/sglang/pull/40857)：** 仍为 open。最后一次活动是 09-24 19:26，作者 @ 了 wyzhang；之后没有新的评审或提交。CI 三项仍然失败，Accuracy 和 Speed 两栏仍为空。
- **[#40681](https://github.com/sgl-project/sglang/pull/40681)：** 维护者 kpham-sgl 于 09-26 关闭了这个 draft，没有合入。目前通用放开 EAGLE 的 PR 只剩 #40857。
- **新发现 [#38292](https://github.com/sgl-project/sglang/pull/38292)：** "L2-Only decode-side radix cache"，open，09-28 仍有更新，assignee hzh0425 已加上 `run-ci` 标签。
  - 设计：decode 端的前缀缓存只放在主机内存，传来的 KV 先暂存到主机，HBM 只保留正在运行的请求需要的部分。
  - 前提条件：需要 decode radix、HiCache，以及 `--hicache-mem-layout layer_first`；我们用的是 `page_first`。
  - 作者的数据：Qwen3-32B、131K 上下文下，TTFT 和准入明显改善，ITL 不变。
  - 拆分：已拆成 6 个堆叠 PR（[#40314](https://github.com/sgl-project/sglang/pull/40314)–#40319）。其中 1/6 自 09-19 提交后没有评审，也还没有 `run-ci` 标签。
  - 同步开销：它**没有**减少现有的每轮同步。diff（#40319，即 6/6）没有改动 `check_hicache_events()`，没有删除任何 all_reduce，反而新增了 9 处 `_all_reduce_attn_groups`，都只在 L2-Only 模式下执行。`SGLANG_DISAGGREGATION_UNIFORM_HICACHE_POLLING` 的作用是保证各 rank 调用一致：原来 `is_load_back_event_done()` 在本地加载完成时才会调用 `loading_check()`（内含 all_reduce），开关打开后改为每轮在所有 rank 上固定调用一次。它的性能对比中两边都开着 HiCache，所以也测不出我们在 C 组看到的、相对于不开 HiCache 的每步开销（见 [PERF-AB.zh-CN.md](PERF-AB.zh-CN.md)）。
  - 与投机解码：它没有涉及。decode radix 与 EAGLE 的互斥检查同样挡住了 L2-Only + MTP。

## 结论

**否，还没有进入上游 main；但已有在审 PR 覆盖了其中一部分。**

- main 仍然拒绝 decode radix cache 与 EAGLE、EAGLE3、NEXTN 同时开启。唯一放开的投机算法是 DSPARK，由 [#40263](https://github.com/sgl-project/sglang/pull/40263) 于 2026-09-19 合入。main 上没有任何通用放开，也没有环境变量或开关。
- decode HiCache 与 MTP 在 main 上同样不可用。原因与基线相同：PD 钩子强制 decode 关闭 radix，之后与 HiCache 的互斥检查报错。
- 通用放开 EAGLE/EAGLE3 的 PR 有两个正在进行：
  - [#40857](https://github.com/sgl-project/sglang/pull/40857)：open，2026-09-23 创建。ShangmingCai 评审意见为 "disagg part LGTM"，CI 三项均失败。
  - [#40681](https://github.com/sgl-project/sglang/pull/40681)：draft，作者为维护者 kpham-sgl。

  两者都修复了 SWA 模型上 EAGLE bigram key 引起的 KV 双重归属问题，然后把 EAGLE/EAGLE3 加入白名单。
- 上述 PR 都没有覆盖以下几点：
  - `NEXTN` 原始别名（PD 钩子看到的是未归一化的 CLI 值）；
  - decode HiCache + MTP；
  - DSA 模型；
  - ROCm；
  - 带真实多 token 接受率的准确率证据。

  其中最后一项已被 [#39150](https://github.com/sgl-project/sglang/pull/39150) 明确列为未验证项：该 PR 的验证中 draft token 接受数为 0，作者因此保持 draft 状态并关闭了它。我们的 B/C 组结果正好补上这些证据。
- 我们的补丁 `01-sglang-decode-radix-allow-eagle.patch` **不能原样应用到 main**：第 2 个 hunk 要删除的那一行已被 #40263 改掉。

## 1. main 的现状

| 项目 | main（`4142235c2b`） | 基线（`a9fb1c3238`） |
|---|---|---|
| decode radix 与投机解码的检查 | [`pd_disaggregation_hook.py` L97-L102](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L97-L102)：`if cfg.speculative_algorithm not in (None, "DSPARK"): raise ValueError(...)` | [L89](https://github.com/sgl-project/sglang/blob/a9fb1c3238/python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L89)：`if cfg.speculative_algorithm is not None:` |
| 通用放开或开关 | 无。`environ.py` 中没有 decode radix 相关变量。`SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE` 只出现在未合入的 DSV4 PR [#27831](https://github.com/sgl-project/sglang/pull/27831) 的描述里。`_allow_dsv4_decode_radix_speculative` 在 issue/PR 搜索中无结果，main 的 PD 钩子和 `environ.py` 中也没有（**未验证**：GitHub 代码搜索需要登录，无法确认它是否存在于某个 fork 或分支中） | 同左 |
| 钩子顺序 | PD 钩子在 [`pipeline.py` L127](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/pipeline.py#L127) 运行，投机钩子在 [L285](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/pipeline.py#L285)。NEXTN→EAGLE 的归一化在 [`speculative_hook.py` L108-L115](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/speculative_hook.py#L108-L115)，只在 L153 被调用。所以 PD 钩子看到的是原始的 `NEXTN` | 相同（L160 / L326） |
| decode HiCache + MTP | 没有直接禁止它的检查：[`hicache_hook.py` L123-L128](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/hicache_hook.py#L123-L128) 只在 `dcp_size > 1` 时限制为 DSPARK；[`kv_cache_builder.py` L296-L321](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/mem_cache/kv_cache_builder.py#L296-L321) 只拒绝 SWA + HiCache、DSV4 和 SWA-compress 模型，GLM-5.2（DSA、全注意力）不在其中。但是 PD 检查使 decode 被强制设为 `disable_radix_cache=True`，随后 [`kv_cache_hook.py` L393-L397](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/arg_groups/kv_cache_hook.py#L393-L397) 报 "mutually exclusive"。结论：**不可用** | 相同（`kv_cache_hook.py` L168） |
| MTP draft 层的 HiCache 打包 | `_build_hicache_draft_plan` 未变（[`base_spec_worker.py` L253](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/speculative/base_spec_worker.py#L253)） | L256 |
| ROCm 上的 decode retraction | 在 HIP 上，或者开启 decode radix 时，固定使用 `cpu_tensor` 备份（[`kv_cache_builder.py` L200-L215](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/python/sglang/srt/mem_cache/kv_cache_builder.py#L200-L215)） | 相同 |
| 现有测试 | [`test_server_args.py`](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/test/registered/unit/server_args/test_server_args.py#L1128-L1172) 中有 decode radix 拒绝 hisparse、拒绝 fake 后端、允许 `mooncake_tcp` 的测试，**没有**投机解码相关的测试。e2e 测试 [`test_disaggregation_decode_radix_cache.py`](https://github.com/sgl-project/sglang/blob/4142235c2bfa3b7ffd9945ae1b6a17bb92a9fe4f/test/registered/disaggregation/test_disaggregation_decode_radix_cache.py) 和 `_swa.py` 只注册在 CUDA CI 上（8-gpu-h20 / 8-gpu-h200），没有投机解码的变体，也没有 AMD 的注册 | — |

## 2. 相关 PR 与 issue

状态以 2026-09-24 网页或 search API 返回为准。"覆盖" 指的是我们的场景：GLM-5.2 DSA + EAGLE topk 1 + DPA + Mooncake + ROCm，以及 decode HiCache + MTP。

### 2.1 通用：decode radix + 投机解码

| PR | 状态 / 日期 | 作者 | 内容 | 覆盖 | CI / 评审 |
|---|---|---|---|---|---|
| [#40857](https://github.com/sgl-project/sglang/pull/40857) Fix decode radix cache ownership and allow EAGLE/EAGLE3 | open，09-23 | pyc96 | `UnifiedRadixCache.cache_unfinished_req` 改为用最终插入路径上的 canonical indices 重新指向请求，不再重新 match，并断言覆盖整个页对齐区间，以修复 SWA 模型上的双重归属问题。白名单改为 `(None, "DSPARK", "EAGLE", "EAGLE3")`，同时更新 help 文案并增加 1 个 URC 单元测试。验证环境为 gpt-oss-120b（SWA）+ EAGLE3（topk 1），PR 描述中的 Accuracy 部分为空 | 使用 `--speculative-algorithm EAGLE` 时，参数层面覆盖；不含 `NEXTN`；不限制 topk；没有 HiCache、DSA 或 ROCm 验证 | Base/Extra/AMD 三项均失败；ShangmingCai："disagg part LGTM"；assignee 为 ispobock |
| [#40681](https://github.com/sgl-project/sglang/pull/40681) Keep a full SWA window below the EAGLE bigram insert boundary | draft，09-22 | kpham-sgl | 从另一个角度修同一个 SWA 问题：`_swa_tail_len` 考虑 bigram key，并断言 rematch 覆盖插入区间。白名单同样放开 EAGLE/EAGLE3。新增 e2e 测试 `TestDisaggregationDecodeRadixCacheSWAEagle3Nixl`；在 H200 上用 gpt-oss 验证，GSM8K 为 0.938/0.956 | 同上 | 三项均失败；draft |
| [#39150](https://github.com/sgl-project/sglang/pull/39150) Allow decode radix caching with EAGLE/EAGLE3 | 已关闭未合入，09-11→09-12 | ByronHsu | 只改检查和 help 文案。作者说明其验证中 draft 接受数为 0，"Multi-token acceptance remains unvalidated" | — | draft 后关闭 |
| [#32170](https://github.com/sgl-project/sglang/pull/32170) Fix decode-side release crashes and allow decode radix cache with spec | open，07-23，最后更新 07-27 | caijixueIT | 修复 `DecodeKVCacheOffloadManager` 的释放路径（decode offload 配置），并把投机算法的报错整体改为 EXPERIMENTAL 警告，对所有算法生效 | 参数层面覆盖，但不区分算法和模型；没有测试 | 两项失败；无人评审；评论中有人问 GLM-5.2 能否使用（无回复） |
| [#36686](https://github.com/sgl-project/sglang/pull/36686) [wip] decode radix cache and spec compatible | open，08-27 | heziiop | 只删除 6 行（去掉检查），没有描述 | — | 三项失败 |
| [#37725](https://github.com/sgl-project/sglang/pull/37725) Support NEXTN with HiCache in the PD decode radix cache | open，09-03 | number-eleven11 | 放开 EAGLE/NEXTN，并在 DSV4 钩子中接受原始 `NEXTN` 别名。DSV4 的 decode radix 只能与 HiCache 一起使用。另外修复 CP 下只传后缀时的偏移。验证：DSV4-Flash + NEXTN，GSM8K 97.0% | 部分覆盖；NEXTN 别名的处理思路可以借鉴；以 DSV4 为主 | 三项失败；无人评审 |
| [#40263](https://github.com/sgl-project/sglang/pull/40263) Allow decode radix cache and HiCache L1/L2 with DCP | **merged 09-19** | kpham-sgl | 放开 DSPARK（描述原文："EAGLE and other spec algorithms stay rejected"），同时放开 DCP 下的 decode radix 和 HiCache L1/L2，以及 hybrid SSM 的 decode radix | 不覆盖，但它是"按算法加入白名单、不设环境变量"的先例 | 合入时三项 CI 均为失败状态 |
| [#19746](https://github.com/sgl-project/sglang/pull/19746) support decode side radix cache | merged 05-01 | ishandhanani | 引入 decode radix cache 和投机解码检查。评论区 06-11、07-20、07-22 三次有人追问为何与投机解码不兼容，均无回复 | — | — |

### 2.2 DeepSeek-V4、ROCm 相关（模型专用）

| PR | 状态 / 日期 | 作者 | 内容 | 覆盖 |
|---|---|---|---|---|
| [#27831](https://github.com/sgl-project/sglang/pull/27831) DSV4 decode radix + MTP | open，06-10，最后更新 08-27 | zhangxiaolei123456 | 通过 `SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE=1` 开启 DSV4 decode radix，支持 EAGLE topk 1；+1440 行 | 不覆盖（DSV4 压缩 KV）；缺 `run-ci` 标签 |
| [#31097](https://github.com/sgl-project/sglang/pull/31097) DSV4 decode radix cache with MTP support | open，07-14，最后更新 08-27 | TobyMint | 把 #27831 移植到上游 API；+1038 行 | 不覆盖；CI 失败 |
| [#35837](https://github.com/sgl-project/sglang/pull/35837) Decode radix for DSV4 (DSA compressed KV) | open，08-21 | TobyMint | 在 CUDA 上放开 DSV4 和 SWA-compress 模型，仅限非投机解码；修复 nixl 全命中死锁 | 不覆盖；缺 `run-ci` 标签 |
| [#30929](https://github.com/sgl-project/sglang/pull/30929) DSV4 decode radix on ROCm unified KV | open，07-12，最后更新 09-08 | AMD-yanfeiwang | ROCm DSV4；排除投机解码；修复 SWA 锁归属问题 | 不覆盖；CI 失败 |
| [#31024](https://github.com/sgl-project/sglang/pull/31024) URC `query_storage_hit_length` | open，07-13 | AMD-yanfeiwang | 为 URC 补上 decode HiCache 需要的方法 | 基线和 main 的 URC 中已有该方法（基线 L1876，main L1932），此 PR 实际已被取代 |
| [#26288](https://github.com/sgl-project/sglang/pull/26288) [PD][AMD] incremental KV transfer with decode radix | merged 06-13 | inkcherry | 把 decode radix 扩展到 mori 后端 | 不覆盖 Mooncake；说明 ROCm 上的非投机 decode radix 已有人在用 |

### 2.3 decode HiCache，以及与 radix 相关的其他 PR

| PR / issue | 状态 | 与本任务的关系 |
|---|---|---|
| [#26227](https://github.com/sgl-project/sglang/pull/26227) decode HiCache prefetch + incremental transfer | merged 06-02 | 我们的 C 组依赖它（`decode_hicache_mixin`） |
| [#27770](https://github.com/sgl-project/sglang/pull/27770) SWA hybrid decode radix（unified tree） | merged 08-21 | 不涉及 GLM-5.2；#40857 和 #40681 修的正是它留下的 SWA 路径 |
| [#35694](https://github.com/sgl-project/sglang/pull/35694) keep speculative overshoot out of the radix cache key | open，09-24 仍在更新 | 通用的投机解码 + radix 修复：verify 一次提交的 token 可能超过停止点，这部分不应进入 key。对 decode 树同样适用；影响命中率，不影响正确性 |
| [#38978](https://github.com/sgl-project/sglang/pull/38978) request-owned speculative KV | merged 09-23 | 这里的 "speculative" 指 optimistic prefill，不是投机解码；改动 prefill 侧和 URC 的 `advance_unpublished_req`，与我们的补丁不冲突 |
| [#40075](https://github.com/sgl-project/sglang/pull/40075) Release up to `owned_kv_len` | merged 09-18 | `kv_len_to_handle` 改名为 `owned_kv_len`，`effective_kv_committed_len` 改名为 `owned_kv_len()`。SOLUTION 第 2 节引用的是旧名字 |
| [#40238](https://github.com/sgl-project/sglang/pull/40238) decode host receive | merged 09-24 | 大幅重写 `decode.py`；默认关闭；开启时要求 host pool 为 `layer_first` 布局，与 C 组使用的 `page_first` 不兼容 |
| [#38212](https://github.com/sgl-project/sglang/pull/38212) / [#39156](https://github.com/sgl-project/sglang/pull/39156)（已关闭）/ [#40134](https://github.com/sgl-project/sglang/pull/40134) | open / closed / open | 修复 GLM-5.3-Flash（hybrid DSA + Mamba，压缩 indexer）在 HiCache 中恢复 DSA index 的问题。GLM-5.2 是普通 DSA；C 组已确认 indexer 主机池为 79 层（含 draft 层），所以不直接相关，但值得关注 |
| issue [#30322](https://github.com/sgl-project/sglang/issues/30322) | open | decode radix + HiCache 在 decode 重启后 L3 加载失败；修复 PR #32278 的状态**未核实**。我们没有配置 L3 |
| issue [#28771](https://github.com/sgl-project/sglang/issues/28771) | 因不活跃被关闭 | GLM-5.1 + EAGLE + HiCache + DSA 下接受长度随时间下降。维护者回复称这是"已知且较根本的问题，有缓解措施"。C 组只跑了短时间测试，没有覆盖这一点 |
| issue [#32459](https://github.com/sgl-project/sglang/issues/32459) | open | GLM-DSA 聚合部署下，EAGLE 使多轮前缀复用率下降（97% → 40-53%）；候选修复为 [#32574](https://github.com/sgl-project/sglang/pull/32574)（open） |
| issue [#38031](https://github.com/sgl-project/sglang/issues/38031) | open | GLM-5.3-Flash 的 HiCache host 加载回显存后输出损坏，与投机解码无关 |

## 3. 差距分析

### 3.1 补丁能否直接应用到 main

在内存中逐个 hunk 比对（不写文件）：

- hunk 1（加入 `_decode_radix_spec_allowed`）可以在第 18 行精确匹配。
- hunk 2 **失败**：它要删除 `if cfg.speculative_algorithm is not None:`，而 main 上这一行已被 #40263 改为 `not in (None, "DSPARK")`。

手工 rebase 时必须保留 DSPARK，否则会让 #40263 放开的 DSPARK 重新被拒绝。main 的 `pd_disaggregation_hook.py` 自基线以来还有两处无关改动：role switch（[#28403](https://github.com/sgl-project/sglang/pull/28403)）和 host receive（#40238）。

### 3.2 我们的补丁与 #40857 对比

| 项 | 我们的补丁 | #40857 | main |
|---|---|---|---|
| 允许的算法 | EAGLE、NEXTN（先转大写再比较） | EAGLE、EAGLE3、DSPARK | DSPARK |
| topk | 仅限 1 | 不限制 | — |
| 开关 | `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` | 无 | 无 |
| 树侧修复 | 无。GLM-5.2 只有 FULL 组件，插入后重新 match 能拿回整个区间，因此不需要 | 有，针对 SWA 模型 | — |
| 模型限制 | **无**：设置环境变量后，SWA 模型也会被放开，会触发 #40857 描述的双重归属问题 | 通过修复树的行为来支持 SWA | — |
| 测试 | 没有上游风格的测试（只有 Infera 侧的测试） | 1 个 URC 单元测试 | — |
| 验证 | GLM-5.2 DSA，MI355X，Mooncake，DPA，真实 MTP；GSM8K；decode HiCache 从主机加载回显存 | gpt-oss SWA + EAGLE3 多轮对话；没有准确率数字 | — |

### 3.3 达到上游质量还需要补的内容

1. **NEXTN 别名。** 如果 #40857 合入，`--speculative-algorithm NEXTN` 仍会被拒绝，因为 PD 钩子早于别名归一化运行。可以把 `NEXTN` 加入白名单，或者在检查前调用 `_resolve_speculative_algorithm_alias`。后者需要读取 draft 配置，有代价；#37725 采用的是前一种思路。
2. **SWA 模型。** 放开检查必须与 #40857 或 #40681 的树修复一起提交；否则只能对非 SWA 模型（全注意力、MLA、DSA）放开。
3. **topk > 1。** 所有上游和我们的验证都使用 topk 1。理论上，树形 draft 在 verify 后只把被接受路径写入 `req_to_token`，radix 只看已提交的 KV，应该也是安全的。但目前**没有人验证过**。可以二选一：限制为 topk 1 并给出明确报错；或者补一个 topk > 1 的 e2e 测试。
4. **draft KV 与接受长度。** 需要给出命中与未命中时接受长度的对比，这正是 #39150 留下的空白。我们在 C 组看到接受长度约 3.5–4.0，但 **B 组没有按命中和未命中分开统计**，待补。此外，我们的部署中 prefill 没有 draft 模型；上游常见配置是 prefill 也运行 EAGLE。这两种配置下 prompt 位置 draft KV 的来源不同，需要在 PR 中说明。
5. **retraction。** decode radix 下，retraction 后重新 bootstrap 时不做前缀匹配（代码中有 TODO）；HIP 上固定使用 `cpu_tensor`。B/C 组都没有发生 retraction，这条路径**未覆盖**。
6. **测试。**
   - server-args 单元测试：EAGLE、EAGLE3、NEXTN 被接受；其他算法被拒绝；DSPARK 仍被接受；decode 开启 radix 后，decode HiCache + EAGLE 能通过 `handle_cache_compatibility`。
   - URC 单元测试：现有测试已按 `is_eagle` 参数化，可以扩展。
   - e2e：在 `test_disaggregation_decode_radix_cache.py` 中增加一个 EAGLE3 + Mooncake 类（main 的 `test_utils` 中已有 `DEFAULT_TARGET_MODEL_EAGLE3` / `DEFAULT_DRAFT_MODEL_EAGLE3`），可选加一个 NEXTN 类（`DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN`），并断言命中数、GSM8K 分数和平均接受长度 > 1。
   - 可选：decode HiCache + EAGLE 的变体，把显存池调小以强制淘汰。
   - 可选：AMD CI 注册（#38978 的 CI 报告中提到 `stage-b-test-large-8-gpu-mi35x-disaggregation-amd` 这个 job）。
7. **overshoot。** 与 #35694 正交，不阻塞本改动，但在 PR 中应注明。

### 3.4 基线之后上游的相关改动（rebase 时需要注意）

- #40263：检查行被改写（与我们的 hunk 2 冲突）；DCP 下的限制被放开。
- #40075：改名 `owned_kv_len`，retraction 相关函数改名为 `backup_kv_cache` / `restore_kv_cache` / `discard_kv_cache_backup`。只影响触及这些函数的本地补丁。
- #40238：`decode.py` 有约 760 行改动，新增 host receive（默认关闭）。
- `decode.py` 中删除了 DSA top-k 播种在 DPA 下强制 draft eager 的逻辑（`requires_dp_attention_eager_forward`），同时在 prebuilt 阶段新增了可选的 KV checksum 校验。**来源 PR 未核实。** 这两处都不针对 radix，但涉及 GLM-5.2 的 MTP + DPA 路径，rebase 后需要重新验证。
- [#40780](https://github.com/sgl-project/sglang/pull/40780) / [#40787](https://github.com/sgl-project/sglang/pull/40787)：去掉 `SGLANG_ENABLE_UNIFIED_RADIX_TREE`，删除 HiRadixCache（main 上已没有 `hiradix_cache.py`）。GLM-5.2 本来就走 URC，没有影响。
- 仍在审、可能影响命中率的 PR：#35694、#32574。

## 4. 建议的上游路径

1. **不要另开一个与 #40857 竞争的通用 PR。** #40857 已有 disagg 评审的认可，#40681 的作者是维护者本人，方向一致：直接加入白名单，不设环境变量，并附带树修复。重复提交只会分散评审精力。
2. **先补证据，再补缺口。** 经团队同意后（本次调研按要求没有发表任何评论），在 #40857 下提交我们的验证数据。这可以补上它空着的 Accuracy 部分和多 token 接受率的空白，并指出 `NEXTN` 别名会被拒绝。
3. **在 #40857 合入后提交一个小的后续 PR**（草稿见 [`patches/upstream-pr-draft.md`](../patches/upstream-pr-draft.md)），内容包括：
   - 白名单加入 `NEXTN`；
   - 不设环境变量，保留 EXPERIMENTAL 日志（与 DP attention 的警告风格一致）；
   - 以下测试：server-args 测试（投机算法矩阵，以及 decode HiCache + EAGLE 的兼容性）；Mooncake 上的 EAGLE3 e2e 测试，断言命中数、GSM8K 分数和接受长度；可选的 decode HiCache 变体和 AMD 注册。

   topk 的处理：默认跟随 #40857，不限制 topk，并在 PR 中写明只验证了 topk 1；如果评审要求保守，再改为 topk > 1 直接报错。
4. **如果 #40857 长期没有进展**（例如两周以上），可以改提一个范围更窄的独立 PR：只对非 SWA 模型放开 EAGLE、EAGLE3 和 NEXTN，SWA 模型在有投机解码时继续拒绝。这样不依赖树修复，也不会触发 SWA 的双重归属问题。
5. **提交前的内部准备：**
   - 在 rebase 到 main 的镜像上重跑 B 组和 C 组，因为 `decode.py` 和 DSA/MTP 路径都有改动。
   - 补上命中与未命中的接受长度对比。
   - 等性能 A/B 结果出来。
   - 在 CUDA 上用小模型（Llama-3.1-8B + EAGLE3，Mooncake）复现一次。维护者的 CI 以 CUDA 为主，只有 ROCm 证据可能不够。
