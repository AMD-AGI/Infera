# 在 PD + DPA + MTP 上打通 kv-aware：三个 bug 与完整 workflow

配套 [`REPORT.zh.md`](REPORT.zh.md)（§5 是本文的摘要版）。本文只讲 kv-aware 这一条线：
三个 bug 的现象/根因/修复，以及从启动到出数的完整操作流程。

环境：两台 8×MI325X（gfx942），prefill `10.32.17.210:30001`（g46）、decode
`10.32.17.209:31001`（g45），router + etcd 在 g46。GLM-5.2-FP8，TP8 + DP-attention，
MTP=1（EAGLE 3 步 / topk 1 / 4 draft token），KV 走 Mooncake RoCE。

---

## 0. 结论摘要

kv-aware 现在在 PD + DPA + MTP 上完整工作：router 的缓存镜像能填上，重复前缀拿到
**333/333 全命中**，agentic 负载的缓存效率从 57.3% 提到 **88.2%**、TTFT p50 从 14.6s
降到 **2.5s**，TPOT 不变（MTP 未受影响）。贴近客户生产形状的 profile（p50 91K / p90 380K
prompt、42% 新会话）同样跑通，引擎侧零错误，可命中请求的 token 加权效率 **95.0%**（§3.1）。

前三个 bug 挡住了 kv-aware 本身，**互不相关**却串成一条链——前一个不修，后一个根本不会显形。
后三个是跑生产 profile 时暴露的，都在压测工具侧：不影响服务正确性，但会让你把好结果读成坏结果。

| # | bug | 层次 | 现象 | 处理 |
|---|---|---|---|---|
| 1 | KV 事件端口撞进 K8s NodePort 区间，被 IPVS 截走 | 端口分配 | 镜像恒空，**无任何报错** | 已修 `infera/common/net.py` |
| 2 | MTP 下 `token_ids` 是 bigram 对，解码失败且不能直接哈希 | Python router | 日志 `kv decode failed: Expected int, got array` | 已修 `infera/router/kv_event/{events,client}.py` |
| 3 | 同一个 bigram 问题在 Rust router 里被静默丢弃 | Rust router | 镜像恒空，**连警告都没有** | 已修 `rust/router/src/kv_event.rs` |
| 4 | bench 客户端 240s 硬超时，装不下 300K+ 的 prefill | AgenticBench | 请求被判失败，**引擎侧无任何错误** | 待上游改，见 §1.4 |
| 5 | `efficiency` / `eviction_rate` 把物理上不可能命中的首轮算成损失 | AgenticBench 指标 | 表头效率 35.3%，实则 95.0% | 换口径读数，见 §1.5 |
| 6 | 生产 profile 的 `tokenizer` 是占位符路径 | workload yaml | 启动即 `ERROR: Tokenizer path does not exist` | 命令行覆盖，见 §1.6 |

一个贯穿全文的性质，先说在前面：**router 侧的命中判断算错也不会影响正确性。** router 只
决定把请求送到哪个 rank，引擎自己会对 token 做精确前缀匹配，哈希误判的代价是白跑一次
prefill，不会返回错内容。所以 kv-aware 是纯性能特性。

---

## 1. Bug 清单

§1.0–1.3 是挡住 kv-aware 的三个 bug（已修），§1.4–1.6 是跑生产 profile 时暴露的压测工具问题。

### 1.0 先说一个被证伪的假设，免得别人再走一遍

最初的怀疑是 **ZMQ 线程安全**：SGLang 的 `ZmqEventPublisher` 在主线程 `_socket_setup()`
里创建并 bind PUB socket，却在发布线程里 `send_multipart`，用的还是进程级共享的
`zmq.Context.instance()`。这个组合确实可疑，而且当时的证据看起来吻合：引擎日志（插桩）
明确打出 `sent seq=1 bytes=104583`，而连在同一端口、订阅空 topic 的独立 SUB 收不到任何东西。

写了个复刻脚本证伪它：让发布器先空转 6 秒、订阅者后连（和引擎里的时序完全一致）——

```
publisher up on tcp://*:42780
idling 6.0s before any subscriber connects...
subscriber connected
published a 217-event batch
RECEIVED
```

**帧正常收到。** 所以发布器和它的线程模型都没问题，方向错了。

教训：`sent` 这条日志只证明 `send_multipart` 返回了，不证明"这个地址上的订阅者能收到"。
下一步应该换的变量不是线程，而是**地址**。

### 1.1 Bug 1：KV 事件端口撞进 Kubernetes NodePort 区间，被 IPVS 吃掉

#### 现象

router 一切就绪、引擎也确实在发，但镜像全是 0，而且**没有任何一侧报错**：

| 环节 | 状态 | 证据 |
|---|---|---|
| 引擎记录事件 | ✅ | 插桩 `took 217 events`（累计 396），数量与 block 数吻合 |
| 引擎发帧 | ✅ | 插桩 `sent seq=1 bytes=104583 ep=tcp://*:32760` |
| 端口归属 | ✅ | `ss -ltnp` 显示 `0.0.0.0:32760` 属于 pid 349661，正是记下 `sent` 的那个 scheduler |
| router 订阅 | ✅ | `kv events: subscribing to ... (block_size=64, ranks=8)`，8 条 ESTAB |
| router 镜像 | ❌ | `cache-view` 8 个 rank 全 `block_count: 0` |
| 路由决策 | ❌ | pick 日志恒 `cache_hits=0` |

#### 定位：把地址变成唯一变量

单地址分两次跑是有陷阱的——帧只在请求在飞时出现，"没收到"和"错过了窗口"分不开（我第一次
就踩了：引擎 `sent seq=4` 的时间正好压在 90s 轮询边界上）。所以在**同一个进程里同时订阅
loopback 和节点 IP**，发一条请求，同一瞬间对比：

| SUB 连的地址 | 结果 |
|---|---|
| `tcp://127.0.0.1:32760` | ✅ 1 帧，73 KB，topic `kv-events`，msgpack 里是 `BlockStored` |
| `tcp://10.32.17.210:32760` | ❌ 0 帧 |

再往下一层，直接用裸 socket 试连，反差更干脆：

```python
socket.create_connection(("127.0.0.1", 32760))       # OK
socket.create_connection(("10.32.17.210", 32760))    # ConnectionRefusedError
```

**监听是 `0.0.0.0`，同一个 netns（两者 `/proc/*/ns/net` 都是 `4026531840`），
路由表还写着 `local 10.32.17.210 dev lo`——却是 connection refused。**
这个组合只能是 netfilter/IPVS 在拦。

#### 根因

```bash
# 解出 /proc/net/ip_vs 里落在 KV 事件端口段的服务
$ python3 - <<'EOF'
import re
for line in open("/proc/net/ip_vs"):
    m = re.match(r"^TCP\s+([0-9A-F]{8}):([0-9A-F]{4})", line)
    if m and 32755 <= int(m.group(2), 16) <= 32767:
        ip = ".".join(str(int(m.group(1)[i:i+2], 16)) for i in (0, 2, 4, 6))
        print(f"{ip}:{int(m.group(2), 16)}")
EOF
10.32.17.210:32760      # real server 列表为空
10.36.1.210:32760       # 本机每个网卡地址都有一条
172.16.3.1:32760
...
```

集群里有个 Service 占用了 NodePort **32760**，kube-proxy（IPVS 模式）就在本机所有地址上
建了对应的 IPVS 服务，而本节点没有它的 endpoint，于是 real server 列表为空。**IPVS 在数据包
到达本地 `0.0.0.0` 监听之前就把它截走，内核回 RST。** `127.0.0.1` 不在 IPVS 的绑定地址里，
所以只有 loopback 通——而 router 用的恰恰是引擎 advertise 出去的节点 IP。

端口是怎么选到 32760 的：`free_tcp_port_block()` 从 `ip_local_port_range` 下界往下扫，
`32768 - 8 = 32760`，**整块都落在 NodePort 区间（30000-32767）内**。更麻烦的是它只用
`127.0.0.1` 试 bind——而 IPVS 拦截既不影响 bind、也不影响 loopback 连通，所以这个探测
**原理上就发现不了**这类端口："能 bind、loopback 能连、从节点 IP 打不通"。

这也解释了为什么整条链一声不响：ZMQ 的 `connect()` 是异步重试语义，连不上不报错；PUB 端
没有订阅者就静默丢弃。两边都觉得自己没问题。

#### 修复

`infera/common/net.py`：扫描时绕开 NodePort 区间，并把这个外部约束写在注释里（它不是从
代码能看出来的）。

```python
# Kubernetes' default --service-node-port-range. A port in this window can be
# claimed cluster-wide by any Service at any time; kube-proxy in IPVS mode then
# creates an IPVS service for it on *every* node address, with no real server
# backing it on this node. Traffic to a node IP is then swallowed by IPVS before
# it reaches a local 0.0.0.0 listener and the kernel answers RST, so the port
# stays bindable and reachable over loopback while being "connection refused"
# from the node IP -- which is the address we advertise to peers. Override with
# INFERA_NODEPORT_RANGE="lo-hi" (or "none") for clusters that moved the range.
_NODEPORT_RANGE_DEFAULT = "30000-32767"


def _reserved_nodeport_range() -> tuple[int, int] | None:
    spec = os.environ.get("INFERA_NODEPORT_RANGE", _NODEPORT_RANGE_DEFAULT).strip()
    if spec.lower() in ("", "none", "off"):
        return None
    try:
        lo, _, hi = spec.partition("-")
        return int(lo), int(hi)
    except ValueError:
        return None
```

扫描循环（`net.py:65-71`）：起点同时压到 NodePort 区间下方，并逐个跳过与该区间重叠的块
（后者在默认配置下冗余，但集群把区间挪到别处时就需要）：

```python
    reserved = _reserved_nodeport_range()
    start = low - count
    if reserved:
        start = min(start, reserved[0] - count)
    for base in range(start, 1024, -1):
        if reserved and base + count - 1 >= reserved[0] and base <= reserved[1]:
            continue
```

效果：dp8 下基址 `32760 → 29992`。`free_tcp_port()`（单 rank 路径）不用改——内核分配的
临时端口在 32768 以上，本来就在 NodePort 区间之外。

测试 `tests/unit/common/test_net_ports.py`：钉住"返回的块不与区间重叠"（含自定义区间、
`none` 关闭、畸形值回退），以及块连续可 bind。

### 1.2 Bug 2：MTP 下 `token_ids` 是 bigram 对，router 解不开

#### 现象

Bug 1 修完、帧终于到达 router 的**同一秒**，真正的第二个问题冒出来：

```
WARNING:infera.router.kv_event.client:kv decode failed for 10.32.17.210:30001:
    Expected `int`, got `array` - at `$[1][0][3][0]`
```

`$[1]` 是 batch 里的 events 列表，`[0]` 第一个事件，`[3]` 该事件的第 4 个字段
（tagged array 布局 `[tag, block_hashes, parent_block_hash, token_ids, ...]`，
也就是 `token_ids`），`[0]` 它的第一个元素——**期望 int，实际是数组**。

#### 根因

SGLang 自己的结构体声明是 `token_ids: list[int]`，但 `mem_cache/events.py` 往里塞的是
别的东西（msgspec 编码不做类型校验，所以声明与实际不符也不会报错）：

```python
# sglang/srt/mem_cache/events.py:58-68
is_bigram = node.key.is_bigram
raw = node.key.token_ids
...
    # Preserve historical event payload: bigram pages expose tuples.
    if is_bigram:
        page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
    else:
        page_tokens = list(raw[start:end])
```

而 `is_bigram` 的来源是 `RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle)`
（`unified_radix_cache.py:763`）：**开了 MTP/EAGLE，radix 树就按 bigram 建键**，事件里
每个 block 的 token 于是变成重叠的 `(t[i], t[i+1])` 对。

所以这不是偶发的版本兼容问题，而是 **MTP 与 kv-aware 的必然交点**：两个一起开就会踩到。
`RadixKey` 的语义（`radix_cache.py:60-98`）：`is_bigram=True` 时 `token_ids` 存的还是原始
token（N 个 bigram 对应 N+1 个 token），`len(key)` 是 `raw_len - 1`，第 j 个 bigram 是
`(t[j], t[j+1])`。

#### 关键：不只是"放宽解码类型"

router 给请求算哈希用的是**扁平** token 序列（`hash_request` 按 block_size 切块、链式
XXH3）。如果把 bigram 对原样喂进哈希，算出来的视图和任何请求都不会匹配——错误会从"解码
报错"变成"永远 0 命中"，更难查。

正确做法是**取每对的第一个元素**：一页覆盖 bigram 位置 `[start, end)`，各对的首元素就是
`raw[start:end]`，恰好是请求侧看到的那一段；而 radix 节点按页边界分裂
（`page_aligned(page_size)`），节点起点是 64 的整数倍，两边的切块因此天然对齐。

#### 修复

`infera/router/kv_event/events.py:81`——SGLang 侧 schema 放宽，并说明为什么会有两种形状：

```python
class SglangBlockStored(_SglangKVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    # With EAGLE/MTP the radix key is a bigram view (``RadixKey.is_bigram``, set
    # from ``is_eagle``), and the engine reports a block's tokens as the
    # overlapping pairs ``(t[i], t[i+1])`` instead of bare ints -- so this field
    # is list[int] on a plain engine and list[tuple[int, int]] under MTP. See
    # KvEventClient._flat_tokens for how the pairs map back onto flat tokens.
    token_ids: list[int | tuple[int, int]]
```

vLLM 侧那份 schema 不动（没有 bigram 概念）。

`infera/router/kv_event/client.py:207-220`——哈希前折平：

```python
    @staticmethod
    def _flat_tokens(token_ids: list) -> list[int]:
        """Flat token ids for a stored block, whatever view the engine reports.

        Under EAGLE/MTP SGLang keys its radix tree on bigrams, so a block's
        tokens arrive as the overlapping pairs ``(t[i], t[i+1])``. Taking the
        first element of each pair rebuilds ``t[start:end]`` -- the same flat
        slice ``hash_request`` chunks on the query side, and radix nodes split on
        page boundaries so the two chunkings stay aligned. Hashing the pairs
        as-is would instead produce a view that never matches any request.
        """
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            return [pair[0] for pair in token_ids]
        return token_ids
```

调用点（`client.py:233-238`）：

```python
        bs = sub.block_size
        tokens = self._flat_tokens(ev.token_ids)
        n = len(tokens) // bs
        for i in range(n):
            chunk = tokens[i * bs : (i + 1) * bs]
            parent = hash_chunk(parent, chunk)
```

测试 `tests/unit/router/test_kv_event_e2e.py:240`：走真实 ZMQ + 真实 `KvEventClient` +
真实策略，用 `(1,2),(2,3),(3,4),(4,5)` 这种形状发两个链式事件，断言扁平请求
`[1..8]` 拿到 2 个命中——钉的是"bigram 视图必须和扁平 token 哈希到同一批 block"这个契约，
不是解码不报错而已。

### 1.3 Bug 3：Rust router 同一个 bug，而且更隐蔽

`rust/router/src/kv_event.rs` 是 Python 客户端的孪生实现，同样的哈希链、同样的
per-rank 扇出。它的 `token_ids` 解码是：

```rust
// 修复前
fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| as_u64_any(x).map(|n| n as u32)).collect())
        .unwrap_or_default()
}
```

`as_u64_any` 只认 `Value::Integer`，bigram 对是 `Value::Array` → 返回 `None` →
被 `filter_map` **静默丢弃**。一页 64 个 bigram 于是解出**空** token 列表，
`n = 0 / 64 = 0`，一个 block 都不入视图。**镜像默默保持为空，连 Python 那条
`kv decode failed` 警告都不会有**——如果先在 Rust 后端上遇到，会难查得多。

修复后（`rust/router/src/kv_event.rs:431-452`）：

```rust
/// Flat token ids from a `BlockStored`'s `token_ids`, in either view the engine
/// may report. Under EAGLE/MTP, SGLang keys its radix tree on bigrams and sends
/// each block's tokens as the overlapping pairs `(t[i], t[i+1])`; the first
/// element of each pair rebuilds `t[start:end]`, which is the flat slice
/// `hash_request` chunks on the query side (radix nodes split on page
/// boundaries, so the two chunkings stay aligned). Without this, pairs are not
/// integers, every element is dropped, and the view silently stays empty.
fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| match x {
                    rmpv::Value::Array(pair) => {
                        pair.first().and_then(as_u64_any).map(|n| n as u32)
                    }
                    _ => as_u64_any(x).map(|n| n as u32),
                })
                .collect()
        })
        .unwrap_or_default()
}
```

测试 `decodes_sglang_bigram_batch_under_mtp`（同文件 `:528`）：16 个 bigram 对覆盖扁平
token `1..=17`，断言 `prefix_hits` 对 `hash_request(&seq(1,16), 16)` 返回 1。

本机跑 Rust 测试需要指定 libclang，否则依赖链里的 `onig_sys` 起不来：

```bash
cd rust/router
LIBCLANG_PATH=/opt/rocm-7.2.0/lib/llvm/lib cargo test      # 59 项全过
```

### 1.4 Bug 4：bench 客户端 240s 硬超时，把 300K+ 的请求判成失败

#### 现象

生产 profile 那轮 23 条请求里 1 条失败（`success_rate=0.9565`），但**引擎侧翻不到任何错误**：
prefill / decode 日志没有 abort、没有超限、没有 5xx。bench 自己只打了一行：

```
WARNING: Request timeout - traffic timing may diverge from seed (non-deterministic)
```

#### 根因

超时写死在客户端，与 prompt 长度无关：

```python
# Optimus-AgenticBench agent/agent_throughput.py:929（另有 :2229、:2972 三处相同）
timeout=aiohttp.ClientTimeout(total=240)
```

而这个 profile 的尾部就在 240s 之外。用 `pdops/smoke_longctx.py` 实测的**空队列**单流耗时：

| prompt token | 墙钟 | 结论 |
|---|---|---|
| 125,222 | 45.0s | 安全 |
| 314,039 | 136.6s | 安全但已过半 |
| 374,033 | 170.5s | 空队列就用掉 71% 的预算 |

单流 prefill 吞吐约 2.2–2.8K tok/s，也就是 240s 的理论上限约 530–670K token；一旦并发排队，
380K 的请求就必然突破。profile 自己声明上限 `max_prompt_tokens: 380000`，而客户端预算装不下
它——**profile 与客户端的默认值互相矛盾**。

#### 解决

这是上游问题，不在 Infera 侧。跑长尾 profile 前把三处 `total=240` 按 prompt 上限放大，例如
按 380K / 2.2K tok/s ≈ 173s 再留 3 倍余量取 900s。判断依据很明确：**失败但引擎侧无错、且失败
集中在最长的那几条**，就是撞了这个超时，不要去查引擎。

### 1.5 Bug 5：`efficiency` / `eviction_rate` 把物理上不可能命中的部分算成损失

#### 现象

生产 profile 的表头 `efficiency` 只有 **35.3%**（128K profile 是 88.2%），`eviction_rate`
报 **64.7%**，看着像 kv-aware 在真实负载下失效。但逐请求看，21 条完成里有 9 条是
`0.990/0.990`、`0.984/0.984` 这种几乎满命中。

#### 根因

两个独立的口径问题叠在一起：

1. `eviction_rate` **不是实测淘汰量**。两轮数据都严格满足 `eviction_rate == 1 − efficiency`
   （88.2%/11.8%、35.3%/64.7%），它只是效率的补数，与引擎是否真的淘汰了 KV 无关。
2. `ideal_hit_rate` 把**新会话首轮**的初始前缀也算作"可复用"。首轮那段前缀从未发送过，任何
   缓存都不可能命中，却进了分母。这个 profile `new_session_rate=0.42`，16 个会话有 10 个是
   运行中新建的，光这些首轮就占全部 token 的 **56.4%**（128K profile 里只占 1.1%）。

所以表头效率衡量的是"这个 workload 有多少 token 落在首轮"，而不是"路由做得有多好"。

#### 解决

改读数口径，只统计**能命中的那批**（`actual ≥ 0.05` 即非首轮）的 token 加权效率：

```bash
zcat pdops/results/agentic_4_prod_p50_60k_p90_300k/metrics.jsonl.gz | python3 -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
act = [h for x in rows for h in x.get('new_cache_hit_rates', [])]
idl = [h for x in rows for h in x.get('new_ideal_cache_hit_rates', [])]
pl  = [p for x in rows for p in x.get('new_prompt_lengths', [])]
hit = [(p, a, d) for p, a, d in zip(pl, act, idl) if a >= 0.05]
print('hittable=%d  token-weighted efficiency=%.3f'
      % (len(hit), sum(p*a for p,a,_ in hit) / sum(p*d for p,_,d in hit)))
"
# hittable=12  token-weighted efficiency=0.950
```

同一口径下 128K profile 是 0.917，生产 profile 是 **0.950**——路由在真实负载下反而更好，
因为可命中的请求都是十万级前缀，一旦落对 rank 就几乎全量命中。

### 1.6 Bug 6：生产 profile 的 `tokenizer` 是占位符路径

`code_agent_glm52_p50_60k_p90_300k.yaml:45` 写的是 `tokenizer: /path/to/GLM-5.2-MXFP4`
（该 profile 是为 MXFP4 权重调的）。好在它**快速失败且报错明确**，不会静默用错 tokenizer 去
估算 token 数：

```
Loading tokenizer: /path/to/GLM-5.2-MXFP4
ERROR: Tokenizer path does not exist: /path/to/GLM-5.2-MXFP4
```

解决：命令行传 `--tokenizer /wekafs/models/GLM-5.2-FP8`。同理 `gpus: 8` 也要覆盖成 16
（双节点），否则每 GPU 归一化的吞吐会虚高一倍。

---

## 2. 完整 workflow

### 2.1 前置

两条腿都必须先打两个补丁，否则脚本会拒绝启动（理由见 `REPORT.zh.md` §1.3 / §3.1）：

```bash
cd examples/sglang_glm5.2
bash patch_mooncake_hip.sh   # 关掉 mooncake 的 HIP IPC transport
bash patch_sglang.sh         # KV wait-event barrier + DSA padded-rows
```

ssh 进 pod 的非交互 shell **不继承容器环境**（镜像设了 ~15 个 `SGLANG_*` / `AITER_*`，
丢掉任何一个都会静默改变引擎行为，例如 `SGLANG_USE_AITER` 决定 DSA page size，丢了会让
两条腿的 KV 布局不一致）。所以每个远端会话先：

```bash
source pdops/podenv.sh   # 从 /proc/1/environ 导出容器真实环境
```

### 2.2 启动

顺序：etcd → router → prefill 腿 → decode 腿。两条腿冷启动各约 20 分钟（704 GiB 权重，
MTP 还要再读一遍以抽出 nextn 层做 draft model），**不要中途 kill**。

```bash
# --- g46（prefill 节点，10.32.17.210）---
cd /wekafs/llying/code/Infera/examples/sglang_glm5.2
bash infera_0_etcd.sh
bash infera_1_server.sh              # ROUTER_POLICY 默认 kv-aware

# prefill 腿：KV_EVENTS=1 是 kv-aware 的开关
MTP=1 KV_EVENTS=1 MEM_FRAC=0.80 EXTRA_ARGS=--enable-cache-report \
  bash infera_2_sglang_prefill.sh

# --- g45（decode 节点，10.32.17.209）---
cd /wekafs/llying/code/Infera/examples/sglang_glm5.2
MTP=1 MEM_FRAC=0.80 EXTRA_ARGS=--enable-cache-report \
  ETCD_ENDPOINT=10.32.17.210:2379 bash infera_3_sglang_decode.sh
```

**decode 腿刻意不开 `KV_EVENTS`。** 一开 kv 事件，Infera 就会给 mooncake decode worker
追加 `--disaggregation-decode-enable-radix-cache`（`infera/engine/sglang/args.py:257-263`），
而 SGLang 不接受这个 flag 与投机解码共存，会和 MTP 冲突。关掉之后 decode 的
`kv_block_size` 是 `None`，pick 日志上表现为恒 `cache_hits=0 request_blocks=0`
（连块都切不出来），decode 路由退化成纯按负载——**prefill 仍然拿到
完整的前缀感知路由，收益本来也在那一侧**（prefill 命中能跳过整个 prefill，decode 命中只省
一点负载，这也是 `--kv-prefill-overlap-weight 20.0` 比 decode 的 `2.0` 高 10 倍的原因）。

router 侧关键参数（`infera_1_server.sh` 已内置）：

```
--router-policy kv-aware --kv-event-transport zmq
--kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0
--router-tokenizer-path /wekafs/models/GLM-5.2-FP8
```

### 2.3 观测

**a) 端口基址是否安全**（Bug 1 的日常体检，一眼就能看出来）：

```bash
$ grep -aoE '\{"publisher".*?\}' infera_2_sglang_prefill.log | head -1
{"publisher": "zmq", "endpoint": "tcp://*:29992", "topic": "kv-events"}

$ ss -ltnp | grep -E ":2999[0-9]" | wc -l     # dp8 → 8 个端口
8
```

若基址落在 30000-32767，立刻查 IPVS（见 §1.1 的解码脚本）。

**b) router 是否订阅到正确端点**：

```bash
$ grep -a "kv events: subscribing" infera_1_server.log | tail -1
kv events: subscribing to 10.32.17.210:30001 (block_size=64, ranks=8) at tcp://10.32.17.210:29992
```

**c) 缓存镜像是否非空**（最直接的判据）：

```bash
for r in 0 1 2 3 4 5 6 7; do
  curl -s "http://127.0.0.1:8000/v1/admin/cache-view/10.32.17.210:30001?dp_rank=$r"
done
# {"worker_id":"10.32.17.210:30001","dp_rank":0,"block_count":333}
```

**d) 每条请求的路由决策**：

```bash
$ grep -a "pick policy=kv-aware role=prefill" infera_1_server.log | tail -2
... picked=10.32.17.210:30001#dp0 cache_hits=0   request_blocks=333 active_blocks=0 w_overlap=20.00
... picked=10.32.17.210:30001#dp0 cache_hits=333 request_blocks=333 active_blocks=0 w_overlap=20.00
```

`/metrics` 上对应 `infera_router_pick_cache_hits`、`infera_router_pick_request_blocks`、
`infera_policy_cache_view_size`。

**e) 一条命令跑完整契约**（首次 miss → 镜像填充 → 重发全命中）：

```bash
$ python3 verify_kv_aware.py
cache view before : {0: 333, 1: 0, ...}
[first] wall=7.45s prompt=21359 engine_cached=None
cache view after  : {0: 666, 1: 0, ...}  total=666
[repeat] wall=1.13s prompt=21359 engine_cached=21312
router prefill picks (cache_hits, request_blocks): [(0, 333), (333, 333)]

PASS: full prefix hit on repeat, repeat wall time 1.13s
```

失败时会分别指出是"镜像空"（事件没到）还是"命中不足"（哈希对不齐），退出码非 0。

**一个坑：`cached_tokens` 不能用来验证 kv-aware 是否生效。** 响应里的
`prompt_tokens_details.cached_tokens` 是**引擎侧**的 radix 匹配结果、且按页对齐——实测同一个
19258 token 的 prompt，首发 `prompt_tokens_details` 为 `null`，重发报 `19200 = 300 × 64`。
它说的是"请求落到的那个 rank 上引擎自己匹配到多少"，和 router 以为的命中是两件事：router
视图坏掉时，请求偶然落回原 rank 一样会有高 `cached_tokens`（这也正是对照组仍有 54.3% 命中的
原因）。判断路由本身要看 `infera_router_pick_cache_hits` 和 `cache-view`。

### 2.4 正确性核验

```bash
cd /wekafs/llying/code/inference_glm5p2_sglang
python3 verify_correctness.py \
  --base-url http://127.0.0.1:8000 \
  --model /wekafs/models/GLM-5.2-FP8 \
  --json-out /tmp/verify_kvaware_pd_mtp.json
```

打 router（而不是直连引擎），这样覆盖的是完整的 kv-aware 路由路径。全量套件约 6 分钟。
它是**严格串行**的（每个用例一次阻塞 POST），在飞请求恒为 1。

结果（`结论: 全部通过`）：

| 检查项 | 结果 |
|---|---|
| weights（checkpoint 反量化自洽） | 2/2 |
| basic | 7/7 |
| determinism | 1/1 |
| idle | 3/3 |
| needle（最长 58,695 token） | 9/9 |
| humaneval（短上下文） | 20/20 |
| humaneval-long（8.2-8.5K 上下文） | 19/20 |
| code-retrieval | 2/2 |
| deep-api | 3/3 |

`humaneval-long` 那一例（HumanEval/17，生成的代码抛 `KeyError: ''`）是本轮唯一失败，
套件按 -5% 差值判定"未见长上下文退化"。它不是路由带来的风险：如前所述，router 的命中
判断不参与正确性，引擎会自己做精确前缀匹配。

### 2.5 agentic bench

工具 Optimus-AgenticBench，负载形状 `agent/workloads/code_agent_128k.yaml`：初始前缀均值
40K（中位 34K）、每轮追加均值 2.5K、`new_session_rate=0.04`（**96% 的请求复用已有会话前缀**）、
生成均值 500。选它而不是 `code_agent_glm52_p50_60k_p90_300k.yaml`（初始前缀均值 290K、
中位 58K），是因为后者的会话一开就逼近 128K 上限、很快被 `max_prompt_tokens` 退休，
一轮里跑得出的请求太少，前缀复用来不及体现。

```bash
cd /wekafs/llying/code/Optimus-AgenticBench

# 让三组可比：跑前把两条腿的 radix cache 清空（router 镜像会随 AllBlocksCleared 归零）
curl -s -X POST http://127.0.0.1:30001/flush_cache
curl -s -X POST http://10.32.17.209:31001/flush_cache

agent-bench agent --server http://127.0.0.1:8000 \
  --model /wekafs/models/GLM-5.2-FP8 --tokenizer /wekafs/models/GLM-5.2-FP8 \
  --workload-config agent/workloads/code_agent_128k.yaml \
  --max-qps 0.1 --initial-qps 0.05 --ramp-duration 20 \
  --max-inflight 8 --gpus 16 --dashboard-mode --name kvaware-pd-mtp
```

QPS 压到 0.1、`max_inflight` 压到 8，是因为默认值（0.3 / 16）在当前吞吐下会让队列一直堆积，
测出来的是排队而不是缓存行为。`sustain_duration` 用 yaml 里的 600s，没在命令行覆盖——启动时
它会把生效的覆盖项都打出来，照着核对即可：

```
Loaded workload config from: agent/workloads/code_agent_128k.yaml
  Skipped (CLI override): 6 parameters
    - initial_qps (CLI override: 0.05)   - max_qps (CLI override: 0.1)
    - ramp_duration (CLI override: 20.0) - gpus (CLI override: 16)
    - max_inflight (CLI override: 8)     - tokenizer (CLI override: ...)
```

产物在 `benchmarks/<name>/<timestamp>/{summary.json,metrics.jsonl,metadata.json}`，
单次约 11 分钟（676s，60 个请求 @ 0.089 QPS）。

**归因用的对照组。** 因为修复后那轮跑之前清过缓存，必须排掉"提升来自空缓存"这个解释。
做法是只把 Bug 2 的 schema 改回 `list[int]`（镜像重新变空，行为等价于修复前），
**其余配置、清缓存动作、workload 全不变**，重启 router 后再跑一遍：

```bash
# 临时把 infera/router/kv_event/events.py 的 SglangBlockStored.token_ids 改回 list[int]
bash infera_1_server.sh          # router 重启只要 10 秒，引擎不用动
# 清缓存 + 同一条 agent-bench 命令，--name control-blindview
```

### 2.6 生产 profile：p50 60K / p90 300K

`code_agent_glm52_p50_60k_p90_300k.yaml` 才是贴近客户真实场景的那个，和 128K profile 有三处
本质差异：初始前缀是重尾对数正态（中位 58K、**均值 290K**、上限 380K）、
`new_session_rate` 从 0.04 跳到 **0.42**、`max_prompt_tokens` 380K。

**先确认服务端扛得住，再跑 bench。** 之前正确性套件覆盖的最长上下文只有 58,695 token，而这个
profile 的尾部要到 380K，直接开跑很可能只收集到一堆错误。用 [`pdops/smoke_longctx.py`](pdops/smoke_longctx.py)
逐级试探（每条都加盐，不吃缓存）：

```bash
$ python3 pdops/smoke_longctx.py 60000 150000 180000
target=  60000  prompt_tokens= 125222  wall=  45.0s  finish=stop  reply='OK'
target= 150000  prompt_tokens= 314039  wall= 136.6s  finish=stop  reply='OK'
target= 180000  prompt_tokens= 374033  wall= 170.5s  finish=stop  reply='OK'
```

三条全部正确返回，说明 380K 以内没有上下文墙——引擎的 `context_len` 是模型自带的
**1,048,576**，启动脚本没有设 `--context-length`；`/get_server_info` 显示
`token_capacity=1,858,688` 且 8 个 rank 条目**各自**都是这个数，所以 380K 的前缀只占单 rank
KV 池的 20%，容量不是瓶颈。单流 prefill 吞吐约 2.2–2.8K tok/s。

据此定并发：均值 107K 的 prompt，按聚合吞吐推算可持续 QPS 只有 0.03 量级，所以把 profile 自带的
`max_qps: 0.5` / `max_inflight: 32` 压下来，否则测的是排队。tokenizer 在 yaml 里是占位符
`/path/to/GLM-5.2-MXFP4`，必须覆盖：

```bash
curl -s -X POST http://127.0.0.1:30001/flush_cache
curl -s -X POST http://10.32.17.209:31001/flush_cache

agent-bench agent --server http://127.0.0.1:8000 \
  --model /wekafs/models/GLM-5.2-FP8 --tokenizer /wekafs/models/GLM-5.2-FP8 \
  --workload-config agent/workloads/code_agent_glm52_p50_60k_p90_300k.yaml \
  --max-qps 0.04 --initial-qps 0.02 --max-inflight 8 --gpus 16 \
  --dashboard-mode --name prod-p50_60k_p90_300k
```

**一个硬约束要提前知道：** bench 客户端对每条请求写死了 240s 超时，而 380K 的 prompt 装不进
这个预算——这一轮 23 条里正是有 1 条这样超时（`success_rate=0.957`，引擎侧无任何错误）。
要把 300K 以上的尾部跑干净，先按 §1.4 把它改大。tokenizer 与 `gpus` 的覆盖理由见 §1.6。

---

## 3. 现在的结果

三组各 60 请求、`success_rate=1.0`，同一 workload 种子（三组的 `total_tokens=3,169,196`、
`prefix=3,002,633`、`ideal_hit=94.7%` 完全一致，所以可直接对比）：

| 指标 | ① 基线：镜像空 + 未清缓存 | ② 对照：镜像空 + 已清缓存 | ③ 修复后：镜像正常 + 已清缓存 |
|---|---|---|---|
| 实际缓存命中率 | 47.2% | 54.3% | **83.6%** |
| 缓存效率（实际/理想） | 49.9% | 57.3% | **88.2%** |
| eviction rate | 50.1% | 42.7% | **11.8%** |
| cached / uncached token | 1,497,216 / 1,671,980 | 1,720,448 / 1,448,748 | 2,648,192 / **521,004** |
| TTFT p50 | 16.9s | 14.6s | **2.5s** |
| TTFT p90 | 38.8s | 29.8s | 24.3s |
| TPOT p50 | 31.7ms | 31.9ms | 32.3ms |

拆开看：**清缓存值 7 个点**（47.2 → 54.3），**kv-aware 本身值 29 个点**（54.3 → 83.6），
uncached token 少掉 64%，TTFT p50 快 5.8 倍。TPOT 三组都在 32ms，说明这条路径没有动到 MTP。

**机制不是"分散得更均匀"，恰恰相反，是"别再乱分散"。** 看 router 的 pick 日志：

| | ② 对照（镜像空） | ③ 修复后 |
|---|---|---|
| 60 个请求的 rank 分布 | dp0-dp4 = 19/15/13/7/6 | **全部 dp0** |
| 有命中的请求数 | 0 | 59 / 60 |

> 这两行是当时从 `infera_1_server.log` 的 pick 日志统计的，而 **router 一重启该日志就被覆盖**
> （`pdops/results/` 里只有 bench 自己的产物，它不记录请求落在哪个 rank）。以后跑对照实验，
> 重启 router 前先把日志留一份：
> `grep -a "pick policy=kv-aware" infera_1_server.log | gzip > picks_<tag>.log.gz`。

打分函数是 `cost = w_overlap × (request_blocks − hits) + active_blocks`。镜像为空时命中项
恒为 0，只剩负载项，于是并发的几个请求被摊到不同 rank，会话下一轮就落到不持有其前缀的
rank 上、只能重算——bench 把这部分记成 "eviction"，其实是**落错了 rank**。镜像正常之后，
命中项（prefill 权重 20）把每个会话稳稳钉回持有它前缀的那个 rank，重算就消失了。

> **`eviction_rate` 不是实测淘汰量。** 两轮数据都满足 `eviction_rate == 1 − efficiency`
> （88.2%/11.8%、35.3%/64.7%），它就是效率的补数，不代表引擎真的淘汰了多少 KV。下面 §3.1
> 说明为什么这个补数会把"物理上不可能命中"也算成损失。

### 3.1 生产 profile（p50 60K / p90 300K）：跑通了，但表头数字会骗人

23 条请求、22 条完成、**引擎侧零错误**（唯一那条失败是 §2.6 说的客户端 240s 超时），
末态前缀 mean 147K / max 380K，实际新会话率 43.5%（目标 42%）：

| 指标 | 128K profile（③ 修复后） | 生产 profile |
|---|---|---|
| 请求数 / 完成 | 60 / 60 | 23 / 22（1 条客户端超时） |
| prompt 长度 p50 / p90 | 50K / 90K | **91K / 380K** |
| `new_session_rate` | 0.04 | **0.42**（实际 0.435） |
| 表头缓存效率 | 88.2% | **35.3%** |
| TTFT p50 / p90 | 2.5s / 24.3s | 53.6s / 176.3s |
| TPOT p50 | 32.3ms | 33.5ms |

表头效率从 88.2% 掉到 35.3%，看着像 kv-aware 在真实场景失效了。**逐请求拆开看，结论完全相反：**

| | 128K profile | 生产 profile |
|---|---|---|
| 几乎全 miss 的请求（actual < 0.05） | 1 / 59 | 9 / 21 |
| 这批请求占全部 token | **1.1%** | **56.4%** |
| 其余请求的 token 加权效率 | 91.7% | **95.0%** |

那 9 条全 miss 的请求正是**新会话的首轮**——本轮 16 个会话里有 10 个是运行中新建的
（`new_session_times` 10 条、`existing_session_requests` 13 条，合计 23 = 发出总数）。首轮的前缀
从未发过，任何缓存都不可能命中；而 bench 的 `ideal_hit_rate` 按会话模型把这段初始前缀算作
"可复用"，于是把物理上不可达的部分也计入了分母。这个 profile 的前缀又特别大，光这 9 条就占了
56.4% 的 token，所以表头效率被压到 35%。

**真正衡量路由质量的是"能命中的那批命中得多好"，这个数字是 95.0%——比 128K profile 的
91.7% 还高**，因为这里可命中的请求都是十万级前缀，一旦路由正确就几乎全量命中
（逐请求实测有 0.990/0.990、0.984/0.984、0.915/0.915 这样的成绩）。结论是 kv-aware 在客户
真实场景下工作正常，`efficiency` 这个指标在高 `new_session_rate` 下不能直接当路由质量看。

---

## 4. 剩余问题：新会话全部从 dp0 开始

TTFT p90 仍有 24.3s（生产 profile 更是 176s），来源是一个**策略问题，不是 bug**：

```python
picked = min(targets, key=lambda t: (cost(t), active(t)))
```

新会话在 8 个 rank 上命中都是 0、`active` 也基本是 0（pick 日志里恒为 `active_blocks=0`），
于是完全平票，`min` 返回列表里第一个目标——**永远是 dp0**。128K 那轮 60 条请求全部落在 dp0。

这不是 KV 容量问题：`/get_server_info` 显示 `token_capacity=1,858,688` 且 8 个 rank 条目**各自**
都是这个数（不是共享池的 1/8），8 个会话末态共约 568K token，单 rank 装得下——所以剩余的
效率缺口来自首轮 miss 和零星部分命中（见 §3.1），不是淘汰。真正的代价是**并行度**：所有会话
都钉在一个 DP rank 上，prefill 在该 rank 的调度器里排队，其余 7 个 rank 的算力和 KV 池空转，
长 prompt 一堆积就直接体现在 p90 上。

要再往上走：平票时按负载/容量把**新**会话铺开，同时保持老会话对持有其前缀那个 rank 的
粘性。生产 profile 下这件事更值钱——它 42% 是新会话，且单条 prompt 动辄十万级，铺不开就等于
把 8 倍的 prefill 算力浪费在一个 rank 上。

改 `infera/router/policy/kv_event_aware.py` 的 `cost`/tie-break 即可，**不需要重启引擎**
（router 重启 10 秒），所以迭代很快。

另外几个副产品值得提上游/内部：

- Optimus-AgenticBench 的 `efficiency` / `eviction_rate` 在高 `new_session_rate` 下会严重
  低估路由质量（把新会话首轮那段从未发送过的前缀也算进可命中分母），建议补一个只统计
  重复轮的口径；`aiohttp.ClientTimeout(total=240)` 也该随 profile 的 prompt 上限放大。

- `free_tcp_port_block` 只用 loopback 探测，原理上发现不了 IPVS 拦截这类"能 bind、
  从节点 IP 打不通"的端口——任何在 K8s 上给 peer 广播端口的组件都有同样的风险。
- SGLang 的 `BlockStored.token_ids` 声明是 `list[int]`，而 EAGLE 路径实际写入 bigram 对，
  声明与实际不符（msgspec 编码不校验，所以一直没暴露）。

---

## 5. 快速排查表

| 现象 | 先看什么 | 大概率原因 |
|---|---|---|
| `cache-view` 全 0，两侧都没报错 | 事件端口基址是否在 30000-32767；`/proc/net/ip_vs` 有没有同名端口 | Bug 1：IPVS 截包 |
| `cache-view` 全 0，router 有 `kv decode failed` | 报错里的字段路径 | Bug 2：wire 格式不匹配（`$[1][0][3]` = `token_ids`） |
| `cache-view` 全 0，用的是 Rust router | 直接怀疑 bigram | Bug 3：静默丢弃，无日志 |
| 镜像非空但 `cache_hits` 恒 0 | 请求侧 `request_blocks` 是否合理 | 哈希两侧不对齐（block_size / tokenizer / bigram 折平方式） |
| 命中正常但 TTFT p90 仍高 | pick 日志的 rank 分布 | §4：新会话全落一个 rank，7/8 算力空转 |
| `efficiency` 很低但逐请求有大量满命中 | profile 的 `new_session_rate` | §3.1：首轮不可命中被算进分母，不是路由问题 |
| 长 prompt 报客户端超时、引擎侧无错 | bench 的 `ClientTimeout(total=240)` | §2.6：240s 装不下 300K+ 的 prefill |
| 独立 SUB 收不到但引擎日志有 `sent` | 换 loopback 再抓一次 | 地址问题，不是发布器问题（§1.0/§1.1） |

定位这几种情况用的诊断脚本都在 [`pdops/`](pdops/)（逐个用途见
[`pdops/README.md`](pdops/README.md)）：`zmq_kv_probe.py`（独立订阅者，分开"引擎没发"和
"router 没应用"）、`zmq_dual_probe.py`（同进程同时订阅 loopback 和节点 IP，消除时序歧义，
**Bug 1 就是它找出来的**）、`zmq_host_matrix.py`（地址 × payload 尺寸矩阵，不涉引擎）、
`zmq_pub_ordering.py`（复刻"发布器先起、订阅者后连"的时序，**证伪了线程安全假设**）。
日常检查用 `verify_kv_aware.py` 和 `probe_kv_aware.sh`。

本文引用的三组 bench 与正确性套件的原始结果都在 [`pdops/results/`](pdops/results/)。
