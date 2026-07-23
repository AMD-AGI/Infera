# crsuse — PD 成功试验记录(AMD spur 集群 `crsuse2-m2m`)

本文件夹汇总所有在 **spur 集群**(`crsuse2-m2m` / `amd-spur`,8×MI355X gfx950,
data-plane IP `10.245.x`)上跑通的 PD(prefill/decode 分离)试验交付。
另一集群(vultr / `chi28xx`)的记录见 `../vultr/`。

子目录命名规则:`{单/多node}_{TP}_{NIC}_{注册路径}_{model}_{方式}_{日期}`。

| 目录 | node | TP | NIC | KV 注册路径 | model | 结论 |
|------|------|----|-----|------------|-------|------|
| `multinode_tp8_mlx5_dmabuf_dsv4_manualcommit_20260724/` | 多(2) | 8 | 单张 mlx5 | `ibv_reg_dmabuf_mr`(ODP dynamic attach,不 pin/不翻倍) | DSv4-Pro | ✅ 跨节点 PD,手动 `docker commit` 镜像 |
| `multinode_tp8_mlx5_dmabuf_dsv4_dockerfile_20260727/` | 多(2) | 8 | 单张 mlx5 | `ibv_reg_dmabuf_mr` | DSv4-Pro | ✅ 跨节点 PD,正式 `Dockerfile.sglang.dmabuf` 可复现构建 + 单机 TP4 三传输方式对比 |
| `singlenode_tp4_noNIC_loopbacktcp_dsv4_20260723/` | 单 | 4+4 | 无(loopback TCP) | 无(`mooncake_tcp`,零 RDMA) | DSv4-Pro | ✅ 节点内 1P1D;⏸️ 跨节点卡 mooncake RDMA `ENOMEM` |

## 关键背景(为什么 spur 上强制 mlx5)

spur 节点有 8×AMD Pensando **ionic**(400G RoCE,**无 ODP**)+ 1×Mellanox **mlx5**
(200G,**有 ODP**)。`ibv_reg_dmabuf_mr` 在无 ODP 的 ionic 上会 **pin 整个 KV 池 → 翻倍 →
KFD 耗尽崩溃**;只有在 mlx5 上做 dynamic attach 才能零 pin。用户明确"带宽回退可接受、
KV 翻倍不可接受",故所有 dmabuf 跨节点 PD 都强制走单张 mlx5。

> 相关:`../spur_repro/` 是同集群更早的**单机 R4 serving**(非 PD)交付,
> `singlenode_tp4_...` 原是它的子实验,现移入本文件夹统一归类。
