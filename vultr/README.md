# vultr — PD 成功试验记录(legacy 集群 `chi28xx`)

本文件夹汇总所有在 **vultr / legacy 集群**(hostname `chi27xx`/`chi28xx`,
data-plane IP `10.2.122.x`)上跑通的 PD(prefill/decode 分离)试验交付。
spur 集群(`crsuse2-m2m`)的记录见 `../crsuse/`。

子目录命名规则:`{单/多node}_{TP}_{NIC}_{注册路径}_{model}_{日期}`。

| 目录 | node | TP | NIC | KV 注册路径 | model | 结论 |
|------|------|----|-----|------------|-------|------|
| `multinode_tp8_ionic_ibvregmr_mooncake_dsv4_20260723/` | 多 | 8 | 8×ionic | `ibv_reg_mr`(裸,配 host-libionic 注入) | DSv4-Pro | ✅ 跨节点 PD(mooncake + mori)。chi2865/chi2879 |
| `multinode_tp8_ionic_mori_gptoss120b/` | 多(2) | 8 | 8×ionic | MoRI-IO(AINIC RDMA) | gpt-oss-120b | ✅ 1P1D over MoRI + kvd L3(可复用 harness,节点参数化) |

## 关键背景(为什么 vultr 用 ionic 裸 `ibv_reg_mr`)

legacy 集群的数据面是 8×ionic(一 GPU 一张,400G),PD 走 8 卡 ionic。此路径用裸
`ibv_reg_mr` + 把 host 的 libionic provider 注入容器(解决 ABI 不匹配),
**不是** dmabuf 路径——与 spur 上强制 mlx5+dmabuf 的方案不同,两者不可互换。

> ⚠️ `multinode_tp8_ionic_ibvregmr_mooncake_dsv4_20260723` 日期与 spur 的
> `singlenode_...20260723` 同天,但**不是同一集群**:它的环境是 chi2865/chi2879
> (`10.2.122.x`),属 vultr。其 ionic/mori 的成功**不适用于 spur**(spur ionic 无 ODP)。
