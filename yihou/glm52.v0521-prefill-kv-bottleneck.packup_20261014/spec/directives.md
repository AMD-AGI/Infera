# User directives behind this packup (2026-10-13, UTC)

Source of record: `spec/mission.md` (copy of
`/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/mission.md`, section
"UPDATE (user, 2026-10-13 ~05:15 UTC …)") and the ledger `analysis/ledger/working_process.md`.

## ~05:15 — verbatim (mission.md)

> 并行展开一下实验：1. 大hi-cache 大conc的我我不理解为啥没有提升，因为之之前强行8 expert开大kv cache是可以到26k, 你去并行/串行做两个实验a.单独让decode使用8 expert腾出更多的kv-cache. b. 让prefill单独开expert, 获得更大的kv-cache观察下那边有提升，看看谁是瓶颈(conc就按昨天实验最高的我记得是52？8 expert怎么玩你可以去~/dev/git/infera.glm52.pd/yihou底下的packup找)。2. 关于decode限制transfer和prealloc cache那个事情。实验验证一下直接用最初conc=40那一版的hi-cache容量压conc=56/64, 期望是比~/dev/git/infera.glm52.pd/yihou之前sweep的性能回退少一点。

Reading recorded in mission.md: (a) decode leg only on 8 experts, (b) prefill leg only on 8 experts; whichever gains
shows the bottleneck side; CONC = C48 (best score) with C52 as A3 on the same stack. Part 2 (runshare at C56/C64,
rounds tb56/tb56n) is NOT in this packup.

## ~15:00 — x8pc (ledger poll 15:15 and `scripts/harness/env.x8pc-206.yihou.sh` header)

User reply "跑" to the proposed split of x8p: prefill-only synthetic 8 experts with `--max-total-tokens 2433088`
(= the 256-expert prefill pool), isolating expert8 compute/weights from prefill KV capacity. [Only the one-word reply is
verbatim; the design text is ours.]

## ~15:55-16:05 — x128p (ledger "two inserted experiments" and `scripts/harness/config.yihou.expertnleg.sh` header)

> prefill侧expert砍半

Recorded as: prefill experts halved, 256 → 128 (truncated checkpoint, own top-8 routing), decode unchanged; compare with
x8p and q1. The same message also asked for hc50 (HiCache 50 GB at C40) — not in this packup.
User 16:0x: use oai nodes (up to 4); 135-137 only if not llying's.

## Standing rules from mission.md (apply to reproduction)

English for work, Chinese for reports; never edit a running bash script in place; drivers detached; VRAM drain after a
HiCache-450 stack stops 28-35 min — gate on idle; ask before using other users' nodes or touching their containers.
