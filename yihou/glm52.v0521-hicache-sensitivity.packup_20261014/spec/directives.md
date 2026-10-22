# Directives relevant to this packup

## The hc50 directive (user, 2026-10-13 ~15:55 UTC)

The Chinese original was not written to `mission.md`; the session ledger recorded it in English
(`analysis/ledger/working_process.md`, section "2026-10-13 ~16:05 two inserted experiments (user ~15:55)"):

> hc50 (user: "HiCache 50 GB at C40 on the best v0.5.21 config (e2g), everything else unchanged" — HiCache sensitivity)

Wording as relayed to the pack-up task: "run hi-cache 50 GB at conc 40 on the best v0.5.21 config (e2g), everything
else unchanged, to see how HiCache affects performance."

Same message, second experiment (not in this packup): "prefill experts halved, 256 -> 128" -> round x128p.
User 16:0x: "use oai nodes (up to 4), 135-137 only if not llying's" (136 was then free; see ledger).

## Session directive 2026-10-13 ~05:15 UTC (verbatim, `spec/mission.md` "UPDATE (user, 2026-10-13 ~05:15 UTC ...)")

> 可以，那你申请2-3台oai机器开始吧，如果机器不够的话，可以在135-139之间找一个空闲的来凑（limou占了一个机器一直在干啥，好几天了，没有真实利用率给他杀了）

> 并行展开一下实验：1. 大hi-cache 大conc的我我不理解为啥没有提升，因为之之前强行8 expert开大kv cache是可以到26k, 你去并行/串行做两个实验a.单独让decode使用8 expert腾出更多的kv-cache. b. 让prefill单独开expert, 获得更大的kv-cache观察下那边有提升，看看谁是瓶颈(conc就按昨天实验最高的我记得是52？8 expert怎么玩你可以去~/dev/git/infera.glm52.pd/yihou底下的packup找)。2. 关于decode限制transfer和prealloc cache那个事情。实验验证一下直接用最初conc=40那一版的hi-cache容量压conc=56/64, 期望是比~/dev/git/infera.glm52.pd/yihou之前sweep的性能回退少一点。

(Context for the HiCache question: "大hi-cache 大conc ... 没有提升" — large HiCache at large CONC did not help;
hc50 probes the other direction, a much smaller HiCache at the C40 best config.)

## 2026-10-14 (user, relayed with this pack-up request)

> 非常好，感谢你效率如此之高，做这个多的实验，你把尽量多的实验分门别类packup（使用packup技能）一下，放在worktree的yihou文件夹底下。然后同步按照你的想法测0.90和0.92 prefill kv

The second sentence is the pf88/pf90/pf92 rounds (prefill mem 0.88/0.90/0.92) — in flight, not in this packup.

## Standing rules (mission.md)

Same-node 1P1D P4D4, AgentX full 3600 s, warmup 10/lane; A1 fresh (unscored) -> flush decode then prefill -> A2
scored; correctness waived (`SGLANG_SIMULATE_ACC_LEN=3.61`); never delete outside yihou dirs; English for work.
