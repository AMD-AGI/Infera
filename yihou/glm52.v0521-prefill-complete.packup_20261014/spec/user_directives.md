# User directives relevant to this packup (verbatim, from the session transcript; UTC timestamps)

The session mission file (`spec/mission.md`, copied) records updates only up to 2026-10-13 05:15; the directives below were given
later in the same session and are quoted verbatim from the conversation log
(`~/.claude/projects/-home-yihou-dev-git-16-19-infera-glm52-pd/739a4843-c97e-4a41-b91f-66eb2904a1db.jsonl`, local machine).

## 2026-10-13T07:52:41Z — the experiment
> 0.5.21有个config，可以让decode侧在prefill计算结束后，再分配显存,你调研下如何正确使用，找一台新的机器,oai或者135-139里空暇或者limou在用的，hi-cache=370, decode gmu=0.96 conc=48测一下这个

Reading: research `--disaggregation-decode-allocation-policy prefill_complete`; run it on a fresh node (oai, or a free/limou 135-139
node) with prefill HiCache 370 GB, decode mem-fraction 0.96, CONC 48.

## 2026-10-13T09:32:43Z — node choice
Assistant had proposed: let x8p A2 finish on 274, cancel its planned A3 (C52), and give 274 to pc48. User:
> 好的

## 2026-10-13T15:09:33Z — control round
Assistant had asked whether to run (1) the pc48b control (allocation policy back to `early`, everything else as pc48b) and
(2) the x8p split. User:
> 跑

## 2026-10-14T01:52:22Z — this pack-up
> 非常好，感谢你效率如此之高，做这个多的实验，你把尽量多的实验分门别类packup（使用packup技能）一下，放在worktree的yihou文件夹底下。然后同步按照你的想法测0.90和0.92 prefill kv

(The 0.90 / 0.92 prefill-KV runs are a separate experiment, not part of this kit.)

## Standing rules from the mission (2026-10-12/13) that shaped this experiment
- Never delete a file whose path lacks `yihou`; never disturb other users' containers/GPUs; `/home/yihou/dev/git/infera.glm52.pd`
  is read-only except our own `yihou/` packups.
- Warm-stack protocol: A1 fresh (unscored) → flush decode then prefill → A2 scored; AgentX full 3600 s.
- Measure, don't assume; suspend, don't conclude. Correctness is waived (`SGLANG_SIMULATE_ACC_LEN=3.61`).
