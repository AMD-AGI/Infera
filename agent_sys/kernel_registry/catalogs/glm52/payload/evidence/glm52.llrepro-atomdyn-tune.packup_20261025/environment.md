> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-24 07:58Z .. 2026-10-25 03:00Z)

Same hardware, image, kit, KF overlay and model as `registry/glm52.llrepro-kf-ar-topk.packup_20261024/environment.md`:
compute-node-03 (prefill PP4 x TP1, router, AgentX client) and compute-node-04 (decode TP4/DP4 DPA), 8 x MI355X each,
GPU4-7 used; ionic_0..7 registered, traffic on ionic_4..7 (400 Gb/s per port); image `historical-sglang-image-b`
(sglang 0.5.22.dev20261018+g8dee3f6bdb, image sglang tree `e9b22a0818`); kit Infera 06796e13 + contributor kit-v6-pp4 + registry
decode hooks; KF overlay `REDACTED_LOCAL_PATH`.

Run windows (UTC): llr7r 07:58-09:26 (10-24); llr9c 10:56-12:27; llr10 12:3x-14:19; llr11 14:2x-15:49; llr12 15:5x-17:2x;
llr13 17:3x-18:54; llr14 19:0x-20:25; llr15 20:3x-22:19; llr16 22:3x-23:51 (stopped); llr14r 23:5x-01:23 (10-25);
llr17 01:3x-02:39 (void); llr17b from 02:5x.

[Private access details omitted.]
