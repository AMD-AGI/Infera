# crusoe v3-200 colocated curves (2026-11-21/22) — R1 vs E4 (connector-forced kernel path)

All official 3600 s AgentX points, DP2 on 2 GPUs, contract_ok=True, crusoe v3-200 (job 302960), 4 points concurrent on the node
(GPU0,1 / 2,3 / 4,5 / 6,7). Ledger: /shared_nfs/yihou/dsv41_vllm_pd_20261119/results/ledger.csv. tok/s/GPU.

| conc | R1 (r1-dp2) | e4c (connector, sparse fast path kept) | e4b (connector, sparse off) |
|---|---|---|---|
| 128 | 267,106 (n=2: 267,644 / 266,568) | 245,266 (n=2: 245,693 / 244,839) -8.2% | 226,397 (n=2: 225,603 / 227,190) -15.2% |
| 192 | 334,260 (n=2: 335,588 / 332,932); v3-014 327,639 / 306,857 | 304,122 (n=2: 302,528 / 305,715) -9.0%; v3-014 305,002 | 269,950 (n=2: 269,148 / 270,751) -19.2%; v3-014 273,460 |
| 224 | 357,987 (n=2: 359,383 / 356,590); v3-014 354,795 / 358,588 | 310,745 (n=2 v3-014: 312,962 / 308,527) -13.2%; rep3 310,101 | 266,635 (n=2 v3-014: 260,850 / 272,419) -25.5%; rep3 258,140; rep4 264,250 |
| 256 | 364,694 (n=4: 372,545 / 363,732 / 360,773 / 361,726 v3-014); v3-014 rep5 349,146; v3-014/304125 GPU4,5 348,490; 305057 348,535 / 361,419 | 313,936 (n=2 v3-200: 313,169 / 314,703) -14.1%; v3-014 n=2: 274,121 / 273,894; v3-014/304125 307,861; 305057 308,084 / 314,775 | 273,330 (n=2: 273,150 / 273,509) -25.3%; v3-014 n=4: 264,714 / 273,434 / 275,769 / 275,010 |
| 288 | 355,584 (n=2: 355,705 / 355,463); v3-014 336,189 / 345,618; 305057 356,394 | 289,622 (n=2 v3-014: 298,221 / 281,023) -18.5%; v3-014 rep3 303,331; rep4 304,974; 305057 262,020; 305057 rep 294,664 | 251,675 (n=1) -29.2%; v3-014 272,316 / 245,934 / 262,732; 305057 274,198 |
| 320 | 321,420 (n=1; first try 340,558 void: R2 stall); v3-014 349,596 / 342,335 / 335,773 | 271,730 (n=2: 261,381 v3-200 / 282,079 v3-014) -15.5%; v3-014 139,858 ANOMALY (half; see note); v3-014 GPU0,1 repeat 290,965; 305057 253,484 | 230,882 (n=2: 242,279 v3-200 / 219,485 v3-014) -28.2%; v3-014 rep2 233,730; 305057 248,621 |
| 384 | 269,859 (n=2: 290,889 v3-200 / 248,828 v3-014; 14.5% apart); v3-014 rep2 268,263; 305057 285,485 | 162,633 / 164,277 (n=2 v3-014) vs R1 v3-014 268,263 | 157,397 (n=1 v3-014) -41.3% |

Percentages vs the R1 mean at the same conc. Observed: R1 peak at c256 (365,683, n=3, range 3.2%); the E4 gap widens with conc
for both variants. Not measured here: R2 (UMBP) on crusoe (2 node-wide stalls, research/crusoe_r2_umbp_stall.md); why the gap grows.
Replicate spread so far <= 3.2% (R1 c256), most pairs < 1%.

### Anomaly 2026-11-22: e4c-cru8-c320-11221401 = 139,858 (job 304125, v3-014, GPU4,5)
- About half the other e4c c320 runs (261,381 / 282,079). contract_ok True, rc=0, GPU4,5 read 100% at every watchdog sample
  (14:13-15:36Z), no "sample_tokens timed out"; "No available shared memory broadcast" only at graph capture (14:07-14:08) and
  after 15:38 (end of window).
- Client tput_in_srv (cumulative) 690-820k/s through 15:12, then falling to 555k/s by 15:42; waiting queue grew 10 -> 250.
  For comparison the e4b c256 run on 303579 GPU0,1 ended at 1,039k/s for 264,714.
- Cause unknown. Not measured: per-GPU clocks/power during the run. Next point on the same pair (R1 c256, reference ~360k) is the
  differential check; if it is also low, the pair (GPU4,5, PCIe domain 0003) is suspect.
- Differential result (17:16Z): R1 c256 on the same pair GPU4,5 (e3-r1cru8-c256-11221545) = 348,490, matching R1 c256 v3-014
  rep5 349,146 on other GPUs. So the pair performed normally for R1 right after the anomaly. The anomaly is not explained by a
  persistently slow GPU4,5 pair; cause still unknown. e4c c320 repeat on GPU0,1 (e4c-cru9-c320) is running.
- Repeat result (18:48Z): e4c c320 on GPU0,1 (e4c-cru9-c320-11221713) = 290,965, in line with 261,381 / 282,079. So the
  139,858 run is not reproduced by the same config on the same node (other pair). The same pair GPU4,5 then stalled at 17:44
  (e4b c224, shm_broadcast TimeoutError; see crusoe_r2_umbp_stall.md). Treat 139,858 as a void outlier tied to a GPU4,5 episode;
  cause unknown.
