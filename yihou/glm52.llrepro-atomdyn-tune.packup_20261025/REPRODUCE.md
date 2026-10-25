# Reproduce

Prerequisites: those of `yihou/glm52.llrepro-pp4-dpa4.packup_20261024` (image `infera-sglang:v0522-fdbafb60` on both
nodes, model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, GPU4-7 free on crsuse2-m2m-135 / -136, ssh 135 -> 136)
and the kit + KF overlay of `yihou/glm52.llrepro-kf-ar-topk.packup_20261024` (kit/, kf_overlay/ at
`/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay`).

```bash
D=/shared_nfs/yihou_glm52/llrepro_20261023
cp scripts/* $D/                                  # site files + run/guard/analysis scripts
# ON 135, one round at a time (~95 min each):
r=llr14
bash $D/run_llr.yihou.sh $D/$r.site.env $r-c80 > $D/run_llr.$r.yihou.log 2>&1 &
bash $D/guard_llr.yihou.sh $r > $D/guard.$r.yihou.log 2>&1 &
# after RUN_LLR_DONE in the run log:
python3 -I $D/llsum.yihou.py  $D/work/runs/$r-c80          # summary row
python3 -I $D/bneck.yihou.py  $D/work/runs/$r-c80          # steady-state per-rank queue stats
python3 -I $D/qseries.yihou.py $D/work/runs/$r-c80 <profiling start HH:MM:SS> <+60 min> 10
python3 -I /shared_nfs/yihou_glm52/v0521cap2_20261013/hist.yihou.py $r <server_metrics_export.csv>
```

Profile capture as in llr7r: start `prof_llr7r.yihou.sh` on 135 next to run_llr (it waits for the profiling phase,
sleeps 25 min, probes, py-spy, `POST /start_profile` decode 40 steps then prefill 20 steps; traces copied out after the
run). Expect the prefill trace save to stall each PP stage ~45 s (score perturbed).

Checks after start: decode server_args (`cuda_graph_bs_decode`, `mem_fraction_static`, `stream_interval`), container env
(`docker exec yihou-llr-decode env | grep -E 'TOPK_V2|CLIP'`), prefill `pp_async_batch_depth`, router startup line
(`decode_input_demand: On`, `session_affinity`).
