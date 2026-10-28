> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Reproduce

Prerequisites: those of `registry/glm52.llrepro-pp4-dpa4.packup_20261024` (image `historical-sglang-image-b` on both
nodes, model `REDACTED_LOCAL_PATH`, GPU4-7 free on compute-node-03 / -REDACTED_NODE, ssh REDACTED_NODE -> REDACTED_NODE)
and the kit + KF overlay of `registry/glm52.llrepro-kf-ar-topk.packup_20261024` (kit/, kf_overlay/ at
`REDACTED_LOCAL_PATH`).

```bash
D=REDACTED_LOCAL_PATH
cp scripts/* $D/                                  # site files + run/guard/analysis scripts
# ON REDACTED_NODE, one round at a time (~95 min each):
r=llr14
bash $D/run_llr.registry.sh $D/$r.site.env $r-c80 > $D/run_llr.$r.registry.log 2>&1 &
bash $D/guard_llr.registry.sh $r > $D/guard.$r.registry.log 2>&1 &
# after RUN_LLR_DONE in the run log:
python3 -I $D/llsum.registry.py  $D/work/runs/$r-c80          # summary row
python3 -I $D/bneck.registry.py  $D/work/runs/$r-c80          # steady-state per-rank queue stats
python3 -I $D/qseries.registry.py $D/work/runs/$r-c80 <profiling start HH:MM:SS> <+60 min> 10
python3 -I REDACTED_LOCAL_PATH $r <server_metrics_export.csv>
```

Profile capture as in llr7r: start `prof_llr7r.registry.sh` on REDACTED_NODE next to run_llr (it waits for the profiling phase,
sleeps 25 min, probes, py-spy, `POST /start_profile` decode 40 steps then prefill 20 steps; traces copied out after the
run). Expect the prefill trace save to stall each PP stage ~45 s (score perturbed).

Checks after start: decode server_args (`cuda_graph_bs_decode`, `mem_fraction_static`, `stream_interval`), container env
(`docker exec historical-e2e-container-decode env | grep -E 'TOPK_V2|CLIP'`), prefill `pp_async_batch_depth`, router startup line
(`decode_input_demand: On`, `session_affinity`).
