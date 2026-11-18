export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
# sglang #47897 (gfx950 gluon paged MQA logits) in vLLM, aggregated A/B on one node after the P4D4 runs:
#   vpd-aggK  GPUs 0-3, overlay + VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1      vpd-aggR  GPUs 4-7, stock v0.34.0
# Same serving config as the P4D4 decode (DP4 TP1, fp8 KV, MTP3, block 64). Correctness gate on both, then decode
# ITL with vllm bench serve (random 64K-token prompts, 1024 output tokens, ignore EOS) at concurrency 64 and 104.
while [ ! -e queue/1118-000826-Lp4d4s6c7.rc ]; do sleep 20; ls queue > /dev/null; done
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx vpd-aggK vpd-aggR > /dev/null 2>&1; sleep 20
export IMAGE=infera-vllm:v0340-0c5e5e89
out=$R/results/kf47897-ab; mkdir -p $out
V=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops
O=$R/overlay/vllm/v1/attention/ops
a=(--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
   --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192
   --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice)
( ENV_EXTRA="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1" \
  MOUNT_EXTRA="$O/rocm_aiter_mla_sparse.py:$V/rocm_aiter_mla_sparse.py:ro $O/rocm_gluon_paged_mqa_logits.py:$V/rocm_gluon_paged_mqa_logits.py:ro" \
  bash scripts/vllm_up.sh vpd-aggK 0,1,2,3 8100 "${a[@]}" --data-parallel-rpc-port 29610 ) &
sleep 30
( bash scripts/vllm_up.sh vpd-aggR 4,5,6,7 8200 "${a[@]}" --data-parallel-rpc-port 29710 ) &
wait
curl -sf -m 5 http://127.0.0.1:8100/v1/models > /dev/null && curl -sf -m 5 http://127.0.0.1:8200/v1/models > /dev/null || {
  echo "$(date -u +%m%d-%T) kf47897-ab: deployment failed" | tee -a logs/summary.out
  for c in vpd-aggK vpd-aggR; do docker logs --tail 300 $c > $out/$c-fail.log 2>&1; done
  docker rm -f vpd-aggK vpd-aggR > /dev/null 2>&1; exit 0; }
PREFILL_NUM_WORKERS=0 bash scripts/gate.sh kf47897-on http://127.0.0.1:8100 "http://127.0.0.1:8100/metrics" > $out/gate-on.out 2>&1 &
PREFILL_NUM_WORKERS=0 bash scripts/gate.sh kf47897-off http://127.0.0.1:8200 "http://127.0.0.1:8200/metrics" > $out/gate-off.out 2>&1 &
wait
image_id=$(docker image inspect -f '{{.Id}}' infera-vllm:v0340-0c5e5e89)
for conc in 64 104; do
  for side in on:8100 off:8200; do
    name=${side%%:*} port=${side##*:}
    docker run --rm --network host -v $out:/out -v /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4:/model:ro \
      --entrypoint vllm infera-vllm:v0340-0c5e5e89 bench serve --backend vllm --base-url http://127.0.0.1:$port \
      --endpoint /v1/completions --model glm5.2-mxfp4 --tokenizer /model --trust-remote-code --dataset-name random \
      --random-input-len 65536 --random-output-len 1024 --num-prompts $((conc * 2)) --max-concurrency $conc --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /out --result-filename bench-$name-c$conc.json \
      > $out/bench-$name-c$conc.log 2>&1 &
  done
  wait
done
for c in vpd-aggK vpd-aggR; do docker logs $c 2>&1 | gzip > $out/$c.log.gz; done
zcat $out/vpd-aggK.log.gz | grep -c "gluon paged-MQA logits" > $out/gluon-used-count.txt
docker rm -f vpd-aggK vpd-aggR > /dev/null 2>&1
python3 - "$out" <<'EOF' | tee -a logs/summary.out
import json, sys, glob, os
out = sys.argv[1]
row = []
for f in sorted(glob.glob(f"{out}/bench-*.json")):
    d = json.load(open(f))
    row.append(f"{os.path.basename(f)[6:-5]}: tpot p50 {d.get('median_tpot_ms', 0):.1f} ms, itl p50 {d.get('median_itl_ms', 0):.1f} ms, "
               f"out tok/s {d.get('output_throughput', 0):.0f}")
print("kf47897-ab " + "; ".join(row))
EOF
exit 0
