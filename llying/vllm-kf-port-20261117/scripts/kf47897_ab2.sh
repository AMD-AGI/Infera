export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
# #47897 aggregated A/B, second attempt. The first never ran the gluon path: on ROCm vLLM flattens MTP decode to one
# query per row (indexer._use_flattening), so next_n is 1 and the overlay required next_n >= 2. Fixed overlay accepts
# flattened rows (LPT group = 1 + num_speculative_tokens). vpd-aggR (stock, GPUs 4-7) from the first attempt keeps
# running; vpd-aggK is restarted on GPUs 0-3 with the fixed overlay.
export IMAGE=infera-vllm:v0340-0c5e5e89
out=$R/results/kf47897-ab2; mkdir -p $out
# GPUs 0-3 are free: function-level test (incl. flattened form) and the microbenchmark with aiter's flattened call.
GPU=0 bash kf/run_overlay_test.sh > $out/overlay-test.txt 2>&1 &
( image_id=$(docker image inspect -f '{{.Id}}' $IMAGE)
  gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
  timeout 900 docker run --rm --name vpd-kf-logits2 --network host --ipc host --device /dev/kfd --device /dev/dri "${gids[@]}" \
    --security-opt seccomp=unconfined -v "$R/kf:/kf" -v "vpd-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" \
    -e HIP_VISIBLE_DEVICES=1 --entrypoint python3 $IMAGE /kf/bench_paged_logits.py > $out/paged_logits.txt 2>&1 ) &
wait
docker rm -f vpd-kf-ovl vpd-kf-logits2 > /dev/null 2>&1
V=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops
O=$R/overlay/vllm/v1/attention/ops
a=(--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
   --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192
   --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice)
ENV_EXTRA="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1" \
  MOUNT_EXTRA="$O/rocm_aiter_mla_sparse.py:$V/rocm_aiter_mla_sparse.py:ro $O/rocm_gluon_paged_mqa_logits.py:$V/rocm_gluon_paged_mqa_logits.py:ro" \
  bash scripts/vllm_up.sh vpd-aggK 0,1,2,3 8100 "${a[@]}" --data-parallel-rpc-port 29610 > $out/aggK-up.log 2>&1
grep -a -c "gluon paged-MQA logits" logs/engines/vpd-aggK.log > $out/gluon-used-after-start.txt
if curl -sf -m 5 http://127.0.0.1:8100/v1/models > /dev/null && curl -sf -m 5 http://127.0.0.1:8200/v1/models > /dev/null; then
  PREFILL_NUM_WORKERS=0 bash scripts/gate.sh kf47897-on2 http://127.0.0.1:8100 "http://127.0.0.1:8100/metrics" > $out/gate-on.out 2>&1
  grep -a -c "gluon paged-MQA logits" logs/engines/vpd-aggK.log > $out/gluon-used-after-gate.txt
  for side in on:8100 off:8200; do
    name=${side%%:*} port=${side##*:}
    docker run --rm --network host -v $out:/out -v /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4:/model:ro \
      --entrypoint vllm $IMAGE bench serve --backend vllm --base-url http://127.0.0.1:$port \
      --endpoint /v1/completions --model glm5.2-mxfp4 --tokenizer /model --trust-remote-code --dataset-name random \
      --random-input-len 65536 --random-output-len 1024 --num-prompts 208 --max-concurrency 104 --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /out --result-filename bench-$name-c104.json \
      > $out/bench-$name-c104.log 2>&1 &
  done
  wait
else
  echo "$(date -u +%m%d-%T) kf47897-ab2: deployment failed" | tee -a logs/summary.out
fi
for c in vpd-aggK vpd-aggR; do docker logs $c 2>&1 | gzip > $out/$c.log.gz; done
docker rm -f vpd-aggK vpd-aggR > /dev/null 2>&1
python3 - "$out" <<'EOF' | tee -a logs/summary.out
import glob, json, os, sys
out = sys.argv[1]
row = []
for f in sorted(glob.glob(f"{out}/bench-*.json")):
    d = json.load(open(f))
    row.append(f"{os.path.basename(f)[6:-5]}: tpot p50 {d.get('median_tpot_ms', 0):.1f} ms, itl p50 {d.get('median_itl_ms', 0):.1f} ms, "
               f"out tok/s {d.get('output_throughput', 0):.0f}")
print("kf47897-ab2 " + "; ".join(row))
EOF
exit 0
