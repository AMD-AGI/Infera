export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/liyingli USER=liyingli R=/home/liyingli/bench_agentx/vllm-pd-20261117 NODE0=$(hostname -s)
# Decode MoE communication on the P4D4 decode config (DP4 TP1, fp8 KV, MTP3, block 64), aggregated, stock v0.34.0.
# The s6c decode trace shows 2 RCCL generic kernels per layer (71.5 us each, ~26% of a decode step) for the MoE
# AllGather+ReduceScatter dispatch/combine, although vLLM has an aiter custom AG/RS path for DP groups.
#   round 1: default (GPUs 0-3) vs --enable-expert-parallel (GPUs 4-7)
#   round 2: --enable-expert-parallel --all2all-backend mori (GPUs 0-3)
# Each: vllm bench serve (random 64K prompts, 1024 out, ignore EOS, concurrency 104) with a torch trace mid-decode.
while ! ls queue/*-Lp4d4s7c.rc > /dev/null 2>&1; do sleep 20; ls queue > /dev/null; done
docker rm -f vpd-aggK vpd-aggR vpd-aggA vpd-aggB > /dev/null 2>&1; sleep 20
export IMAGE=infera-vllm:v0340-0c5e5e89
out=$R/results/decode-comm; mkdir -p $out
a=(--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
   --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192
   --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
   --profiler-config '{"profiler":"torch","torch_profiler_dir":"/trace","torch_profiler_with_stack":false}')
bench() {  # name port
  local name=$1 port=$2
  ( sleep 420; bash scripts/trace_decode.sh http://127.0.0.1:$port 3 > $out/$name/trace.log 2>&1 ) &
  docker run --rm --network host -v $out/$name:/out -v /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4:/model:ro \
    --entrypoint vllm $IMAGE bench serve --backend vllm --base-url http://127.0.0.1:$port --endpoint /v1/completions \
    --model glm5.2-mxfp4 --tokenizer /model --trust-remote-code --dataset-name random --random-input-len 65536 \
    --random-output-len 1024 --num-prompts 208 --max-concurrency 104 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /out --result-filename bench-c104.json > $out/$name/bench-c104.log 2>&1
  wait
}
run_round() {  # "name gpus port rpc extra..." ...
  local specs=("$@") s
  for s in "${specs[@]}"; do
    read -r name gpus port rpc extra <<< "$s"
    mkdir -p $out/$name && chmod 777 $out/$name
    ( MOUNT_EXTRA="$out/$name:/trace" bash scripts/vllm_up.sh vpd-$name $gpus $port "${a[@]}" --data-parallel-rpc-port $rpc $extra \
        > $out/$name/up.log 2>&1 ) &
    sleep 30
  done
  wait
  for s in "${specs[@]}"; do
    read -r name gpus port rpc extra <<< "$s"
    if curl -sf -m 5 http://127.0.0.1:$port/v1/models > /dev/null; then bench $name $port & else echo "$name: not up" >> $out/summary.txt; fi
  done
  wait
  for s in "${specs[@]}"; do
    read -r name gpus port rpc extra <<< "$s"
    docker logs vpd-$name 2>&1 | gzip > $out/$name/engine.log.gz; docker rm -f vpd-$name > /dev/null 2>&1
  done
  sleep 20
}
run_round "agrs 0,1,2,3 8100 29610" "ep 4,5,6,7 8200 29710 --enable-expert-parallel"
run_round "epmori 0,1,2,3 8100 29610 --enable-expert-parallel --all2all-backend mori"
for d in $out/*/; do
  n=$(basename $d); f=$d/bench-c104.json
  [ -s $f ] && python3 -c "import json,sys; d=json.load(open('$f')); print('$n: tpot p50 %.1f ms, itl p50 %.1f ms, out tok/s %.0f' % (d['median_tpot_ms'], d['median_itl_ms'], d['output_throughput']))" >> $out/summary.txt
  t=$(ls $d/dp0_*.pt.trace.json.gz 2>/dev/null | head -1)
  [ -n "$t" ] && python3 scripts/trace_cat.py "$t" --top 12 > $d/trace_cat.txt 2>&1
done
echo "$(date -u +%m%d-%T) decode-comm: $(tr '\n' ';' < $out/summary.txt)" >> logs/summary.out
exit 0
