# Reproduction — conc=128 digit loop on GLM-5.2-MXFP4, DPA + mooncake PD

Est. ~10 min bring-up (weights warm) + ~1 min per stress arm. The whole thing is fast because the
failure shows up in the first 128-request wave.

## 0. Prerequisites

- `infera/engine-sglang:pd-unified-waitevent` on **both** nodes. Not on a registry — it is
  `pd-unified` + Infera 854ebf70, `docker commit`-ed. Verify the patch is really in:

      docker exec pd_uni grep -n "wait_event.synchronize" \
        /sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py
      # -> 1233:  kv_chunk.wait_event.synchronize()

- Model `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST mount).
- A `pd_uni` container on each node with host `libionic` injected, 8 ionic NICs `PORT_ACTIVE`.
  (`glm52.longctx.packup_20260729/scripts/up_dpa_longctx.sh` does the container prep; this kit
  reuses those containers and only relaunches the server processes.)
- Shared kit dir `KIT=/mnt/vast/c_huggingface/glm52_longctx_pd`, and
  `pd_leg_dpa_longctx.sh` staged there (unchanged from packup exp07).

Scripts assume a jump host `ssh root@149.28.124.225` then `ssh <node>`; adjust `J()` if different.

## 1. Stage the client

    cat scripts/stress_capture.py | ssh <jump> "ssh chi2867 'cat > $KIT/stress_capture.py'"

## 2. Relaunch both legs at exp07 1k/1k capacity

    bash scripts/up_conc128.sh        # ctx 32768, chunk 65536, maxrun 2048, cgbs 128, DPA=1

Wait for both (~2-6 min if weights are in page cache, 6-10 min cold):

    grep -ac "ready to roll" $KIT/pd_prefill_30000_c128.log   # 1
    grep -ac "ready to roll" $KIT/pd_decode_30001_c128.log    # 1

Confirm the capacity knobs and RDMA actually took:

    grep -ao "chunked_prefill_size=[0-9]*\|max_running_requests=[0-9]*" $KIT/pd_prefill_30000_c128.log | head -4
    # -> chunked_prefill_size=8192   max_running_requests=256   (per-rank, /dp_size 8)
    grep -ac "HIP dmabuf disabled" $KIT/pd_prefill_30000_c128.log   # >0  (mooncake RDMA)
    grep -aic "MC_FORCE_TCP" $KIT/pd_prefill_30000_c128.log          # 0

Then the router (from the longctx packup, unchanged):

    bash ../glm52.longctx.packup_20260729/scripts/run_router.sh
    # -> {"object":"list","data":[{"id":"glm5.2-mxfp4",...}]}

## 3. Baseline at conc=1 (must be clean)

    bash scripts/run_arms.sh base
    # tail $KIT/cap_base.log  ->  CLEAN 32, BAD 0/32

## 4. The stress arm

    bash scripts/run_arms.sh c128
    # tail $KIT/cap_c128.log  ->  ~506 CLEAN, ~6 bad, BAD 5-7/512

Expect **1.2–2 %** bad. Every bad row has `finish=length`; every good row has `finish=stop`.

## 5. The decisive control — replay the failures at conc=1

Prompt content is a pure function of `idx` + `salt`, so replaying an index reproduces the exact
bytes that failed under load:

    IDX=71,84,101,112,119,121 REP=4 python3 $KIT/stress_capture.py \
        http://10.2.122.44:8002 glm5.2-mxfp4 1 0 1024 1024 $KIT/cap_replay1.json 0

    # -> 24/24 CLEAN. Not the prompts; the concurrency.

Substitute your own run's failing indices (`jq '.rows[]|select(.finish=="length")|.idx'`).

## 6. Second run with a fresh salt (rules out prefix-cache masking)

    python3 $KIT/stress_capture.py http://10.2.122.44:8002 glm5.2-mxfp4 128 512 1024 1024 \
        $KIT/cap_c128b.json 777

Salt 777 makes every prompt novel. Same failure rate.

## 7. The control arm — single-node mix, no DPA, no PD

Kills the PD leg on the prefill node and relaunches the same container as a plain colocated
server. Per-rank chunk is set to 8192 so the compute shape matches the PD run (65536 ÷ dp8):

    bash scripts/up_single_nodpa.sh          # ~2-8 min cold start
    grep -ac "ready to roll" $KIT/single_nodpa_30000.log     # 1

Hit it **directly** — it is not a PD leg, there is no router:

    K=/mnt/vast/c_huggingface/glm52_longctx_pd
    for a in "1 32 sn_base 0" "128 512 sn_c128 0" "128 512 sn_c128b 777"; do
      set -- $a
      docker exec pd_uni python3 $K/stress_capture.py http://10.2.122.44:30000 glm5.2-mxfp4 \
        $1 $2 1024 1024 $K/$3.json $4
    done
    # -> 32/32, 512/512, 511/512 CLEAN. 0 digit loops in 1024 stressed requests.

Then the one-variable split — same single node, DP-attention back ON:

    K=/mnt/vast/c_huggingface/glm52_longctx_pd
    cat scripts/single_dpa1.sh | ssh <jump> "ssh chi2867 'cat > $K/single_dpa1.sh'"
    ssh <jump> "ssh chi2867 'docker exec pd_uni pkill -9 -f launch_server; sleep 5;
        docker cp $K/single_dpa1.sh pd_uni:/single_dpa1.sh;
        docker exec -d pd_uni bash /single_dpa1.sh'"
    grep -ac "ready to roll" $K/single_dpa1_30000.log     # 1
    grep -aoE "enable_dp_attention=True|chunked_prefill_size=[0-9]+" $K/single_dpa1_30000.log | sort -u
    # -> enable_dp_attention=True, chunked_prefill_size=8192  (per-rank, matches the other arms)

    for a in "sd_c128 0" "sd_c128b 777"; do set -- $a
      docker exec pd_uni python3 $K/stress_capture.py http://10.2.122.44:30000 glm5.2-mxfp4 \
        128 512 1024 1024 $K/$1.json $2
    done
    # -> 512/512 and 511/512 CLEAN. Still 0 digit loops.

> **Launch this one as a script FILE.** `docker exec -d $CTR bash -c "... > $LOG"` through nested
> ssh makes the redirect evaluate on the outer shell: the log gets the ssh banner (76 bytes) and
> the server never starts. Same class of bug as the pitfalls below.

Bringing the PD arm back afterwards is just `bash scripts/up_conc128.sh` + `run_router.sh`.

## 8. Score / inspect

    python3 - <<'PY'
    import json, collections
    d = json.load(open('cap_c128.json'))
    print(collections.Counter(r.get('verdict_v2', r['verdict']) for r in d['rows']))
    for r in d['rows']:
        if r['finish'] == 'length':
            print(r['idx'], r.get('verdict_v2'), 'needle=', r['expect'] in r['output'],
                  'thinktags=', r['output'].count('</think>'))
            print('  TAIL:', repr(r['output'][-140:]))
    PY

## Pitfalls (both cost us a relaunch)

- **`J()` quoting.** `J()` already wraps the remote command in `'...'`. Putting a
  `bash -c '...'` inside it silently collapses the quoting and the command never runs. Our kill
  step failed that way, the old legs survived, and the new legs died instantly with
  `ValueError: port_base at 30234 is not available`. Use `docker exec $CTR pkill -9 -f launch_server`
  with no nested quotes.
- **Stale router.** The router process is named `sglang::router`, so `pkill -f sglang_router`
  does **not** match it. It keeps :8002 and :29000, and a fresh router dies with
  `failed to install Prometheus metrics exporter: Address already in use`. Find it with
  `ss -ltnp | grep :8002` and `kill -9` the pid **on the host**, not in the container.
- **Never send a request directly to `:30000` / `:30001`** — `req.bootstrap_room should not be
  None` SIGQUITs the whole leg. Router only.
- **Don't judge by `bench_serving`.** A digit loop returns HTTP 200 with a full token budget; the
  completion rate is 512/512 either way. You must capture and classify the text.
- **Don't trust a naive repeated-digit detector.** A correct chain-of-thought quotes its numeric
  answer many times. Keep the density guard (>25 % of output) and re-run the regression suite in
  `notes.md` §classifier if you change the rules.
