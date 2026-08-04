# REPRODUCE

Top-to-bottom. Cold start is **5–8 min per leg** (weights → tilelang JIT) — not
a hang. Total: ~15 min bring-up + ~17 min per concurrency point.

```bash
JUMP=root@149.28.124.225
W=/mnt/vast/c_huggingface/bench_20260801        # node workspace (scripts + logs)
J=/root/agentx_rr_20260804                      # jump-host workspace
BENCH=/root/agentx_20260803/bench               # corpus + tokenizer, from the previous kit
PREFILL_NODE=chi2835 ; PREFILL_IP=10.2.122.78
DECODE_NODE=chi2879  ; DECODE_IP=10.2.122.10
ROUTER=http://10.2.122.78:8100
```

---

## 0. Prerequisites

- **Slurm holds belong to `yeandy-debug` on both nodes. Never `scancel`.** Kill
  only your own engine / router processes.
- The corpus, tokenizer and `aiperf-agentx:v1.0` image from
  `../agentx.caseA.customer.packup_20260803` must be staged at `$BENCH`. If they
  are not, run that kit's REPRODUCE steps 1–4 first.
- No secrets required.

## 1. Fix the port-block bug — REQUIRED before prefill DPA can start

Skip only if your node's `ip_local_port_range` already starts well above 1024
**and** you never need the in-range fallback. Check first:

```bash
ssh $JUMP "ssh $PREFILL_NODE 'cat /proc/sys/net/ipv4/ip_local_port_range'"
```

### 1a. The engine patch

`patches/net.py.patched` is the finished file (md5
`efb8e3621c1be99744f3e7698bde6937`); `patches/net_port_block_low_ephemeral.patch`
is the same change as a diff if you would rather apply it to your own checkout.

```bash
cat patches/net.py.patched | ssh $JUMP "cat > $W/scripts/net.py.patched"
ssh $JUMP "ssh $PREFILL_NODE '
  docker cp $W/scripts/net.py.patched bench_run:/opt/infera/infera/common/net.py
  docker exec bench_run rm -f /opt/infera/infera/common/__pycache__/net*.pyc'"
```

**Verify the bytecode, not the source** (par8 Trap 2):

```bash
ssh $JUMP "ssh $PREFILL_NODE 'docker exec bench_run bash -c \
  \"strings /opt/infera/infera/common/__pycache__/net.cpython-310.pyc | grep -c no.room.below\"'"
# -> 1   (and the .pyc mtime must be AFTER the patch)
```

### 1b. The sysctl — needed if the range starts at 1024

```bash
ssh $JUMP "ssh $PREFILL_NODE 'sysctl -w net.ipv4.ip_local_port_range=\"32768 60999\"'"
```

Without this the patch's fallback picks a block **inside** the ephemeral range
and one rank loses its port between probe and bind
(`ZMQError: Address already in use`). See `patches/README.md` §3.
**Runtime only — reverts on reboot.**

## 2. Stage the router script

`start_router.sh` in the par8 kit hardcodes `--router-policy kv-aware`.
`scripts/start_router_pol.sh` is that file with a `POLICY` variable added; with
`POLICY` unset it emits a byte-identical command line.

```bash
scp scripts/start_router_pol.sh $JUMP:$W/scripts/
```

## 3. Launch the prefill leg — DPA ON, CHUNK 65536, gmu 0.70

**`CHUNK` is the load-bearing argument.** sglang divides
`chunked_prefill_size` by `dp_size` **only** under DPA
(`server_args.py:4902`), so:

| passed | resolved per forward |
|---|---|
| 16384 | 2048 ← wrong; 1/8 of the 20260803 posture |
| **65536** | **8192** ← what this kit ran |
| 131072 | 16384 ← matches 20260803's per-forward work, but **crashed DP6** (see `notes.md`) |

```bash
ssh $JUMP "ssh $PREFILL_NODE 'cd $W && \
  ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
  DPA=1 CHUNK=65536 GMU_P=0.70 MTP=0 TAG=r4 bash scripts/start_leg.sh'"
```

Gate before going further — read the **resolved** value out of the log:

```bash
ssh $JUMP "ssh $PREFILL_NODE 'strings $W/logs/r4_prefill.log \
  | grep -a \"chunked prefill size is adjusted\" | tail -1'"
# -> ...adjusted to 8192...
```

Wait for readiness (**never grep an appended log without scoping** — it matches
the previous run):

```bash
ssh $JUMP "ssh $PREFILL_NODE 'strings $W/logs/r4_prefill.log | grep -a \"fired up\" | tail -1'"
```

## 4. Relaunch the decode leg — REQUIRED after any prefill restart

The decode leg holds a bootstrap connection to the **previous** prefill
instance. Without this step every request fails with
`KVTransferError: Aborted by AbortReq` on the prefill side and
`Lost connection with prefill instance` on the decode side, while
`/health` still cheerfully reports `active_workers: 2`.

```bash
ssh $JUMP "ssh $DECODE_NODE 'cd $W && \
  ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP MTP=1 TAG=r4d bash scripts/start_leg.sh'"
```

Confirm GLM52_P1V3 is still in the decode container's **bytecode** (it is baked
in from the par8 work; a container recreate would lose it):

```bash
ssh $JUMP "ssh $DECODE_NODE 'docker exec bench_run bash -c \
  \"strings /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__/dsa_indexer*.pyc \
    | grep -c _p1v2_rows\"'"
# -> 1
```

## 5. Start the router

```bash
# kv-aware leg (this kit's main run)
ssh $JUMP "ssh $PREFILL_NODE 'cd $W && \
  MY_IP=$PREFILL_IP POLICY=kv-aware BACKEND=rust PW=2.0 DW=2.0 bash scripts/start_router_pol.sh'"

# round-robin leg
#   MY_IP=$PREFILL_IP POLICY=round-robin BACKEND=rust bash scripts/start_router_pol.sh
```

The script prints both the **requested** and the **resolved** policy. They can
differ — see `notes.md` Trap 2. Insist on:

```
resolved --router-policy: kv-aware
```

## 6. Verify every feature BEFORE spending the window

| check | command | expected |
|---|---|---|
| both legs | `curl -s $ROUTER/health` | `active_workers: 2` |
| **prefill DPA on** | `curl -s $ROUTER/v1/workers` | prefill `dp_size: 8` (was `null` in the 20260803 posture) |
| 8 schedulers | `docker exec bench_run ps -eo args \| grep -oE 'scheduler_DP[0-9]+' \| sort -u \| wc -l` | `8` |
| **chunk resolved** | see step 3 | **8192** |
| gmu | `ps -eo args \| grep '[l]aunch_server'` | `--mem-fraction-static 0.70` |
| **policy** | `docker exec bench_run ps -eo args \| grep '[i]nfera-router'` | the policy you asked for |
| RDMA | `strings <leg>.log \| grep -cE 'MC_FORCE_TCP\|GID is NULL'` | `0` on both legs |
| MTP | `strings r4d_decode.log \| grep -oE 'accept len: [0-9.]+' \| tail` | 1.5–3.9. **4.00 is BAD** (repetition loop) |
| kvd baseline | `docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock` | record it; do not expect zeros on a warm node |

Smoke test **through the router** (never probe a leg's own port — it hangs):

```bash
ssh $JUMP "curl -sf -m180 $ROUTER/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"glm5.2-mxfp4\",\"messages\":[{\"role\":\"user\",
       \"content\":\"What is 17 multiplied by 23? Answer with the number only.\"}],
      \"max_tokens\":200,\"temperature\":1.0,\"top_p\":0.95}'"
# -> 391, coherent, finish_reason: stop
```

## 7. Run the sweep

```bash
scp scripts/{run_caseA.sh,rescue_artifacts.sh,analyze.py} $JUMP:$J/
ssh $JUMP "cd $J && nohup bash rescue_artifacts.sh > rescue.log 2>&1 &"
ssh $JUMP "cd $J && setsid nohup bash run_caseA.sh > run_caseA.log 2>&1 < /dev/null &"
```

`run_caseA.sh` sets env only and calls the customer's script unmodified; it
prints the md5 so you can see that. `CONCS=8` here.

The rescue loop exists because `OUT` sits outside `$HERE`, which the customer's
script does not mount — see `notes.md` Trap 3. **Stop it only after the aiperf
container has exited**, or the last artifacts are lost:

```bash
ssh $JUMP 'docker ps --filter ancestor=aiperf-agentx:v1.0 -q'   # must be empty
ssh $JUMP 'pkill -f "rescue_[a]rtifacts"'   # bracket avoids matching your own ssh cmdline
```

Expect the sweep to print `FAILED` for a run that succeeded — same known defect.
The truth is in the rescued artifacts:

```bash
ssh $JUMP "grep -aE 'Phase profiling sending complete' $J/rescue/c8_art/logs/aiperf.log | tail -1"
# -> sent=136, completed=131, ...
```

## 8. Analyse

```bash
ssh $JUMP "cd $J && python3 analyze.py rescue/c8_art"
```

Expected for this kit's kv-aware leg:

```
profiling requests : 135   window 914.8 s   0.148 req/s
TTFT p50 25,188 ms | ITL p50 14.85 ms | server cache hit 27.37 % (n=65)
```

Also read the router's own view of what the policy did:

```bash
ssh $JUMP "ssh $PREFILL_NODE 'docker exec bench_run gzip -c /tmp/router.log'" > router.log.gz
zcat router.log.gz | sed 's/\x1b\[[0-9;]*m//g' | grep role=Prefill \
  | grep -oE 'dp[0-9]+ cache_hits=[0-9]+ request_blocks=[0-9]+'
```

**If `cache_hits` is 0 on requests with `request_blocks` in the thousands, the
kv-aware policy is not actually running** — that is what happened here. See
`analysis/policy_ab.md`.

`/tmp/router.log` lives **inside the container and is truncated by the next
`start_router_pol.sh`.** Capture it before restarting the router.

## 9. Re-derive the overlap weight (optional, no GPU needed)

```bash
ssh $JUMP "cd $BENCH && python3 -c \"
import json,glob,statistics as st
miss=[];tot=[]
for f in sorted(glob.glob('corpus/*.json')):
    d=json.load(open(f)); turns=d if isinstance(d,list) else d.get('turns') or d.get('requests') or []
    seen=set()
    for t in turns:
        h=t.get('hash_ids') or []
        if not h: continue
        tot.append(len(h)); miss.append(len(h)-len([x for x in h if x in seen])); seen.update(h)
q=lambda a,p: sorted(a)[int(p*len(a))]
print('blocks p50', q(tot,.5), 'warm-miss p50', q(miss,.5), '=> hits', q(tot,.5)-q(miss,.5))
\""
# -> blocks p50 1158  warm-miss p50 192  => hits 966
```

Reasoning from that number to `pw=2.0`: `analysis/overlap_weight_derivation.md`.

## External dependencies

| what | where |
|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST) |
| corpus, tokenizer copy, aiperf image | `/root/agentx_20260803/bench/` + local image `aiperf-agentx:v1.0` |
| customer bench | `github.com/ROCm/MAD` PR **#173**; `spec/replay_caseA.sh` here is verbatim |
| the deployment | chi2835 + chi2879, image `infera/engine-sglang:merged-e` **plus the `net.py` patch** |

## Secrets

**None.** No registry login, no API key, no cluster credential beyond the SSH
access already needed to reach the jump host.
