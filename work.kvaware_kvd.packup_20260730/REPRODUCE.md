# Reproduction kit

Three independently reproducible things. §A is a 30-second desk check, §B needs
one node for ~10 min, §C needs two nodes for ~20 min.

Prerequisites for §B/§C: read `environment.md` first — image, weights path,
`libionic` injection, and the ionic fabric check all matter.

Throughout:

```bash
JUMP=root@149.28.124.225
J(){ ssh -o StrictHostKeyChecking=no $JUMP "ssh -o StrictHostKeyChecking=no $1 '$2'"; }
KIT=/mnt/vast/c_huggingface/glm52_kvexp     # shared FS, visible from BOTH nodes
```

---

## A. The port-collision bug + its fix (no GPU, no cluster)

Reproduce the bug against the pre-fix code:

```bash
cd <infera repo>
mkdir -p /tmp/oldnet
git show 362192e7:infera/common/net.py > /tmp/oldnet/net_old.py
python3 - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("net_old", "/tmp/oldnet/net_old.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bases = [m.free_tcp_port_block(4) for _ in range(10)]
print("OLD bases:", bases, "| distinct:", len(set(bases)))
EOF
```

Expect all ten identical (`32764` on a box whose `ip_local_port_range` starts at
32768) — that is the bug.

Apply the fix and test:

```bash
git apply patches/0001-free_tcp_port_block-randomise-scan-start.patch
cp patches/test_net_port_block.py tests/unit/common/
python3 -m pytest tests/unit/common/test_net_port_block.py -q     # -> 4 passed
```

Details and the two rejected alternative fixes: `patches/0001-note.md`.

---

## B. The sglang arg-compatibility matrix (one node, no model load)

Shows *why* kvd needs kvaware on a decode leg. Runs sglang's own arg parser —
no GPU work, but needs a container with the model path mounted (the parser
touches the tokenizer).

```bash
J chi2879 "docker run -d --name argchk --network=host -v /mnt/vast:/mnt/vast \
   --entrypoint '' infera/engine-sglang:pd-unified sleep infinity"
cat scripts/argcheck.py | ssh $JUMP "ssh chi2879 'docker exec -i argchk bash -lc \"cat > /tmp/argcheck.py\"'"
J chi2879 "docker exec argchk python3 /tmp/argcheck.py 2>&1 | grep -E 'OK  \]|FAIL\]'"
J chi2879 "docker rm -f argchk"
```

Expect rows 1-4 and 6 `OK`; rows 5, 7, 8 `FAIL` with
`enable-hierarchical-cache and disable-radix-cache are mutually exclusive`.
Interpretation: `results/support_matrix.md`.

---

## C. GLM-5.2 two-node PD + DP-attention, 4/4 correctness (the headline result)

**Roles:** prefill `chi2879` (10.2.122.10), decode `chi2867` (10.2.122.44).
**Cold start:** ~6 min to "ready to roll" on both legs. It is not a hang.

### C.0 Pre-flight — the fabric (skip and you may waste the cold start)

```bash
for h in chi2879 chi2867; do J $h "ibv_devinfo | grep -c PORT_ACTIVE"; done   # -> 8, 8
J chi2879 "ping -c2 -W2 10.2.122.44 | tail -2"                               # -> 0% loss
```

### C.1 Stage the kit onto the shared FS

```bash
J chi2879 "mkdir -p $KIT"
cd <this packup>
tar cf - -C scripts glm52_leg.sh net_fixed.py probe.py \
  | ssh $JUMP "ssh chi2879 'cd $KIT && tar xf -'"
```

`glm52_up.sh` runs from your workstation (it drives both nodes over the jump
host) and expects `glm52_leg.sh`, `net_fixed.py`, `probe.py` in `$KIT`.

### C.2 Bring up the baseline (kvaware OFF, kvd OFF)

```bash
KVAWARE=0 KVD=0 POLICY=round-robin TAG=base bash scripts/glm52_up.sh
```

This preps both containers (libionic inject + `docker cp` of the net.py fix),
starts etcd on the prefill node, and launches both legs.

Poll until **both** report 1:

```bash
J chi2879 "docker exec glm52_kvexp grep -ac 'ready to roll' $KIT/pd_prefill_base.log"
J chi2867 "docker exec glm52_kvexp grep -ac 'ready to roll' $KIT/pd_decode_base.log"
```

### C.3 Start the router

`glm52_up.sh` starts it too, but via `docker exec -d … bash -lc`, **which does
not persist** (see notes.md). Use the script-file form:

```bash
scp scripts/run_router.sh $JUMP:/tmp/
ssh $JUMP "scp /tmp/run_router.sh chi2879:/tmp/"
J chi2879 "docker cp /tmp/run_router.sh glm52_kvexp:/run_router.sh && \
           docker exec -d glm52_kvexp bash /run_router.sh"
sleep 20
J chi2879 "docker exec glm52_kvexp curl -s -m10 http://10.2.122.10:8100/v1/workers"
```

Expect two workers, `disagg_mode` `prefill` and `decode`, both `active`. With
`KVAWARE=0` their `kv_events_endpoint` is `null` — that is the switch confirming
itself.

### C.4 Correctness — the actual result

```bash
J chi2879 "docker exec glm52_kvexp timeout 400 python3 /tmp/probe.py \
             http://10.2.122.10:8100 glm5.2-mxfp4"
```

**Expect `4/4 correct`** (probe.py exits non-zero below 3/4). Reference output:
`results/baseline_probe_4of4.txt`.

### C.5 Confirm it was real RDMA, and that PD+DPA were both on

```bash
J chi2879 "docker exec glm52_kvexp bash -c \"grep -ac MC_FORCE_TCP $KIT/pd_prefill_base.log\""          # -> 0
J chi2879 "docker exec glm52_kvexp bash -c \"grep -ac 'HIP dmabuf disabled' $KIT/pd_prefill_base.log\"" # -> 8
J chi2879 "docker exec glm52_kvexp bash -c \"grep -aoE 'enable_dp_attention=True|dp_size=8|ep_size=8' $KIT/pd_prefill_base.log | sort -u\""
```

All three lines must appear on **both** legs — asymmetric DPA mismatches the KV
shard layout across the mooncake transfer. Reference:
`results/{transport_was_real_rdma,pd_dpa_flags_verified}.txt`.

### C.6 Tear down (shared cluster — do this)

```bash
for h in chi2879 chi2867; do J $h "docker rm -f glm52_kvexp"; done
J chi2879 "docker rm -f glm52_kvexp_etcd"
for h in chi2879 chi2867; do J $h "rocm-smi --showmeminfo vram | grep -i used"; done
```

VRAM should fall back to the ~300 MB/GPU idle baseline. It can take ~60 s.

---

## D. Turning kvaware + kvd on — **verified 4/4** (step 1)

Same kit, same nodes, only the two switches differ from §C. Result:
`results/step1_kvaware_kvd_4of4.txt`. Cold start is longer than the baseline
(~11 min to both legs ready, vs ~6).

```bash
# kvd daemon on both nodes FIRST (script-file form; -d + bash -lc does not persist)
scp scripts/run_kvd.sh $JUMP:/tmp/
for h in chi2879 chi2867; do
  ssh $JUMP "scp /tmp/run_kvd.sh $h:/tmp/"
  J $h "docker cp /tmp/run_kvd.sh glm52_kvexp:/run_kvd.sh && docker exec -d glm52_kvexp bash /run_kvd.sh"
done
sleep 25
for h in chi2879 chi2867; do J $h "docker exec glm52_kvexp test -S /tmp/kvd/kvd.sock && echo $h kvd_UP"; done

KVAWARE=1 KVD=1 POLICY=kv-aware TAG=kv HICACHE_GB=16 bash scripts/glm52_up.sh
```

Then C.3 (edit `run_router.sh` to `--router-policy kv-aware`) and C.4 → **4/4**.

Verify the switches were genuinely live, not just requested:

```bash
# kvaware: both workers advertise an endpoint (null when OFF)
J chi2879 "docker exec glm52_kvexp curl -s -m10 http://10.2.122.10:8100/v1/workers"

# kvaware: infera's own probe plane came up
J chi2879 "docker exec glm52_kvexp grep -a 'KV plane up:' $KIT/pd_prefill_kv.log"

# kvd: adapter connected on every DP rank, on BOTH legs (expect 8 and 8)
J chi2879 "docker exec glm52_kvexp grep -ac 'infera-kvd adapter connected' $KIT/pd_prefill_kv.log"
J chi2867 "docker exec glm52_kvexp grep -ac 'infera-kvd adapter connected' $KIT/pd_decode_kv.log"

# kvd: is it actually SERVING, or merely connected? (step 1: all zeros)
J chi2879 "docker exec glm52_kvexp python3 -c \"
import asyncio; from infera.kvd.client import KvdClient
async def m():
    c=KvdClient('/tmp/kvd/kvd.sock', client_id='stats'); await c.connect()
    print(await c.stats()); await c.close()
asyncio.run(m())\""
```

`--hicache-size 16` is deliberate: the default `--hicache-ratio 2.0` sizes the
host pool off `max_total_num_tokens` and tried to allocate **355 GB per DP rank**
in the MVP. See notes.md.

---

## E. Step 2 — make kvd actually serve (prefix-reuse workload)

Runs against the **same live deployment** as §D; nothing needs restarting except
the router (and only if you want the role weights).

```bash
# 1. snapshot kvd counters BEFORE
for h in chi2879 chi2867; do
  ssh $JUMP "scp /tmp/kvdstats.sh $h:/tmp/" 2>/dev/null
  J $h "docker cp scripts/kvdstats.sh glm52_kvexp:/kvdstats.sh; docker exec glm52_kvexp bash /kvdstats.sh"
done

# 2. stage + run the workload (4 sessions x 4 turns, ~6200-token shared prefix)
cat scripts/prefix_reuse.py | ssh $JUMP "ssh chi2879 'cat > $KIT/prefix_reuse.py'"
J chi2879 "docker cp $KIT/prefix_reuse.py glm52_kvexp:/tmp/prefix_reuse.py"
J chi2879 "docker exec glm52_kvexp timeout 900 python3 /tmp/prefix_reuse.py \
             http://10.2.122.10:8100 glm5.2-mxfp4 --sessions 4"

# 3. snapshot kvd counters AFTER — this is the real result
for h in chi2879 chi2867; do J $h "docker exec glm52_kvexp bash /kvdstats.sh"; done
```

Expect **32/32 correct**, and kvd moving from `gets=0` to
`gets=170 hits=170 misses=0` with ~573 MB resident.

Optional — PD role weights (verified active, effect unmeasured with 1 worker/role):

```bash
scp scripts/run_router_weighted.sh $JUMP:/tmp/
ssh $JUMP "scp /tmp/run_router_weighted.sh chi2879:/tmp/"
J chi2879 "docker cp /tmp/run_router_weighted.sh glm52_kvexp:/router_w.sh && \
           docker exec -d glm52_kvexp bash /router_w.sh"
sleep 25
J chi2879 "docker exec glm52_kvexp grep -a overlap /tmp/router.log | head -1"
# -> router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0
```

**Reading the latency:** a second run is much faster (0.71s → 0.26s median), but
kvd's counters do **not** move — that is the GPU radix cache, not kvd. Don't
attribute it to kvd. See `results/step2_prefix_reuse.txt`.

---

## F. Step 3 — cross-restart reuse (proves kvd is the thing doing the work)

The confounder-free version of §E: kill the **engine** but keep **kvd**, so the
GPU cache is empty and the only possible source of a hit is kvd's store.

```bash
# 1. kill engine legs ONLY. run_kvd.sh's daemon must survive.
scp scripts/restart_legs.sh $JUMP:/tmp/
for h in chi2879 chi2867; do
  ssh $JUMP "scp /tmp/restart_legs.sh $h:/tmp/"
  J $h "docker cp /tmp/restart_legs.sh glm52_kvexp:/restart_legs.sh && \
        docker exec glm52_kvexp bash /restart_legs.sh"
done
# expect: "kvd alive: 1" on both

# 2. VERIFY the preconditions — this is what makes the test valid
sleep 30
for h in chi2879 chi2867; do
  J $h "rocm-smi --showmeminfo vram | grep -i used | head -1"   # ~297 MB = engine truly dead
  J $h "docker exec glm52_kvexp bash /kvdstats.sh"              # entries/bytes still there
done

# 3. relaunch legs (same command as §D), wait for both "ready to roll", restart router

# 4. replay the SAME workload, then diff kvd counters
J chi2879 "docker exec glm52_kvexp timeout 900 python3 /tmp/prefix_reuse.py \
             http://10.2.122.10:8100 glm5.2-mxfp4 --sessions 4"
for h in chi2879 chi2867; do J $h "docker exec glm52_kvexp bash /kvdstats.sh"; done
```

**The result to look for** (prefill node): `gets_total` and `hits_total` each
`+170`, while `sets_total` / `entries` / `host_bytes` stay **exactly flat**. A
brand-new engine process read 170 blocks it never wrote. See
`results/step3_restart_reload.txt`.

---

## G. Step 4 — does kv-aware routing actually route? (needs 2 workers/role)

With one worker per role the scorer cannot express a preference. Add a second
decode worker, then run the same workload under two policies and compare where
the requests land.

```bash
# two TP4 decode workers on chi2867. Every one of these three port vars MUST
# differ between the two workers (see notes.md — all three default identically):
#   PORT, KV_PUB_PORT (--kv-events-bind), KV_SNAP_PORT (--kv-snapshot-port)
# GMU=0.70 not 0.85: TP4 doubles weights/GPU (102 GB) and 0.85 OOMs.
J chi2867 "docker exec -d glm52_kvexp env ROLE=decode MY_IP=10.2.122.44 \
   ETCD_IP=10.2.122.10 PORT=30000 TP=4 BASE_GPU=0 GMU=0.70 KVAWARE=1 KVD=1 \
   HICACHE_GB=8 KV_PUB_PORT=5557 KV_SNAP_PORT=8801 LOG=$KIT/pd_decodeA.log bash /glm52_leg.sh"
sleep 25
J chi2867 "docker exec -d glm52_kvexp env ROLE=decode MY_IP=10.2.122.44 \
   ETCD_IP=10.2.122.10 PORT=32000 TP=4 BASE_GPU=4 GMU=0.70 KVAWARE=1 KVD=1 \
   HICACHE_GB=8 KV_PUB_PORT=5657 KV_SNAP_PORT=8802 LOG=$KIT/pd_decodeB.log bash /glm52_leg.sh"

# wait for BOTH "ready to roll", then confirm the router sees THREE workers
J chi2879 "docker exec glm52_kvexp curl -s -m10 http://10.2.122.10:8100/v1/workers"
```

If only two appear, decodeB hit the snapshot-port collision — it logs
"ready to roll" and *then* dies during registration. Check:
`grep -a 'address already in use' $KIT/pd_decodeB.log`.

Now measure the distribution under each policy:

```bash
count(){ for n in A B; do echo -n "decode$n=$(J chi2867 "docker exec glm52_kvexp \
  grep -ac 'Decode batch' $KIT/pd_decode$n.log") "; done; echo; }

count                                            # baseline
# --- kv-aware (weights 20.0/2.0) ---
J chi2879 "docker exec -d glm52_kvexp bash /router_w.sh"; sleep 25
J chi2879 "docker exec glm52_kvexp timeout 900 python3 /tmp/prefix_reuse.py http://10.2.122.10:8100 glm5.2-mxfp4 --sessions 4"
count
# --- round-robin ---
scp scripts/run_router_roundrobin.sh $JUMP:/tmp/ && ssh $JUMP "scp /tmp/run_router_roundrobin.sh chi2879:/tmp/"
J chi2879 "docker cp /tmp/run_router_roundrobin.sh glm52_kvexp:/router_rr.sh && docker exec -d glm52_kvexp bash /router_rr.sh"; sleep 25
J chi2879 "docker exec glm52_kvexp timeout 900 python3 /tmp/prefix_reuse.py http://10.2.122.10:8100 glm5.2-mxfp4 --sessions 4"
count
```

Expected: **kv-aware sends all 32 to decodeA (17/0); round-robin splits (+6/+8)**.
Same workload, same workers — only the policy differs. See
`results/step4_role_weights_routing.txt`.
