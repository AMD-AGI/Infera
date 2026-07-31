# Reproduce

Top-to-bottom. Every script referenced is in `scripts/`. Read `environment.md`
first for the hardware/driver assumptions and the external paths.

Total wall-clock on the original run: **~2 h**, dominated by two image builds
(~25 min), moving a 79 GB image between nodes (~25 min), and two GLM-5.2 cold
starts (~6 min and ~10 min).

## 0. Prerequisites

- Two hosts with 8× MI355X (gfx950), ROCm 7.2.0, ionic RDMA with all 8 rails
  ACTIVE, reachable from each other on the data plane.
- `/mnt/vast` (or equivalent shared storage) mounted on both, holding
  `GLM-5.2-MXFP4` and with ~100 GB free for the image tar.
- ≥ 90 GB free on the **build** node's docker filesystem, and ≥ 90 GB on the
  other node.
- SSH access to both hosts (via the jump host in our setup).

Check the fabric before spending anything:

```bash
for d in /sys/class/infiniband/*; do
  n=$(basename $d); s=$(cat $d/ports/1/state 2>/dev/null)
  drv=$(basename $(readlink -f $d/device/driver 2>/dev/null))
  [[ "$s" == *ACTIVE* && "$drv" == ionic ]] && echo -n "$n "
done; echo
# want: ionic_0 … ionic_7
```

Snapshot the environment on each node — this is what `environment.md` was built
from:

```bash
bash scripts/kvaware_env.sh
```

## 1. Get the branch

```bash
git clone <infera> && cd infera
git checkout -b yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr 8692fb4    # origin/main at the time
git am patches/0001-*.patch patches/0002-*.patch patches/0003-*.patch \
       patches/0004-*.patch patches/0005-*.patch
git log --oneline -5      # expect da65cc7 at HEAD
```

Confirm the fixes and their tests:

```bash
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/unit/kvd/test_storage_classify.py -q
# expect: 51 passed
```

To convince yourself the tests are real, revert **only the two source files**
(keeping the tests) and re-run — expect **3 failed, 48 passed**:

```bash
git checkout 8692fb4 -- infera/common/net.py infera/kvd/storage_classify.py
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/unit/kvd/test_storage_classify.py -q
# 3 failed, 48 passed
git checkout HEAD -- infera/common/net.py infera/kvd/storage_classify.py
```

Use `git checkout <base> --`, not `git stash push`: after `git am` the tree is
clean, so `stash push` has nothing to stash and silently leaves the fixed code in
place — you then "verify" the fix against itself and see 51 passed.

This whole section was run against a fresh clone at `8692fb4` with the five
patches applied: the resulting tree hashes **identical** to the PR branch HEAD
(`a16d0dce342be853e0369681f8fae7fde84d6b2a`).

## 2. Build the image (on the node with the most free disk)

Ship the tracked tree — 6.6 MB, and `.dockerignore` already excludes
`manual/`, `tests/`, `rust/target`:

```bash
git archive --format=tar -o /tmp/src.tar HEAD
scp /tmp/src.tar <build-node>:/tmp/
ssh <build-node> 'mkdir -p ~/build && tar -xf /tmp/src.tar -C ~/build && cd ~/build && \
  docker build -f deploy/docker/Dockerfile.sglang \
    -t infera/engine-sglang:kvaware-kvd-base . && \
  docker build -f deploy/docker/Dockerfile.sglang.kvaware-kvd \
    --build-arg INFERA_SGLANG_IMAGE=infera/engine-sglang:kvaware-kvd-base \
    -t infera/engine-sglang:kvaware-kvd .'
```

Stage 1 (~20 min) rebuilds Mooncake and the Rust router. Stage 2 is seconds and
**must print `kvaware+kvd self-check OK`** — that line is the build asserting the
kvd adapter, the kv probe and the port-collision fix are all present in the image.

## 3. Move the image to the second node

```bash
# On the build node. Note: -o (not a pipe) and setsid — see notes.md, the first
# attempt was silently truncated when the ssh session went away.
setsid nohup bash -c "docker save -o /mnt/vast/<shared>/kvaware-kvd.tar \
  infera/engine-sglang:kvaware-kvd; echo rc=\$? > /mnt/vast/<shared>/save.status" \
  </dev/null >/dev/null 2>&1 &

# wait for save.status to say rc=0 and the tar to be ~79 GB, then on the other node:
setsid nohup bash -c "docker load -i /mnt/vast/<shared>/kvaware-kvd.tar; \
  echo rc=\$? > ~/load.status" </dev/null >/dev/null 2>&1 &
```

Verify the **same digest** on both nodes before continuing:

```bash
docker image inspect infera/engine-sglang:kvaware-kvd --format '{{.Id}}'
# expect sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80
```

## 4. Stage the run kit on shared storage

```bash
KIT=/mnt/vast/<shared>/kvaware_kvd_final
mkdir -p $KIT
cp scripts/glm52_leg.sh scripts/probe.py scripts/stress_capture.py \
   scripts/run_tests.sh scripts/prefix_reuse.py $KIT/
```

## 5. Bring up prefill + etcd + kvd (node A)

Edit the IPs at the top of the script if yours differ.

```bash
bash scripts/kvaware_start_prefill.sh
```

Expect: `active RDMA ports in container: 8` (the entrypoint's libionic injection
worked), `etcd up`, `kvd socket OK`.

Wait ~6 min for the cold start, then:

```bash
L=$KIT/prefill_final.log
echo "ready=$(grep -c 'ready to roll' $L)  kvd=$(grep -c 'infera-kvd adapter connected' $L)"
# want: ready=1  kvd=8
grep -o 'KV plane up:.*' $L
```

## 6. Bring up decode (node B)

```bash
bash scripts/kvaware_start_decode.sh
```

Same checks against `$KIT/decode_final.log`, plus the PD-specific one:

```bash
grep -c 'disaggregation-decode-enable-radix-cache' $KIT/decode_final.log   # want 1
```

If that is 0 the decode leg has no hierarchical cache and kvd is doing nothing
there, whatever else you passed.

## 7. Router (node A)

```bash
bash scripts/kvaware_start_router.sh      # want: "router healthy"
```

Confirm both workers registered:

```bash
docker exec kvaware_kvd_final bash -c \
  "grep -oE 'registered.{0,90}' /tmp/router.log"
# want one PREFILL and one DECODE line
```

## 8. The three acceptance tests

```bash
docker exec -e BASE=http://<node-A-ip>:8100 -e MODEL=glm5.2-mxfp4 \
  -e OUT=/tmp/final_results -e KVD_SOCK=/tmp/kvd/kvd.sock \
  kvaware_kvd_final bash /tmp/run_tests.sh
```

Runs T1 (correctness), T2 (conc=32, 128 req), T3 (conc=128, 512 req), and dumps
kvd counters before/mid/after. ~4 min total.

**Grade on the needle, not on the classifier's verdict** — `stress_capture.py`'s
`CORRUPT_REASONING` rule fires on >5 CJK characters, and GLM-5.2 legitimately
reasons in Chinese sometimes (see `notes.md`):

```bash
docker exec kvaware_kvd_final python3 -c "
import json, collections
for f,l in (('t2_conc32','conc=32'),('t3_conc128','conc=128')):
    d=json.load(open('/tmp/final_results/%s.json'%f)); r=d['rows']
    print(l, 'n=%d'%len(r),
          'needle=%d'%sum(1 for x in r if x['expect'] in x.get('output','')),
          'ERROR=%d'%sum(1 for x in r if x['verdict']=='ERROR'),
          collections.Counter(x.get('finish') for x in r),
          '%.1f req/s'%(len(r)/d['duration_s']))"
```

Expected (ours):

```
conc=32  n=128 needle=128 ERROR=0 Counter({'stop': 128})              3.1 req/s
conc=128 n=512 needle=502 ERROR=0 Counter({'stop': 502,'length': 10}) 6.8 req/s
```

Pass = `ERROR=0`, no hangs, throughput in this range. The `finish=length` tail at
conc=128 is the EOS/run-on mode of this harness (no chat template, EOS suppressed),
not a defect.

## 9. Prove kvd's read path (the part that is easy to fake)

First, the naive check — and watch it *fail* to prove anything:

```bash
docker exec kvaware_kvd_final python3 /tmp/prefix_reuse.py
docker exec kvaware_kvd_final python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

32/32 correct, and `gets_total` still **0**. The repeat was served by the in-GPU
radix cache. This is why a latency win is not evidence.

Now restart the engine — which empties the GPU cache while the kvd daemon and its
L3 survive — and replay:

```bash
bash scripts/kvaware_restart_replay.sh          # ~3 min for the leg to come back
docker exec kvaware_kvd_final python3 /tmp/prefix_reuse.py
docker exec kvaware_kvd_final python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

Expected: `gets_total` and `hits_total` climb (ours: 0 → **102**), `misses_total`
stays 0, and **`sets_total` does not change**. Reads with no new writes, on an
empty GPU cache, is kvd serving.

## 10. Collect

```bash
bash scripts/kvaware_collect.sh    # if you kept it; otherwise docker cp /tmp/final_results
```

## Teardown

```bash
for h in <node-A> <node-B>; do
  ssh $h 'docker rm -f kvaware_kvd_final kvaware_kvd_final_etcd 2>/dev/null'
done
```

Leave other people's containers alone; on a shared cluster verify a container is
yours with `docker inspect` (Binds / Created) before removing it.
