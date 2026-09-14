# Testing the kv-aware short-prompt fix on the amd-mi355 cluster

Verifies #107 (v0.2.9): prompts too short to hash no longer pin the whole fleet
onto one worker.

`kubectl` on this host is a symlink to `k3s`, which reads
`/etc/rancher/k3s/k3s.yaml` and ignores `~/.kube/config`. Every command below
needs the kubeconfig named explicitly:

```bash
export KUBECONFIG=~/.kube/config
kubectl config current-context      # expect: amd-mi355-cluster
```

## 0. Free the GPUs

The cluster has 16 GPUs across `nimble-orca` and `solid-lynx`. A TP8 mixed
worker takes a whole node, so two workers need all 16 — nothing else can be
holding any.

The PD deployment in `infera` is managed by an InferaDeployment CR, so
`kubectl scale` does not stick: the operator reconciles the replica count
straight back. Suspend the CR instead of deleting it — deleting is not
reversible without the original manifest.

```bash
# back it up first, whatever you do next
kubectl get inferadeployment -n infera kimi-k3-opt-pd-dspark -o yaml \
  > /tmp/pd-dspark-backup.yaml

kubectl delete inferadeployment -n infera kimi-k3-opt-pd-dspark
# restore later with: kubectl apply -f /tmp/pd-dspark-backup.yaml
```

Confirm nothing holds a GPU:

```bash
kubectl get pods -A -o json | python3 -c "
import json,sys
for p in json.load(sys.stdin)['items']:
    n=p['spec'].get('nodeName','')
    if n not in ('nimble-orca','solid-lynx'): continue
    g=sum(int(c.get('resources',{}).get('limits',{}).get('amd.com/gpu',0))
          for c in p['spec']['containers'])
    if g: print(n, p['metadata']['namespace']+'/'+p['metadata']['name'], g)
"
```

## 1. Point both deployments at v0.2.9

`kimi-k3-infera` (workers) and `infera` (router) are plain Deployments with no
owner, so they can be edited directly. Both pull the overlay in an
initContainer named `infera-overlay`.

The router is what carries the policy, but update both so the payload version
is unambiguous when reading logs later.

```bash
OV=inferaimage/infera-overlay@sha256:bdb6cb5a0f340ad9a53a3ced890ca1e462f7d0e58834aa24f611d388edb68fd7

kubectl set image deploy/infera          infera-overlay=$OV
kubectl set image deploy/kimi-k3-infera  infera-overlay=$OV
kubectl scale  deploy/kimi-k3-infera --replicas=2
```

Pinning by digest, not by `:v0.2.9`, so a retag cannot silently change what is
under test.

## 2. Wait for both workers

1.5T of weights: roughly 10 minutes per worker, longer on a cold page cache.

```bash
kubectl get pods -o wide -w | grep kimi-k3-infera
```

Both must reach `1/1 Running`, on **different** nodes. Then confirm the router
found them and subscribed to their event streams:

```bash
kubectl logs deploy/infera --tail=50 | grep -E "worker .* kv=|subscrib"
```

Expect two lines with `kv=yes` and a `tcp://…` endpoint each. If a worker shows
`kv=no`, its events are not flowing and every cache lookup will miss — fix that
before reading any routing result.

## 3. Confirm the payload actually carries the fix

Worth 10 seconds, and it rules out the most embarrassing failure mode:

```bash
kubectl exec deploy/infera -c server -- \
  grep -c _UNKNOWN_COST_BLOCKS /overlay/py312/infera/router/policy/kv_event_aware.py
```

Expect `2`. Anything else — including a non-zero exit, which is what `grep -c`
does on no match — means the pod is still on the old overlay.

The container is `server` on the router and `main` on the workers.

## 4. Run the test

From a pod inside the cluster, since the services are ClusterIP:

```bash
kubectl run kv-test --rm -it --restart=Never --image=python:3.12-slim -- \
  bash -c 'pip install -q httpx && python3 - <<PY
import asyncio, httpx
from collections import Counter

ROUTER = "http://infera:8000"

async def arm(name, content_fn, n=60):
    picks = Counter()
    async with httpx.AsyncClient(timeout=300.0) as c:
        for i in range(n):
            r = await c.post(f"{ROUTER}/v1/chat/completions", json={
                "model": "moonshotai/Kimi-K3",
                "messages": content_fn(i),
                "max_tokens": 8, "temperature": 0.0})
            if r.status_code != 200:
                print("HTTP", r.status_code, r.text[:200]); return
    print(f"{name}: {n} requests done")

# SHORT: under the 768-token index block size -> hashes to zero blocks.
# This is the case #107 fixes, and the shape most smoke tests have.
short = lambda i: [{"role":"user","content":f"Explain topic {i%3} briefly."}]

# LONG: several blocks, so the cache term actually participates.
pre = "Review guideline: check correctness, then performance.\n" * 300
long = lambda i: [{"role":"system","content":pre},
                  {"role":"user","content":f"Explain topic {i%2} in detail. "*80}]

asyncio.run(arm("short", short))
asyncio.run(arm("long", long))
PY'
```

## 5. Read the result from the router's own decisions

```bash
kubectl logs deploy/infera --tail=400 | grep "pick policy=kv-aware" \
  | sed 's/.*picked=\([^ ]*\).*request_blocks=\([0-9]*\).*/\1 blocks=\2/' \
  | sort | uniq -c
```

What each outcome means:

| observation | reading |
|---|---|
| `blocks=0` picks split across both workers | **#107 working** — this is the fix |
| `blocks=0` picks all on one worker | fix not active; check step 3 |
| `blocks>0` picks follow cache, hit rate climbs | cache locality healthy |
| every pick `blocks=0`, even long prompts | prompts under 768 tokens, or tokenisation failing |

For the traffic split as the engines themselves saw it — which is how the
original report measured it, and does not trust the router to report on itself:

```bash
for p in $(kubectl get pods -l app.kubernetes.io/name=kimi-k3-infera \
             -o name | cut -d/ -f2); do
  echo -n "$p "
  kubectl exec $p -c main -- \
    sh -c 'curl -s localhost:8000/metrics | grep "^vllm:prefix_cache_queries_total"'
done
```

Take the delta across a run, not the absolute value.

## What this does and does not show

#107 makes short prompts **spread** instead of piling onto one worker. It does
not make them route *by cache* — a prompt carrying no block information has
nothing to route on. To see `--kv-overlap-weight` change placement at all, the
prompts must exceed the index block size; that is what the long arm is for.

## Restore

```bash
kubectl apply -f /tmp/pd-dspark-backup.yaml    # bring the PD deployment back
kubectl scale deploy/kimi-k3-infera --replicas=0
```
