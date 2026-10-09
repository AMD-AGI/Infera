# Cold reproduction — synthetic expert8 PD

Run date2026-09-23; packaged2026-09-24. Allow30minutes for cold JIT/graphs plus smoke and approximately3minutes measured load. No quality equivalence with GLM-5.2 is claimed.

## 0. Prerequisites
- Authorized access to138(prefill10.245.157.237) and136(decode10.245.154.168), GPUs0–3 free on each.8×MI355X installed per host; only4 used per leg. Do not use137. Direct SSH and node-local Docker were used; no Spur allocation was made for these nodes.
- The NEW workspace path must be shared and identically accessible on login/138/136. The scripts use fixed node/port/container names; do not launch beside an existing expert8 deployment. Inspect first and arrange availability without stopping another user's work.
- Original checkpoint `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` must be mounted read-only. No weights included. No external dataset: seed42 random input IDs.
- Host provider `/lib/x86_64-linux-gnu/libionic.so` on both nodes. See environment.md for hash and driver versions.
- Credentials: normal cluster SSH/Docker authorization. No model/API/HF token needed for this local-checkpoint synthetic test. Vendor-image rebuild may require registry access; no credentials included.

## 1. Obtain the exact image
Required tag `infera-sglang:v0519-yihou-0917-nextnfix-hicache`; immutable LOCAL image ID:
`sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`.

It has no registry RepoDigest. Do not use this ID in `docker pull`. Both hosts held this image at packaging time. Verify on each:
```bash
for N in crsuse2-m2m-138 crsuse2-m2m-136; do
  ssh -o ClearAllForwardings=yes "$N" \
    'docker image inspect --format "{{.Id}}" infera-sglang:v0519-yihou-0917-nextnfix-hicache'
done
```
If a destination lacks it, first restore the new workspace using step2 (which does not need Docker), then export from an authorized verified source into `$W` and load on the destination. The large image archive is NOT included in this kit:
```bash
ssh -o ClearAllForwardings=yes crsuse2-m2m-136 \
  'docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache' > "$W/yihou-image.tar"
ssh -o ClearAllForwardings=yes <destination> 'docker load' < "$W/yihou-image.tar"
```
Re-check ID. Historical Dockerfiles under patches/image-provenance document construction, not a bit-identical rebuild promise. They contain original cleanup operations and are archival only under the current deletion rule.

## 2. Restore to a NEW workspace
Preserved scripts/source are verbatim in `payload/`. Do not run them there: some hardcode the old workspace. The restoration helper substitutes only that absolute workspace path and refuses any existing destination.
```bash
P=/absolute/path/to/yihou-expert8-pd.packup_20260924
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-reproduce-20260924
python3 "$P/scripts/restore_workspace.yihou.py" --workspace "$W"
```
Create the lightweight model overlay inside Docker on138, with the shared source read-only:
```bash
ssh -o ClearAllForwardings=yes crsuse2-m2m-138 \
  "docker run --name yihou-expert8-prepare-20260924 \
   -v '$W:$W' \
   -v /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:ro \
   -w '$W' -e PYTHONDONTWRITEBYTECODE=1 \
   infera-sglang:v0519-yihou-0917-nextnfix-hicache \
   python3 '$W/scripts/prepare_model.yihou.py'"
```
Use a fresh preparation-container name if it already exists; never overwrite the original model. The new overlay copies config/metadata and symlinks shards. Startup mounts both source and overlay. `patches/mounts.yihou.tsv` binds patched files into the exact image; do not separately apply the patch to shared SGLang.

## 3. Verify free GPUs and run transfer gate
Inspect GPU processes/memory and container names on both hosts. Baseline is approximately284MiB/card, no compute processes on selected GPUs;136's zero-VRAM gpuagent is expected. Existing running models mean the GPUs are NOT free even if utilization is0.
```bash
for N in crsuse2-m2m-138 crsuse2-m2m-136; do
  ssh -o ClearAllForwardings=yes "$N" 'rocm-smi --showpids --showmeminfo vram; docker ps'
done
R="$W/rounds/yihou-reproduction"
python3 "$W/scripts/rdma_probe.yihou.py" --out "$R/rdma"
```
Require8 directed results, every verified=true and both exit codes0. The launcher also checks archived reference evidence; that check does NOT replace this fresh transfer test. Probe containers use task names; if stale containers point to another workspace, choose fresh names in the restored script rather than reusing them. Never delete unrelated containers.

## 4. Launch and smoke
```bash
bash "$W/scripts/launch.yihou.sh" "$R" > "$W/yihou-launch.log" 2>&1
bash "$W/scripts/smoke.yihou.sh" "$R"
```
Wait for prefill/decode/router healthy. CUDA graph/JIT may take30minutes; check live logs/build progress before declaring a hang. `TMPDIR` must remain workspace-scoped **container tmpfs**, not shared NFS. Keep TP4/DP4/DPA/EP4, shared fusion off, MTP/HiCache off, triton and FP8 KV. Keep per-GPU ionic0–3 mapping and rank affinity.

## 5. One fixed-length C32 point
```bash
bash "$W/scripts/bench_fixed.yihou.sh" "$R/benchmark"
python3 "$W/scripts/collect.yihou.py" --out "$R/after-benchmark"
```
Defaults are32 warmup+256measured,4096input/1024output,concurrency32. Require summary pass=true, completed256, failures0, peak_client_inflight32 and measured lengths correct for every request. Client exit code is saved beside the output directory. Missing server usage is failure, not inferred success.

## Expected results and evidence inspection
Reference: output1424.02tok/s, total7120.09tok/s, meanTTFT1.126s, meanTPOT20.87ms, measured184.09s. Operational pass is primary; no narrow throughput tolerance is imposed across reruns.

Archived results can be inspected without nodes:
```bash
python3 - "$P" <<'PY'
import gzip,json,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[json.loads(x) for x in gzip.open(p/'results/benchmark/requests.jsonl.gz','rt')]
assert len(rows)==288 and all(r['ok'] for r in rows)
assert all((r['usage']['prompt_tokens'],r['usage']['completion_tokens'])==(4096,1024) for r in rows)
print(json.loads((p/'results/benchmark/summary.json').read_text()))
PY
```

## Teardown is a separate operator action
Packaging did not stop the original deployment. When YOUR reproduction is done, deliberately run the restored task-scoped `scripts/bench-harness/stop.sh CONFIG=$W/scripts/config.yihou.expert8.sh`, then verify memory returns to baseline. Its absent-container checks may report errors; inspect actual state. Preserve logs. No cleanup is performed automatically by this packup.
