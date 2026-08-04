# REPRODUCE

Top-to-bottom. Assumes the PD deployment from
`../par8.glm52.dpaoff.packup_20260803` is **already serving** — this kit does not
bring it up. Total wall clock: ~5 min image build + ~35 min measurement.

```bash
# Constants used throughout
JUMP=root@149.28.124.225
W=/root/agentx_20260803          # workspace on the jump host
ROUTER=http://10.2.122.78:8100   # prefill node's router
SERVED=glm5.2-mxfp4
```

---

## 0. Prerequisites

- **The deployment must be up.** Verify both legs before spending a window:
  ```bash
  ssh $JUMP "curl -s $ROUTER/v1/workers" | python3 -m json.tool | grep -E 'disagg_mode|status'
  # expect: prefill/active and decode/active
  ```
- **Slurm holds belong to `yeandy-debug`. Never `scancel`.** This kit only sends
  HTTP traffic; it does not touch the nodes.
- Jump host needs: docker, outbound internet (pip + GitHub), `/mnt/vast` mounted.
- No secrets are required. No registry login, no API key — the router is
  unauthenticated on the internal network and both source repos are public.

## 1. Stage the customer's kit, unmodified

```bash
# from a clone of ROCm/MAD with PR #173 fetched:
#   git fetch origin pull/173/head:pr173
ssh $JUMP "mkdir -p $W/bench"
for f in README.md MANIFEST.md gen_caseA_conformance.py verify_caseA.py \
         replay_caseA.sh caseA_conformance_corpus.tar.gz; do
  git show "pr173:scripts/AgentX_CaseA/$f" | ssh $JUMP "cat > $W/bench/$f"
done
ssh $JUMP "md5sum $W/bench/replay_caseA.sh"
# expect 7cde1afc627c7e4868eac0fd13741baa  — this proves it is unmodified
```

Or copy from this kit: `spec/` holds the same files verbatim.

## 2. Materialize and verify the trace

```bash
ssh $JUMP "cd $W/bench && tar xzf caseA_conformance_corpus.tar.gz && \
           ls corpus/*.json | wc -l && python3 verify_caseA.py corpus"
# expect: 200
#         13/13 axes within band
```

The corpus is deterministic — `python3 gen_caseA_conformance.py corpus 200 42`
produces a byte-identical tree if you prefer to regenerate.

## 3. Stage the tokenizer into `$HERE`

**Required.** `replay_caseA.sh` mounts only `$HERE`, `/models`, and
`/shared_nfs` into the aiperf container — **not** `/mnt/vast`. Copying the
tokenizer files (20 MB, no weights) into the script's own directory keeps the
script unmodified.

```bash
ssh $JUMP "mkdir -p $W/bench/tokenizer && cd /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 && \
  cp tokenizer.json tokenizer_config.json config.json generation_config.json \
     chat_template.jinja $W/bench/tokenizer/"
```

## 4. Build the aiperf image

Upstream's own Dockerfile ends in `FROM nvcr.io/nvidia/distroless/python`, which
needs NGC auth. `scripts/build_aiperf_img.sh` builds an equivalent CLI image from
plain `python:3.13-slim` instead.

```bash
scp scripts/build_aiperf_img.sh $JUMP:$W/
ssh $JUMP "cd $W && bash build_aiperf_img.sh"     # ~4 min
# expect: === built aiperf-agentx:v1.0 ===
#         0.8.0
```

Verify the scenario and loader registered:

```bash
ssh $JUMP 'docker run --rm --entrypoint python aiperf-agentx:v1.0 -c "
from aiperf.common.scenario.registry import get_scenario
print(get_scenario(\"inferencex-agentx-mvp\").min_benchmark_duration_seconds)
import aiperf.dataset.loader.weka_trace"'
# expect: 900
```

## 5. Start the artifact rescue loop — do this BEFORE the run

`replay_caseA.sh` writes `--output-artifact-dir` under `$OUT`, but only mounts
`$HERE`. With `OUT` outside `$HERE` the results land **inside the container** and
die with it — the sweep then reports `FAILED` for a run that succeeded. Rather
than patch the customer's script, poll and `docker cp` the artifacts out.

```bash
scp scripts/rescue_artifacts.sh $JUMP:$W/
ssh $JUMP "cd $W && nohup bash rescue_artifacts.sh > rescue.log 2>&1 &"
```

> Alternative: set `OUT=$W/bench/results` (inside `$HERE`) and skip this step.
> We did not, because we wanted the failure documented rather than avoided.

## 6. Run the sweep

```bash
scp scripts/run_caseA.sh $JUMP:$W/
ssh $JUMP "cd $W && nohup bash run_caseA.sh > run_caseA.log 2>&1 &"
```

`run_caseA.sh` sets only environment and then calls the customer's script:

| var | value | why it differs from the script's default |
|---|---|---|
| `URL` | `http://10.2.122.78:8100` | our router |
| `SERVED` | `glm5.2-mxfp4` | our served-model-name; default `GLM-5.2-MXFP4` is wrong for us |
| `TOK` | `$HERE/tokenizer` | the container cannot see `/mnt/vast` |
| `CONCS` | `8 16` | operator decision; **conc=1 is unsupported by the scenario** |
| `DUR` | `900` | the scenario's **enforced minimum**; the script's default 300 is rejected |
| `IMG` | `aiperf-agentx:v1.0` | default `rocm/atom-dev:latest` lacks the aiperf fork |

Runtime: ~17 min per concurrency point (66 s warmup + 900 s profiling + teardown).

Watch it:

```bash
ssh $JUMP "grep -iE 'Phase warmup complete|Phase profiling sending complete' \
  $W/rescue/c8_art/logs/aiperf.log"
# expect: warmup complete | completed=8, cancelled=0, errors=0
#         profiling sending complete | sent=233, completed=230
```

## 7. Stop the rescue loop and analyse

```bash
ssh $JUMP 'pkill -f "rescue_[a]rtifacts"'   # bracket avoids self-matching the ssh cmdline
scp scripts/analyze.py $JUMP:$W/
ssh $JUMP "cd $W && python3 analyze.py rescue/c8_art rescue/c16_art"
```

Expected headline (this run):

```
c8 : 231 profiling reqs, 901.2 s, TTFT p50 5,146 ms, ITL p50 13.81 ms, cache 88.1 %
c16: 323 profiling reqs, 921.8 s, TTFT p50 14,394 ms, ITL p50 14.68 ms, cache 88.1 %
```

## 8. Archive

```bash
ssh $JUMP "cd $W && for t in c8 c16; do
  gzip -c rescue/\${t}_art/profile_export.jsonl > /tmp/\${t}_profile_export.jsonl.gz
  gzip -c rescue/\${t}_art/logs/aiperf.log      > /tmp/\${t}_aiperf.log.gz
done"
# then scp into results/c{8,16}/ and logs/
```

---

## External dependencies (not in this kit)

| what | where | note |
|---|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST, visible from jump host and both nodes |
| customer bench | `github.com/ROCm/MAD` PR **#173** | also copied verbatim into `spec/` |
| aiperf fork | `github.com/SemiAnalysisAI/aiperf` @ `cquil11/aiperf-agentx-v1.0` | public; pinned by branch, **not** by SHA — see `notes.md` |
| the deployment | chi2835 + chi2879, image `infera/engine-sglang:merged-e` | brought up by `../par8.glm52.dpaoff.packup_20260803/REPRODUCE.md` |

## Secrets

**None.** No registry login, no API key, no cluster credential beyond the SSH
access already required to reach the jump host.
