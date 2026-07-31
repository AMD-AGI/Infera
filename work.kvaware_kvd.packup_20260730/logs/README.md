# Raw engine logs

One directory per experiment, both PD legs each. Copied verbatim off the shared
FS (`/mnt/vast/c_huggingface/glm52_kvexp/` — visible from both nodes, which is
why there is one copy, not one per host). All files are well under 4 MB so none
are gzipped.

| Directory | Experiment | Files | Result file |
|---|---|---|---|
| `glm_baseline/` | GLM-5.2 2-node PD+DPA, **kvaware OFF / kvd OFF** | `pd_prefill_base.log`, `pd_decode_base.log` | `results/baseline_probe_4of4.txt` |
| `glm_step1_kvaware_kvd/` | same, **kvaware ON / kvd ON** | `pd_prefill_kv.log`, `pd_decode_kv.log` | `results/step1_kvaware_kvd_4of4.txt` |
| `glm_step3_restart/` | engine restarted against a **warm kvd** | `pd_prefill_r2.log`, `pd_decode_r2.log` | `results/step3_restart_reload.txt` |
| `glm_step4_routing/` | **2× TP4 decode** workers (routing test) | `pd_decodeA.log`, `pd_decodeB.log` | `results/step4_role_weights_routing.txt` |

Total 2.2 MB.

## Grep recipes — each claim, straight from the log

Every number in `results/` can be re-derived from these files. The commands
below are the exact ones used, so a reviewer can audit rather than trust.

**The baseline really had the switches off, and really used RDMA:**

```bash
grep -ac 'kv-events-config'      glm_baseline/pd_prefill_base.log   # 0  -> kvaware off
grep -ac 'MC_FORCE_TCP'          glm_baseline/pd_prefill_base.log   # 0  -> not the TCP fallback
grep -ac 'HIP dmabuf disabled'   glm_baseline/pd_prefill_base.log   # 8  -> mooncake RDMA on all 8 ionic rails
grep -ac 'ready to roll'         glm_baseline/pd_prefill_base.log   # 1
```

**Step 1 really wired kvd, on every DP rank:**

```bash
grep -ac 'infera-kvd adapter connected'  glm_step1_kvaware_kvd/pd_prefill_kv.log   # 8 (one per DP rank)
grep -ac 'KV plane up:'                  glm_step1_kvaware_kvd/pd_prefill_kv.log   # 1
grep -a  'Allocating .* hierarchical'    glm_step1_kvaware_kvd/pd_prefill_kv.log   # 8x "16.00 GB"
grep -ac 'disaggregation-decode-enable-radix-cache' glm_step1_kvaware_kvd/pd_decode_kv.log  # 1 (infera auto-appended)
```

**PD + DP-attention were symmetric on both legs** (asymmetric DPA would mismatch
the KV shard layout across the mooncake transfer):

```bash
for f in glm_step1_kvaware_kvd/*.log; do
  echo "$f:"; grep -aoE "disaggregation_mode='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8" "$f" | sort -u
done
```

**The randomised port fix (patch 0001) held** — each leg got its own base
instead of both landing on 32764:

```bash
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' glm_step1_kvaware_kvd/pd_prefill_kv.log
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' glm_step1_kvaware_kvd/pd_decode_kv.log
```

## What these logs do NOT contain

- **The Qwen3-1.7B MVP rounds (r1–r5).** That container was removed before this
  step, so its logs are gone. Key excerpts were captured in-session and live in
  `results/kvaware_kvd_activation_evidence.txt`. This is a real gap, recorded
  rather than papered over.
- **kvd daemon logs** (`/tmp/kvd.log`) — inside the containers, which were
  removed at teardown. The daemon's counters, which is what the conclusions rest
  on, were read via `scripts/kvdstats.sh` and are quoted verbatim in the step
  2/3 result files.
- **Router logs** (`/tmp/router.log`) — same, container-local. The one line that
  matters (`router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0`) is
  quoted in `results/step4_role_weights_routing.txt`.
- **decodeB's first, failed start** in step 4 — the run that died on the
  `--kv-snapshot-port` 8801 collision. `pd_decodeB.log` here is the *successful*
  relaunch (port 8802); the failed attempt's log was overwritten by it. The
  traceback is quoted in `results/step4_role_weights_routing.txt` and
  `notes.md`.
