# Reproduction kit — 07 PD mooncake RDMA + DP-attention sweep

Goal: bring up GLM-5.2 mooncake-RDMA PD with **DP-attention on both legs**, verify 4/4 correctness,
then sweep conc 64→2048. Est. ~15 min bring-up + ~40-70 min sweep. **Builds directly on 03** (same
image, container prep, libionic inject, transport) — only the leg launcher and the sweep differ.

## 0. Prerequisites (identical to 03)

- `infera/engine-sglang:pd-unified` on **both** chi2878 + chi2879 (see 03/REPRODUCE §0 for the
  `docker save | ssh <dst> docker load` distribution — it is a local build, not on a registry).
- Model `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST mount).
- Host `libionic` injected into the container (the `up_dpa.sh` prep step does this).
- Fabric pre-check: all 8 ionic NICs `PORT_ACTIVE`, routable GID at index 1.

## 1. Stage scripts to the shared kit dir

    KIT=/mnt/vast/c_huggingface/glm52_p2b        # shared VAST, visible to both nodes' containers
    # from the repo, stream each file through the jump host:
    for f in scripts/pd_leg_dpa.sh scripts/up_dpa.sh scripts/sweep_dpa.sh scripts/probe.py; do
      cat "$f" | ssh <jump> "cat > $KIT/$(basename $f)"
    done

## 2. Bring up both DPA legs + router (one command)

`up_dpa.sh` preps both containers (libionic inject), stages `pd_leg_dpa.sh`, and launches both legs
with `DPA=1`. **Launch the legs with `docker exec -d $CTR env VAR=... bash /script`** — NOT
`bash -lc '...'` (the login-shell detached form does not persist; this bit us — see notes §gotcha).

    bash scripts/up_dpa.sh          # prefill=chi2878:30000, decode=chi2879:30001, DP cold start ~5-8 min

Wait until **both** legs print `ready to roll`:

    ssh chi2878 'grep -c "ready to roll" '$KIT'/pd_prefill_30000_dpa.log'   # -> 1
    ssh chi2879 'grep -c "ready to roll" '$KIT'/pd_decode_30001_dpa.log'    # -> 1

Confirm the DP flags actually took (both legs):

    grep -oE 'enable_dp_attention=True|dp_size=8|ep_size=8' <leg log> | sort -u   # all three present

Then start the router in the prefill container (write it to a script file to avoid `-lc` detach loss):

    ssh chi2878 'docker exec pd_uni bash -c "cat > /run_router.sh <<EOF
    #!/bin/bash
    pkill -9 -f sglang_router 2>/dev/null; sleep 2
    exec python3 -m sglang_router.launch_router --pd-disaggregation \
      --prefill http://10.2.122.3:30000 8998 --decode http://10.2.122.10:30001 \
      --host 0.0.0.0 --port 8002 > /tmp/router.log 2>&1
    EOF
    chmod +x /run_router.sh"'
    ssh chi2878 'docker exec -d pd_uni bash /run_router.sh'
    sleep 16

## 3. Correctness (before any stress)

    ssh chi2878 'docker exec pd_uni python3 /tmp/probe.py http://10.2.122.3:8002 glm5.2-mxfp4'
    # want 4/4 (paris / beijing / 4 / jupiter)

Confirm RDMA (not TCP):

    ssh chi2878 'grep -c "HIP dmabuf disabled" '$KIT'/pd_prefill_30000_dpa.log'   # ~200 (8 NICs × warmups)
    ssh chi2878 'grep -ic "MC_FORCE_TCP" '$KIT'/pd_prefill_30000_dpa.log'          # 0

## 4. Run the sweep

    ssh chi2878 'docker cp '$KIT'/sweep_dpa.sh pd_uni:/sweep_dpa.sh'
    ssh chi2878 'docker exec -d pd_uni env CONCS="64 128 256 512 1024 2048" \
       BASE=http://10.2.122.3:8002 MODEL=glm5.2-mxfp4 \
       TOK=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4 ISL=1024 OSL=1024 \
       OUTDIR='$KIT'/sweep_dpa TAG=dpa bash /sweep_dpa.sh > '$KIT'/sweep_master.log 2>&1'

Each point writes `sweep_dpa/dpa_c<N>.jsonl` (+ `.log`). Poll for 6 jsonl files, or watch
`sweep_master.log`. The sweep uses `num_prompts = min(4×conc, 4096)`, `warmup = min(conc, 64)`.

## 5. Parse

    # in-container or anywhere with the jsonl:
    python3 - <<'PY'
    import json,glob
    for f in sorted(glob.glob('sweep_dpa/dpa_c*.jsonl'), key=lambda x:int(x.split('_c')[-1].split('.')[0])):
        d=[json.loads(l) for l in open(f)][-1]
        print(f.split('_c')[-1].split('.')[0], d['completed'], round(d['output_throughput']),
              round(d['median_tpot_ms'],1), round(d['median_ttft_ms']))
    PY

Expected: all `completed` full (256/512/1024/2048/4096/4096), out tok/s peaking ~12855 at conc=1024.
See `results/sweep_table.txt`.

## If it doesn't reproduce

- **Legs don't start / no log file** → you used `bash -lc '...'` with `-d`; use `env … bash /script`
  (notes §gotcha).
- **Probe connection refused** → router didn't persist; use the `/run_router.sh` file form above.
- **conc≥256 throughput not scaling** → DP flags didn't take; re-check step 2's `enable_dp_attention=True`.
- **decode OOM at high conc** → lower decode `--mem-fraction-static` (0.85→0.80) or `--context-length`.
