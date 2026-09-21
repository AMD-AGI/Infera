# Reproduction kit — GLM-5.2 1P1D AgentX T2 sweep (simulated acceptance)

Goal: reproduce the 4 timing points (CONC 40/56/72/96) of the t2f arm from a
clean machine with cluster access.
Estimated time: ~30 min bring-up (CUDA-graph capture) + ~1 h/point profiling
(3600 s each) + 4-14 min/point dataset setup ≈ **5-6 h for all four points**.

## 0. Prerequisites (arrange before you start)

- **Machines:** `crsuse2-m2m-135` (prefill + control + builder, data IP
  `10.245.148.209`) and `crsuse2-m2m-138` (decode, `10.245.157.237`). GPUs
  **2,3,4,5** on each. Obtained via the cluster's normal allocation.
- **Secrets (values NOT included — source them yourself):**
  - Hugging Face: AgentX pulls the trace corpus; if gated for your account,
    export `HF_TOKEN` on the control node. No other secret is needed.
  - Cluster SSH between 135↔138 preconfigured (the launcher ssh-hops).
- **External dependencies (absolute paths, not in repo):**
  - Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS).
  - Host GPUDirect provider: `/lib/x86_64-linux-gnu/libionic.so` (bind-mounted
    by the launcher).
  - AgentX trace corpus `semianalysisai/cc-traces-weka-062126`
    @ `23f152f6f0f9399a85901b89a6458def0ef16729` (393 traces), fetched
    automatically into `bench/glm5p2_pd/.cache/InferenceX`.
- **Repo state:** `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`,
  HEAD `c19bb2a8fbf70195b80cb2505960ea87f9d17d81`.
- **Image:** `infera-sglang:v0519-yihou-0917` (base, id `4190c3a99d0e`) present on
  BOTH nodes. This is the plain base — NOT the `-nextnfix-hicache` layer (see
  environment.md). If absent, build per that base's own recipe; nothing in this
  pack-up modifies the image.

## 1. Place the workspace configs and apply the engine.sh patch

The `config.yihou.*` files MUST sit exactly two levels under
`bench/glm5p2_pd/` — `config.yihou.base.sh` reaches the repo `config.sh` via
`../..`. Put them in the workspace dir:

    cd <repo>/bench/glm5p2_pd
    mkdir -p results/yihou-agentx-hicache
    cp <packup>/scripts/config.yihou.base.sh \
       <packup>/scripts/config.yihou.full.sh \
       <packup>/scripts/config.yihou.full.dcar.sh \
       <packup>/scripts/sweep.yihou.dcar.sh \
       <packup>/scripts/sweep.yihou.sh \
       <packup>/scripts/coherence_gate.yihou.sh \
       <packup>/scripts/topology.yihou.tsv \
       results/yihou-agentx-hicache/
    cp <packup>/scripts/watch.yihou.sh results/yihou-agentx-hicache/notes/watch.sh  # optional monitor

    # Apply the DECODE_EXTRA_ARGS escape hatch to engine.sh (uncommitted):
    git apply <packup>/patches/0001-engine-sh-extra-args-env.patch   # from <repo> root

Note the two FORCED substitutions already baked into `config.yihou.full.sh`
(the pinned nightly rejects `flydsl`/`aiter`): DSA backends resolve to
`tilelang` and the top-k backend to `sgl-kernel`.

## 2. Bring up the P4D4 stack (one bring-up serves the whole sweep)

    cd <repo>/bench/glm5p2_pd
    W=results/yihou-agentx-hicache
    ./launch.sh CONFIG=$W/config.yihou.full.dcar.sh \
                TOPOLOGY=$W/topology.yihou.tsv \
                OUT_DIR=$W/t2f-launch

Wait for "router ready" (~30 min; watch `$W/t2f-launch/server-logs/decode-0.log`).
`config.yihou.full.dcar.sh` sets `DECODE_EXTRA_ARGS=--disable-custom-all-reduce`
then sources `config.yihou.full.sh`; the resolved cell is: base image, HiCache on
(ratio 1.5), max-running 128, simulated acceptance 3.61, `index_share=false`,
custom-all-reduce OFF.

Verify what actually launched:

    grep -o 'disable-custom-all-reduce' $W/t2f-launch.log        # decode leg: 1
    grep -o 'SGLANG_SIMULATE_ACC_LEN=3.61' $W/t2f-launch.log      # present
    grep -o 'infera-sglang:v0519-yihou-0917' $W/t2f-launch.log    # base image

## 3. Run the sweep (4 points, ungated)

    OUT_PREFIX=t2f ./results/yihou-agentx-hicache/sweep.yihou.dcar.sh 40 56 72 96

`sweep.yihou.dcar.sh` runs `agentx_bench.sh` against the LIVE router once per
point into `$W/t2f-agentx-c<N>/`, with a 20 s late-result guard (the result JSON
can still be settling when the client container's chown-EXIT-trap fires). It does
NOT run a coherence gate — under simulated acceptance garbled text is expected by
construction, so a coherence gate is the wrong check for T2 (see notes.md). Do
NOT substitute `sweep.yihou.sh`: that points at `config.yihou.full.sh` (custom
all-reduce ON = the t2e arm), a different cell.

Optional monitor (records fault-count + /home free each 2 min, exits on fault or
sweep-end): `PREFIX=t2f INT=120 ./results/yihou-agentx-hicache/notes/watch.sh &`.
NB scan `docker logs`, not the on-disk follower, for faults (notes.md #7).

## 4. Parse the numbers

    python3 <packup>/scripts/make_summary.yihou.py $W > t2-summary.csv

Reads each `$W/t2f-agentx-c<N>/agentx_conc<N>.json` and emits the summary table.
The headline is total/per-GPU throughput, TTFT, ITL and the server cache-hit
collapse per point.

## 5. Teardown

    ./stop.sh CONFIG=$W/config.yihou.full.dcar.sh TOPOLOGY=$W/topology.yihou.tsv

It prints `cleanup completed with errors` even on a clean teardown. Verify no
`yihou` containers remain on either node and that GPUs 2-5 return to the ~298 MB
VRAM baseline (prefill's HiCache pool lags ~90 s — expected, not a leak).

## Expected output

`t2-summary.csv` matching `results/t2-summary.csv`: CONC=40 total ≈158 k tok/s,
per-GPU ≈19.8 k; throughput collapsing to ≈24 k at CONC=96; TTFT mean rising
7.4 → 418.6 s; server cache-hit falling 95% → 22%. Only CONC=40 is a usable
operating point.

## If it doesn't reproduce

See `notes.md` — CONC=128's two failure modes, the HiCache release lag, the
frozen-log monitoring trap, the forced-acceptance garble, and the config
placement / substitution gotchas.
