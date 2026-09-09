# Reproduce

Expect ~15 min of setup and ~3 h for the campaign. The loop is time-driven and
runs until `--max-hours` is spent.

## 1. Container and scratch, both on local disk

```sh
export SCRATCH=/tmp/yihou/kf_repro
export REPO=$HOME/dev/git.16-19/KernelForge
mkdir -p $SCRATCH/{runs,logs,knowledge,claude_cfg,triton_cache}
docker run -d --name yihou-kf-repro -v "$HOME:$HOME" -v "$SCRATCH:$SCRATCH" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --network=host --shm-size 64g --security-opt seccomp=unconfined \
  -e HOME="$HOME" -w "$REPO" lmsysorg/sglang:v0.5.14-rocm720-mi30x sleep infinity
```

## 2. Install, including the profiling dependencies

```sh
docker exec yihou-kf-repro bash -lc 'pip install -e ".[dev]" &&
  pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt'
docker exec yihou-kf-repro bash -lc 'kernel-agents status'
```

`kernel-agents status` must print `GPU target: gfx942` and
`rocprof-compute: ready`. If it says `dependencies are not ready`, hardware
profiling degrades silently and the analysis is worth much less.

## 3. Check the driver against the profile before spending anything

```sh
docker exec yihou-kf-repro bash -lc 'cd <workset>/kernel &&
  HIP_VISIBLE_DEVICES=0 python3 driver.py &&
  HIP_VISIBLE_DEVICES=0 python3 measure_baseline.py'
```

Expect `B8_V151936` near **55.4 µs**, which must agree with the 55.59 µs/call in
the profile. If it does not, stop: the driver is not measuring the traced
kernel.

## 4. Run the campaign

```sh
docker exec -d yihou-kf-repro bash -lc "
  export AMD_LLM_GATEWAY_SUBSCRIPTION_KEY=<your key>
  HIP_VISIBLE_DEVICES=0 scripts/run_forge.sh $SCRATCH/runs/repro > $SCRATCH/logs/forge.log 2>&1"
tail -F $SCRATCH/logs/forge.log
```

`--max-hours` must be **> 2.0**. At or below it, KernelForge silently drops to
static-only analysis and a turn cap of 100.

## 5. Re-measure, and do not trust the campaign's own number

```sh
docker exec yihou-kf-repro bash -lc 'cd <a copy of the workset kernel dir> &&
  cp <the optimized kernel> sampler_softmax_kernel.py &&
  HIP_VISIBLE_DEVICES=0 python3 measure_baseline.py --json optimized.json'
```

## Expected output

- `results/forge_result.json` with `improved: true` and
  `checkpoint.validation_passed: true`.
- SNR **> 100 dB** (measured 138.12).
- A mean case speedup in the range **2.4–2.9×**. Do not expect the exact
  number: the agent's search path is stochastic, and the optimized side's
  round-to-round spread here is ~20%.
- What *is* reproducible is the magnitude and the root cause — a launch-geometry
  limit, 8 workgroups on 304 CUs — not the literal constant `1216`.
