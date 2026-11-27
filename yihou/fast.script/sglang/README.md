# SGLang fake-PD decode benchmark

Run `./run.sh --help` for the model path, run directory, and workload options.
The default image is `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`.
Use four allocated MI355X GPUs and local DeepSeek-V4.1-Flash weights for the
reference TP4/DPA4/EP4 configuration. The launcher starts one engine in
decode mode and sends streaming `/generate` requests directly to it.
The launcher forwards `HSA_ENABLE_SDMA` when it is set in the environment.

The fake KV receiver allocates prompt-length KV on the decode worker and
completes the PD handoff locally. The first output token is a synthetic handoff
token. Subsequent tokens execute the model's decode path and use synthetic
prompt cache contents. Interpret the run as decode execution at the requested
context lengths. Output text, prefill throughput, and PD transfer latency are
outside this measurement.

- The client distributes bootstrap rooms across attention-DP ranks. A constant
  bootstrap room, as used by the native serving client's `--fake-prefill`
  switch on the pinned revision, sends all requests to the same rank with
  `follow_bootstrap_room`.
- The server disables decode prefix reuse and reserves zero extra SWA prefix
  tails. The default uses SGLang's native fake PD path with speculation off.
- The largest requested concurrency sets the server's max-running capacity
  rounded up to a multiple of the DP rank count and the CUDA graph batch limit
  per rank. The server stays up across the sweep.
- Use a context limit at least two tokens above input plus requested output
  length to account for the SGLang scheduler's length clamp.

## Optional DSpark compatibility

Set `SGLANG_DSPARK=1` to enable DSpark with a block size of 5.
The pinned SGLang image's DSV4.1 validator permits DSpark PD with Mooncake,
NIXL, or MoRI. The launch adapter adds the native fake backend to this one
predicate for decode only. The engine retains its static verify and CP/DCP
checks. The adapter checks the installed validator source before applying the
change in memory.

Simulated acceptance defaults to 3.51 tokens per verification step.
Set `SGLANG_SIMULATE_ACC_LEN=-1` to disable acceptance simulation while keeping
DSpark. On MI355X the adapter passed startup validation. Request-level GPU
validation for this mode is pending. The native preset completed the sweep
on four MI355X GPUs with speculation disabled.
