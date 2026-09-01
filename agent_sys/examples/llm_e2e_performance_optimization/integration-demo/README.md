# `integration-demo` — accept or reject an optimised kernel on the real service

The integration stage of `llm_e2e_performance_optimization`, and the last one. It
takes the kernel-optimization stage's deliverable, puts it in front of a real
GLM-5.3-Flash deployment on one MI355X node, and decides whether it broke or
slowed anything.

`DESIGN.md` is the design and the record of what was decided and why.
`temp/manual/` holds the hand-run measurements that settled the defaults here,
including the ones that changed the design. `temp/ARTIFACTS.md` is the inventory:
what this pipeline produced, where each piece lives and how long it survives,
what each of the ten handoffs carries and why (section 3), and the exact shape of
the mocked KernelForge deliverable (section 7) — which is the contract the
kernel-optimization stage has to satisfy to replace it.

## What it does

Two deployments, in one session, on one node, differing in **exactly one thing**:
whether the patched files are bind-mounted over their paths inside the image.
Each arm is measured by the same script in the same order.

```
main                     non-leaf, no agent
 ├── seed_patch       → kernel_patch          check_patch_shape       completeness/strong
 ├── apply_patch      → patch_overlay         check_overlay_applies   completeness/strong
 ├── serve_stock      → deployment_stock      check_service_live      completeness/strong
 ├── measure_stock    → acceptance_stock      check_acceptance        completeness/strong
 │                    → bench_stock           check_bench_report      completeness/strong
 ├── serve_patched    → deployment_patched    check_service_live
 │                                            check_patch_live        trustworthiness/strong
 ├── measure_patched  → acceptance_patched    check_acceptance
 │                    → bench_patched         check_bench_report
 ├── compare          → integration_report    check_no_regression     usability/strong
 └── packup           → integration_packup    check_packup_shape      completeness/strong   is_end
```

Five measurements per arm, in this order and recorded with timestamps:

1. **smoke** — one `mixed` worker, the served model name, `17 * 23 = 391`, and a
   long generation with no 8-gram repeated more than four times.
2. **needle** — a lexical passphrase at three depths of a multi-chunk prompt, at
   two lengths. Prompt length is read back from `usage.prompt_tokens`. Read as a
   regression detector, not a capability gate — see below.
3. **probe** — reachability, answerability, stability under repeats and
   concurrency, and that a long shared prefix does not change the answer. It
   gates the eval's *interpretation*, not its execution: a failed probe makes
   every eval comparison `uninterpretable` and a reason to reject, rather than
   skipping the eval and stranding the rest of the run.
4. **llm-eval** — `sglang.test.run_eval` over `gsm8k` and `mixed_prefix_gsm8k`.
5. **trace replay** — AIPerf against a Mooncake production trace, two rounds: the
   first cold for the trace, the second warm.

## Status

**Both arms run, all ten handoffs produced, all eight validators PASS.** Measured
on `smci355-ccs-aus-n04-33`, 2026-08-31, run `20260831T185620-cd3305`:

```
seed_patch        succeeded   kernel_patch        valid   check_patch_shape      PASS
apply_patch       succeeded   patch_overlay       valid   check_overlay_applies  PASS
serve_stock       succeeded   deployment_stock    valid   check_service_live     PASS
measure_stock     succeeded   acceptance_stock    valid   check_acceptance       PASS
                              bench_stock         valid   check_bench_report     PASS
serve_patched     succeeded   deployment_patched  valid   check_patch_live       PASS
                                                          check_service_live     PASS
measure_patched   succeeded   acceptance_patched  valid   check_acceptance       PASS
                              bench_patched       valid   check_bench_report     PASS
compare           succeeded   integration_report  valid   check_no_regression    PASS
packup            (see below) integration_packup          check_packup_shape     PASS
```

`packup` wrote its whole deliverable and was cut off before the seal by the
framework's fixed 1800-second settle budget (`../temp/bugs/003`); its content was
validated by running `check_packup_shape` against it directly. Everything above
it sealed inside the graph.

**The verdict was ACCEPTED, which is the right answer**, because the patch under
test is the semantics-preserving mock: it adds two log lines and changes no
arithmetic, so a comparison that reported a regression would have a fault in the
judgement rather than in the thing judged. Every line came back `same`:

```
smoke      arithmetic, long_generation, workers, models, engine_log     all same
needle     three depths at 31k tokens, both arms retrieved              all same
gsm8k                0.95 -> 0.90   95% CI of the difference [-0.26, +0.15]  same
mixed_prefix_gsm8k   1.00 -> 1.00                            [-0.16, +0.16]  same
throughput          762.28 -> 770.67 tok/s     +1.1%                         same
TTFT (avg)         3389.31 -> 3465.58 ms       +2.3%                         same
inter-token latency 151.59 -> 143.40 ms        -5.4%                         same
requests replayed      166  ->    166                                     context
```

The eval intervals are that wide because this run used 20 questions to fit the
settle budget; the interval is the honest report of what 20 questions can say.

**The patch was proven live, not merely mounted.** `check_patch_live` found the
file inside the running container hashing to `sha256_patched`, the import marker
in the engine log **9 times** — eight tensor-parallel ranks plus the driver — and
the first-call marker **8 times**, meaning real requests entered the patched code
path. That is the evidence the whole stage is built to produce.

### What the runs cost, and what they found

Per-step timings, CUDA graphs on: `serve_stock` 257 s with the checkpoint in page
cache; smoke 22 s; needle at two lengths 52 s; probe 12 s; `gsm8k` plus
`mixed_prefix_gsm8k` at 100 questions each 62 s; a 60-second trace replay 85 s and
166 requests. A whole two-arm run is about 18 minutes at the reduced settings
above.

Five runs, and **four of them found something the design had wrong**. Each is
fixed, and each is written into `DESIGN.md` rather than quietly corrected,
because a design that records only what it got right is not a record:

1. A `froms` edge with no handoff behind it does not order execution. Both serve
   tasks were dispatched at once and collided on the container name.
   `DESIGN.md` §4.1, `../temp/bugs/004`.
2. Gating the eval on the probe strands a whole arm's completed work when the
   probe has a bad afternoon. `DESIGN.md` §5.4.3.
3. The reasoning parser makes the generation budget a knob that disguises itself
   as a correctness problem — at 512 and again at 2048 tokens the long-generation
   check got an empty answer and 1470 words of reasoning. `DESIGN.md` §5.4.1.
4. The patched arm's own evidence names container paths, which the publication
   seal refuses, so `serve_patched` wrote its complete handoff and then aborted
   on the last line. `DESIGN.md` §8.

A fifth is the framework's, not the design's: a graph that outruns the
1800-second settle budget cannot be finished by `--resume` either, because the
interrupted task leaves its output slot open and the store refuses to reopen it
(`../temp/bugs/005`).

## Running it

Needs Python 3.12 or newer. `agent_sys` declares `requires-python = ">=3.10"` but
cannot be imported below 3.12; see
`../temp/bugs/001-requires-python-3.10-but-fails-below-3.12.md`. The repository's
own `.venv` is 3.12 and works.

Needs a Slurm allocation already holding the node, because the login node has no
GPUs and cannot ssh to a compute node:

```bash
sbatch --parsable --partition=Compute-DCPT --nodelist=<node> --nodes=1 \
  --ntasks-per-node=1 --exclusive --time=04:00:00 --job-name=hold \
  --wrap='srun --ntasks=1 sleep infinity'
```

Then:

```bash
export PATH="<repo>/.venv/bin:$PATH"
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/integration-demo \
  --demo-root "$HOME/.agent_sys_runs_integration" \
  --var jobid=<the job id> \
  --var node=smci355-ccs-aus-n04-33 \
  --var node_ip=10.235.192.139 \
  --var model_path=/apps/qiongzhu/models/GLM-5.3-Flash-FP8
```

Four variables carry no default. A Slurm job id, which node it holds, that node's
IP and where the weights are, are facts about one allocation on one cluster; a
default would be one machine's answer shipped as everyone's, and would go stale
the moment the job ends. Omit one and the load fails naming the file, the line
and the variable.

`--demo-root` is worth setting. The default is one directory shared by every
package, and `agent-sys run --clean` from a concurrent run removes everything in
it.

**First bring-up**, which trades resolution for a round in minutes:

```bash
  --var eval_examples=20 --var needle_tokens=31000 \
  --var needle_frontier_tokens=0 --var bench_rounds=1
```

Everything else defaults; `shared.yaml` is the only place this package may name
its own variables, and every one is commented there.

`AGENT_SYS_NO_PERMISSIONS=1` is required by the mission this package is developed
under. The `permissions.grants` blocks are written anyway — they are the answer
to "what does this touch", and writing them now means turning enforcement on
later is a configuration change rather than an archaeology exercise.

## Prerequisites that are not part of the package

1. **The engine image, built on the node that will run it.** Docker images are
   node-local, so an allocation on a different machine needs a rebuild:

   ```bash
   docker build -f examples/glm53flash-demo/patches/Dockerfile.sglang.glm53 \
     -t infera/engine-sglang:glm53-flash .
   ```

   Measured at 9m25s. No released SGLang carries `model_type: glm5_next`.

2. **The AIPerf image**: `docker pull nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`.

3. **The GSM8K test split**, 1319 rows. Already staged at the path
   `gsm8k_data` defaults to; `lm_eval.sh` warns if the row count is wrong,
   because a truncated download moves every score.

Nothing for the weights: `/apps` sustains 921 MB/s single-stream on this node and
the node's 3 TB of RAM keeps the checkpoint in page cache after the first load.

## What it costs

Two cold starts. Measured on `profiling-demo`: 819 s reading 306 GB of FP8
weights off NFS the first time, 243 s for a second bring-up in the same session
served from page cache, plus about 33 s of CUDA graph capture each. The
measurement itself is roughly 13 minutes per arm at the defaults — of which
lm-eval is most (measured: 200 gsm8k questions at 32 threads took 3m17s with
graphs off).

Budget 45 to 70 minutes for a full run. Do not read silence as a hang: a program
body's stdout is not streamed, and `agent_sys` prints nothing between dispatch
and the end of a phase.

## The three things this package is really about

**A baseline you can believe.** `profiling-demo` replayed the same trace against
the same configuration twice and measured 631 output tok/s with 25.9 s mean TTFT
cold, against 1004 tok/s and 484 ms with the deployment reused — a Mooncake trace
carries `hash_ids`, so prefix hit rate decides how much prefill there is to do.
Nothing recorded in another session is comparable, which is why both arms run
here, back to back, in the same order.

**Proof the patch was actually exercised.** A patch that is mounted and never
entered gives two identical arms and a green report. `check_patch_live` is the
only validator here with `dimension: trustworthiness` and it exists for that one
failure. Note that `__file__` proves nothing — a bind mount leaves the path
inside the container unchanged, so it reads identically on both arms. What proves
it is the sha256 taken **inside the running container**, plus the markers the
patch declared appearing in the engine log.

**A verdict that was checked, not read.** `check_no_regression` does not read the
report's `verdict` field. It recomputes every comparison from the raw numbers and
fails if its answer differs — including when both say "accepted", because two
answers that disagree mean one is wrong.

## The needle is a regression detector, not a capability gate

Worth knowing before reading a needle result out of this package. Nine
measurements on this deployment (the table is in `assets/accept/needle.py`, the
raw runs in `temp/manual/FINDINGS.md`) found retrieval to be:

- **not monotonic in prompt length** — 41k retrieved, 51k did not, 62k did, 76k
  did not, with the same needle and the same budget;
- **sensitive to the needle's wording** — a six-digit passphrase in filler that
  is itself numbered is unretrievable at head depth in a 76k prompt, three times
  identically at temperature 0, while a lexical one at the same position is
  retrieved every time;
- **sensitive to the generation budget, in the wrong direction** — raising
  `--max-tokens` from 256 to 2048 turned two passing depths into failures, with
  the model reasoning its way to "I don't have any information about a
  maintenance passphrase". A small budget starves the answer, since the engine
  runs with `--reasoning-parser glm45` and `content` stays empty until thinking
  stops; a large one talks it away.

So no configuration measured here retrieves at every depth twice, and there is no
absolute gate. What is asserted per arm is that the prompt reached the length it
claims — a fact about the request — and that **at least one** depth retrieved,
which separates a working long-context path from a stack that dropped every
prefill chunk but the last. Everything else is comparison material: a depth the
stock arm retrieved and the patched arm did not.

The same trap appears in the long-generation smoke check: at 512 tokens the whole
budget went to reasoning and the answer came back empty. Measured at 2048, the
answer is about a tenth of the tokens generated — 153 words of answer after 1307
of reasoning. Any check on this deployment that budgets for the answer alone
measures the reasoning parser instead.

## Known gaps

- **`seed_patch` is a mock.** The kernel-optimization stage has not landed and
  `analyze-demo`'s design leaves forge-loop's output unspecified, so this package
  defines the `kernel_patch` contract and mocks a producer for it. The mock is
  semantics-preserving on purpose: the expected verdict is "no difference", which
  is what makes the validators themselves checkable. A green report from it is
  evidence about the pipeline, not about any optimisation.
- **Patches that need compiling are out of scope.** `apply_mode: rebuild` fails
  at the first validator naming the reason. HIP, CK and assembly kernels cannot
  be bind-mounted.
- **`runtime_marker` is optional**, so a real forge patch that declares none gets
  only the static half of `check_patch_live`. Whether to require it is open
  question 2 in `DESIGN.md` section 11.
- **The performance bars have no measured basis yet.** 5% on throughput and 10%
  on latency are placeholders; the number they want is the natural run-to-run
  spread of one arm, which the first full run will produce.
- **200 questions is a ±5-point Wilson interval.** Enough to catch a broken
  deployment, not a two-point regression. `eval_examples` goes to the full set at
  about six minutes per eval.
- **No teardown step.** If the graph fails partway the engine container keeps the
  GPUs. Both serve tasks start with an idempotent teardown and a VRAM gate, so
  rerunning is safe; recovering without rerunning is
  `docker rm -f glm53_int glm53_int_etcd`.
- **No `resources` block.** The leaves legitimately want eight GPUs, but the CLI
  composition root declares no pools, so naming one would name a pool that does
  not exist. The Slurm allocation is what keeps two runs from colliding.
- **The bodies exchange files through the attempt zone**, which needs the run
  root on a filesystem the compute node also mounts. `$HOME` here is NFS from the
  same server, so it works; `assets/lib/remote.sh` asserts it rather than
  assuming it.

## Layout

```
integration-demo/
├── DESIGN.md                 the reviewed design, in Chinese
├── README.md                 this file
├── main.yaml                 root closure + subgraph
├── shared.yaml               the program agent, and every package variable
├── steps/
│   ├── patch.yaml            kernel_patch / patch_overlay + their validators + tasks
│   ├── stock.yaml            the stock arm, and the three validators both arms share
│   ├── patched.yaml          the patched arm, and check_patch_live
│   └── verdict.yaml          integration_report / integration_packup + compare + packup
└── assets/
    ├── lib/                  shared by task and validator bodies
    │   ├── remote.sh             srun into the allocation
    │   ├── nodecall.py           the same seam, for python bodies
    │   ├── redact.py             site paths -> @NAME@, then the seal's own rule
    │   ├── patchkit.py           manifest and mount plan: one reader, one writer
    │   ├── container_roots.yaml  @NAME@ -> the path inside the image
    │   ├── eval_stats.py         Wilson intervals and Newcombe differences
    │   └── store.py              reading a published handoff
    ├── serve/                mix_up.sh, mix_worker.sh, mix_smoke.sh, reset_gpus.sh, round.sh
    ├── accept/               measure.sh, smoke.py, needle.py, probe.py, lm_eval.sh
    ├── bench/                aiperf_replay.sh, summarise.py, pythonpath/sitecustomize.py
    ├── <task>.task/          readme.md, entry.sh, and a body where there is one
    └── <validator>.validator/  readme.md, entry.sh, check.py
```

`assets/serve/`, `assets/accept/`, `assets/bench/` and `assets/lib/` carry no
`.task` suffix because they are shared scripts rather than body directories. A
shared script may not be named after a closure: `spec_loader` resolves a body by
matching the closure's name against filenames under `assets/`, and two candidates
make the load refuse to guess.

## Joining up with the rest of the pipeline

**Upstream.** When the kernel-optimization stage lands: delete `seed_patch` and
its section of `steps/patch.yaml`, and point `apply_patch`'s `froms` at that
stage's producing task. The `kernel_patch` kind is defined in `steps/patch.yaml`
rather than in `shared.yaml` precisely so that it moves upstream with its
producer and only the consumer stays here.

**Downstream.** There is none. `integration_report` is the pipeline's verdict and
`integration_packup` is what a human is handed.
