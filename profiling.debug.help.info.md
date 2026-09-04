# profiling-demo — debug/localisation notes (2026-09-02, node crsuse2-m2m-079, job 101052)

## 1. `srun --overlap` does not exist on this cluster (BLOCKER, must localise)

`assets/lib/remote.sh::on()` runs

    srun --jobid=$PD_JOBID --overlap -N1 -n1 -w $PD_NODE --export=ALL bash -lc "$*"

`/usr/local/bin/srun` here is **not Slurm's srun**. It is a spur re-implementation
(Rust/clap, 22 MB, dated 2026-09-01). Measured:

- `--export` is not a recognised argument at all:
  `error: unexpected argument '--export' found`.
  Its accepted set is `--job-name --partition --account --qos --nodes --ntasks
  --ntasks-per-node --cpus-per-task --mem --time --gres --licenses --gpus
  --gpus-per-node --gpus-per-task --chdir` (+ `--jobid --overlap --nodelist`).
- With `--export` dropped it still fails non-interactively:
  `spur: warning: raw mode unavailable (stdin is not a TTY)`, **exit 128**.
  It wants a TTY; agent_sys bodies have no TTY. So it is unusable as a transport.

=> The transport must become `spur exec <jobid> bash -lc '...'`. Keep the srun
form selectable rather than deleting it (other cluster still uses it).

## 2. `spur exec` does NOT carry the environment across (the real localisation trap)

`--export=ALL` in the original is load-bearing: the remote side must see
`AGENT_SYS_OUTPUT_*` and the whole `PD_*` block. Measured from the login node:

    MARK=hello_from_login spur exec 101052 bash -lc 'echo "MARK=[$MARK]"'
    -> MARK=[]

Also measured, and each has cost time before:

    hostname -> crsuse2-m2m-079      (right node, controller proxies by jobid;
                                      you do NOT pass a node name)
    id -un   -> yihou                (NOT root on this cluster)
    PWD      -> /                    (always `cd` first)
    HOME     -> /opt/spur            (not /home/yihou)
    PATH     -> no ~/.local/bin

So `on()` must serialise the environment itself. See section 3 for what was done.

## 3. The 1800 s settle budget is stale — it is 14400 s in this worktree

Widely repeated as a hard constraint ("design your run to fit inside 1800 s").
Not true here. `agent_sys/cli/main.py:902`:

    _SETTLE_TIMEOUT = 14400.0

and `--timeout SECONDS` is an exposed `agent-sys run` flag (main.py:165-175)
defaulting to it. The comment above the constant records the exact regression
that made 1800 s untenable — "it was then 1800 s, which killed a healthy
bring-up ... the agent is abandoned rather than asked to stop, so the containers
and the eight GPUs it held stayed held". Introduced by commit 6571a92.

A run that STOPS MAKING PROGRESS still ends in seconds (stall detection); the
ceiling only bounds a run that never stops. So this is not a licence for a hang,
but it does mean an 819 s cold bring-up is affordable.

## 4. Node facts that differ from the package's assumptions (crsuse2-m2m-079)

- Node system `python3` is **3.12.3 with numpy 1.26.4 and PyYAML present**, so
  README's "Known gaps" entry about `kernel_scan` needing numpy/PyYAML in the
  node's system python3 does NOT bite here. `python3 -m Magpie` can run.
- `/apps` is **ABSENT**. Every default that points into it is dead:
  `magpie_root`, `aiperf_trace`, `model_path`.
  Present instead: `/shared_nfs/chaox/Magpie`, `/shared_nfs/age/hl/Magpie`,
  `/shared_nfs/models/GLM-5.3-Flash`, `/shared_nfs/yihou/models/Qwen3.6-27B`.
- `/home/yihou` and `/shared_nfs/yihou` are both visible on the node, so the
  `require_visible_on_node` assertion in remote.sh can be satisfied by either.
- **No `infera/engine-sglang:*` image exists on this node.** Docker images are
  per-node here; the one the BRIEF says was built on 2026-09-01 lives on a
  different node. `docker images` on 079 shows only unrelated vllm/atom/primus
  images. The base `lmsysorg/sglang:v0.5.17-rocm720-mi35x` had partial layers
  cached and pulls fine.
- `df /` inside `spur exec` shows 50 G free but that is the exec namespace's
  rootfs. Docker's real storage is on `/mnt/m2m_nobackup` (28 T, 21 T free).

## 5. Backgrounding work on the node: `spur exec` from a login-side `&` dies

`nohup spur exec ... &` from the login node exited immediately with an empty log.
What works is to detach on the NODE side and log to NFS so the login node can
poll it:

    spur exec 101052 bash -lc 'setsid nohup docker pull IMAGE \
      > /shared_nfs/yihou/.../pull.log 2>&1 < /dev/null & echo pid=$!'

The `setsid` + `</dev/null` pair is what makes it survive the exec's exit.

## 6. The localisation of `remote.sh` — what it actually took (SMALL)

The fear was that `--export=ALL` had to be re-implemented over a transport that
carries no environment. **It did not.** Verified first-hand:

    grep -rn "PD_\|AGENT_SYS_" assets/serve/mix_up.sh assets/serve/mix_worker.sh \
      assets/serve/mix_smoke.sh assets/serve/reset_gpus.sh assets/analyze/megapie.sh \
      assets/analyze/launchers.py assets/load/capture.sh assets/load/aiperf_replay.sh
    -> no matches

Every far-side script reads plain names (`NODE_IP`, `IMAGE`, `CTR`, ...) and
every `on` call site already supplies them as an explicit `VAR='...'` prefix.
`assets/serve/round.sh:93` documents why: an early run "appeared to work only
because the operator's login shell still had them exported". So `--export=ALL`
was vestigial, and dropping it costs nothing.

The change is therefore confined to `on()` in `assets/lib/remote.sh`: a
`PD_TRANSPORT` switch (`auto` | `spur` | `srun`), resolved once at source time,
`auto` preferring `spur` wherever the binary exists. **The srun form is kept, not
deleted** — the other cluster still needs it. Note that probing for an `srun`
BINARY is not a valid way to choose srun: the spur cluster ships its own
`/usr/local/bin/srun` that is not Slurm.

`spur exec` semantics measured against job 101052, all needed by the call sites:

| property | result |
|---|---|
| exit code propagation | exact (`exit 42` -> 42; `test -e` absent -> 1) |
| nested quoting (the `round.sh:121` `tr '\0' '\n' < /proc/$(pgrep ...)` shape) | survives |
| pipes inside the command string | work |
| two concurrent calls | work — `replay.sh` backgrounds one and needs this |
| node name | NOT passed; the controller routes by job id |

`PD_NODE` is consequently unused by the spur transport but is still a required
`--var`; it stays because the handoff records which node produced the evidence.

## 7. Image: none existed on this node, had to be rebuilt (~25 min)

`infera/engine-sglang:gfx950-local` was built on a DIFFERENT node on 2026-09-01.
Docker images are per-node here, so it did not exist on 079. Rebuilt:

1. `docker pull lmsysorg/sglang:v0.5.17-rocm720-mi35x` (110 GB on disk /
   28.5 GB content; some layers were already cached).
2. Install infera into it. **Two traps:**
   - Mounting the repo **read-only** fails: `error: could not create
     'amd_infera.egg-info': Read-only file system`. Copy it in
     (`cp -a /repo/. /build/`) and install from the copy. The repo is 17 MB
     without `.git`.
   - The build backend is `setuptools_scm`, and a git **worktree**'s `.git` is a
     file pointing outside the mount, so version detection fails. Pass
     `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0`.
3. `docker commit dbg_profiling_build infera/engine-sglang:gfx950-local`.

Verified in the committed image: `python3 -m infera.server --help` and
`python3 -m infera.engine.sglang --help` both work, and
`/sgl-workspace/sglang/python/sglang/srt/models/` carries `qwen3_5_text.py` /
`qwen3_5_mtp.py`. `/shared_nfs/yihou/models/Qwen3.6-27B/config.json` declares
`model_type: qwen3_5`, so this image can serve it. It carries **no** `glm5_next`.

## 8. No Mooncake/AIPerf trace exists on this cluster — synthesised

`find /shared_nfs -maxdepth 6 \( -name 'conversation_trace*.jsonl' -o
-name '*mooncake*' -o -name 'aiperf_trace' \)` returned **nothing**. The default
`PD_AIPERF_TRACE` points into `/apps`, which is absent. A stand-in was
synthesised — see the section below for its exact shape and the loud caveat that
it is NOT a production trace.

## 9. The Qwen substitution WORKS — the GLM hard-coding is inert, not blocking

`assets/serve/mix_worker.sh:70-79` launches the engine with flags that are not
parameterised and are named for GLM-5.3-Flash:

    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang   (line 73)
    --moe-runner-backend triton                                     (line 74)
    --context-length 262144                                         (line 75, ${CTX:-262144})
    --reasoning-parser glm45 --tool-call-parser glm47               (line 77)

`round.sh` does not pass `CTX`/`KV_DTYPE`/`MOE_RUNNER`/`GMU`/`CHUNK` through, so
those defaults are unreachable from the package's own `--var` surface. It LOOKS
like a hard block on substituting another model. **Measured: it is not.**

Brought Qwen3.6-27B up through the package's own `mix_up.sh` at `tp=2` on
`infera/engine-sglang:gfx950-local`:

    ===== MIX_UP_OK  endpoint: http://10.245.155.242:8120 =====
    worker serving after 360s ... router healthy
    profiling control plane ON (probe -> 400 invalid role)

and its own `mix_smoke.sh` then reported `SMOKE_ARITHMETIC_OK`, a coherent
"The capital of France is Paris.", `finish=stop`, and one registered worker with
`disagg_mode: mixed`. Why each suspect flag is harmless:

- the DSA backends are server-side attention-backend selections; Qwen3.5 does
  not use DSA, so they are never consulted;
- Qwen3.6-27B is dense (`num_experts` absent), so the MoE runner is inert;
- `Qwen3.6-27B/config.json` has `text_config.max_position_embeddings = 262144`,
  which is EXACTLY the hard-coded `CTX` default — this one is luck, and a model
  with a shorter context would fail here;
- the GLM reasoning parser happily produced `reasoning_content` for Qwen.

So no body needed editing. **A later reader should still treat `CTX` as the
booby trap**: substituting a model with a context shorter than 262144 will fail
at startup with nothing in the package to override it.

Cold start was **360 s**, not the ~90 s estimated — 51.75 GiB of Qwen weights
off `/shared_nfs`, which is ~98% full and slower than the `/apps` the package's
819 s GLM figure was measured against.

## 10. `docker commit` freezes an `--entrypoint` override into the image

Building the infera image by `docker run --entrypoint bash ... -c "sleep infinity"`
then `docker commit` produces an image whose `ENTRYPOINT` is `["bash"]`. The
package then does `docker run "$IMAGE" sleep infinity`, which becomes
`bash /usr/bin/sleep infinity` — bash tries to run the binary as a SCRIPT:

    /usr/bin/sleep: /usr/bin/sleep: cannot execute binary file    (exit 126)

and `mix_up.sh` reports only `container ... is not running`. `docker commit
--change "ENTRYPOINT []"` does **not** clear it (verified — it stayed `["bash"]`).
Build with a Dockerfile instead, so the base's `ENTRYPOINT=null` is inherited:

    FROM lmsysorg/sglang:v0.5.17-rocm720-mi35x
    COPY . /build
    RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir "/build[sglang]" && rm -rf /build

**`docker build` additionally needs a writable HOME under `spur exec`**:
`HOME=/opt/spur` is not writable and buildkit dies with
`ERROR: mkdir /opt/spur/.docker: permission denied`. Export
`HOME=/tmp DOCKER_CONFIG=/tmp/.docker` first.

Note `--group-add video --group-add render` in `mix_up.sh` is FINE here — both
groups resolve by name on these nodes (44 / 992). The advice elsewhere to use
numeric ids does not apply to this node.

## 11. `extensions.preciousObjects` in a WORKTREE writes to SHARED config — do not

`agent-sys run` refuses to start without it:

    /home/yihou/dev/git/... does not set extensions.preciousObjects, and
    `env_mgr.workspace.cut` refuses without it -- so every output-producing task
    would die in `prepare`.

Its own message warns why this matters here, and it is right:

    Note: in a git worktree this lands in the SHARED common config, so it affects
    the main checkout and every other worktree, and `git gc` will refuse in all
    of them until it is unset.

`git rev-parse --git-common-dir` for this worktree is `/home/yihou/dev/git/infera/.git`
— shared with the main checkout and with **every other agent's worktree**. With
five agents running concurrently, setting it is a change to other people's
repositories, so it should not be done unilaterally.

Two escapes were considered:

- **Do NOT reuse `/shared_nfs/yihou/agent_sys_debug/repo`.** It already has
  `preciousObjects=true`, but it is at a detached HEAD with another module's
  UNCOMMITTED work in its tree (integration-demo's `remote.sh`, `mix_worker.sh`,
  `round.sh`, `shared.yaml`). Pulling or checking out there would destroy it.
- **What was done instead:** a private clone nobody else touches —

      git clone --no-hardlinks --branch dev.yihou.aiopt.task_package \
        /home/yihou/dev/git/infera.aiopt.real.task_package \
        /shared_nfs/yihou/agent_sys_debug/ws2/profiling/pkgrepo
      git -C .../pkgrepo config extensions.preciousObjects true

  A clone has its own config, so `preciousObjects` there affects nothing else.
  The run is launched from the clone; the DELIVERABLE is still committed in the
  worktree, and the clone is re-created from it after each commit.

## 12. Run of record

    run id     20260902T085925-9f72b7
    demo-root  /shared_nfs/yihou/agent_sys_debug/ws2/runroot/profiling-a
    node       crsuse2-m2m-079 (job 101052), ip 10.245.155.242
    model      Qwen3.6-27B at tp=2, served as qwen3.6-27b
    image      infera/engine-sglang:gfx950-local (built here, see section 7)
    ports      router 8120, worker 8121, etcd 8122
    reductions trace_end_ms 120000->180000 (LONGER: the window must fit inside
               the replay), warmup_s 30->60 (AIPerf cold prompt synthesis),
               window_s 15->10, max_conc 256->32, workers_max 16->8,
               tp 8->2. stack_window_s and stack_ranks left at 3 / 2.
    substituted aiperf_trace -> synthesised (section 8)
                magpie_root  -> /shared_nfs/chaox/Magpie
                work_root    -> /mnt/m2m_nobackup/yihou/profiling

## 13. RESULT — run 20260902T085925-9f72b7 passed clean, first attempt

7 tasks succeeded, 7 handoffs valid, 6 validator verdicts PASS, in ~22 minutes:

    serve_baseline 252s -> run_baseline -> serve_profiled 260s -> run_profiled 416s
    -> kernel_scan -> packup

Evidence opened rather than inferred from the exit code:

| file | what it says |
|---|---|
| `torch_trace/items/result/manifest.json` | 2 ranks, 209,779 / 209,796 GPU kernels (floor 1,000), span 11.76 / 11.82 s for a 10 s window |
| `torch_trace/items/result/stacks_manifest.json` | 2 stack ranks, 4,747,427 / 4,708,020 `python_function` events (floor 10,000) |
| `kernel_table/items/result/text.json` | 124 kernels; launchers resolved 46/50, 2 unmapped, `{sglang:30, aiter:15, sgl_kernel:1}`; **22 of top 25** carry a frame (floor 5) |
| `results/gap_analysis.csv` | 124 rows, top 25 hold 92.9% of self CUDA time (floor 50%) |
| `results/aiperf_*.summary.json` | 721 requests replayed per round = every record in the trace |
| `items/codes/REPRODUCE.md` | says `spur exec`, `2 × MI355X`, `Qwen3.6-27B` — the render.py change landed |

Delivered to `/shared_nfs/yihou/agent_sys/debugging/profiling/` with
`PROVENANCE.md`: all seven handoffs plus the synthesised trace and its generator.

### Two things a later reader should NOT be surprised by

1. **`profile_packup/items/codes/results/top_kernels.json` carries only the top
   25 and `launchers: null`.** That is not a fault and not what
   `check_kernel_table` reads — the validator reads the `kernel_table` handoff's
   `items/result/text.json`, which has all 124 rows and the full launcher block.
   The packup copy is a summary. If you are stitching flows, take the ranking
   from `kernel_table`, not from the packup.
2. **The `usage names 'seconds'` line on every task** —
   `... which the task did not declare and which cannot record unreserved spend;
   252.245... is not booked` — is a pre-existing agent_sys accounting warning,
   not a failure. It appeared on all seven tasks of a fully passing run.

### Timings, for anyone budgeting a rerun

| step | cost |
|---|---|
| pull `lmsysorg/sglang:v0.5.17-rocm720-mi35x` | ~20 min (partial layer cache) |
| build the infera image on top | ~2 min |
| Qwen3.6-27B tp=2 cold start off `/shared_nfs` | **360 s** (252 s warm) |
| whole graph | ~22 min |

The package README's "24 minutes from cold" holds, but its 819 s GLM figure and
its `1800 s settle budget` warning do not apply here (see sections 3 and 9).

## 14. The empty-`content` trap, and why the obvious diagnosis is WRONG

A warning went round this effort that `--reasoning-parser glm45` (hard-coded at
`mix_worker.sh:77`) "leaves `content` empty for every request on a non-GLM chat
template", and that a profile built over such a deployment would be a profile of
a model answering nothing while every validator passed. **Half of that is true
here and the stated cause is not.** Measured, because a smoke test passing is
not evidence about what the REPLAY got back.

### What the delivered run actually contains

Over `items/result/profile_export.jsonl.gz`, both rounds, all 721 requests:

    requests=721  osl_total=80684  osl_min=64  osl_max=160  zero_output=0
    reasoning_tokens=80684        requests with any CONTENT token: 0/721

So `content` was indeed empty for every request. But `output_sequence_length` is
never zero and its range is *exactly* the trace's `output_length` range — the
engine decoded all 80,684 tokens. **The GPU work is real; the trace and the
kernel table are measurements of genuine prefill and decode.**

### The cause is truncation, not a mis-parse

Two probes at the same live deployment, same model, same GLM parser, only
`max_tokens` varying:

| max_tokens | finish_reason | reasoning / content tokens | content |
|---|---|---|---|
| 100 | `length` | 100 / 0 | `''` |
| 600 | `stop` | 148 / 9 | `'\n\nThe capital of France is Paris.'` |

and the extracted reasoning is coherent English —
`'Thinking Process:\n\n1. **Analyze the Request:** * Question: "What is the
capital of France?"...'`. **`glm45` parses Qwen3.6-27B's thinking block
correctly.** Qwen3.6 is a thinking model; at a 64–160 token budget it never
finishes thinking, the reasoning block never closes, and no content is emitted.
The earlier smoke test passed because `mix_smoke.sh` uses `max_tokens: 512`.

**Why this distinction matters:** a wrong parser and a truncated thinking model
produce an identical report — empty `content`, successful requests, healthy
token counts. If you see empty `content`, check `finish_reason` and
`reasoning_token_count` before blaming the parser. For a downstream evaluator
scoring 0.00, raising the generation budget is as likely a fix as changing the
parser.

Note also, contrary to the warning: **`--dsa-*-backend tilelang` is NOT rejected
by sglang for Qwen on `lmsysorg/sglang:v0.5.17-rocm720-mi35x`.** The engine
started and served with the flags present. Another SGLang build may differ.

## 15. `DSA_ARGS` / `PARSER_ARGS` are now variables (commit 2735e0a)

Both groups hoisted out of `mix_worker.sh`, defaults byte-for-byte the strings
they replaced, threaded `shared.yaml` -> `round.sh` -> `mix_up.sh` ->
`mix_worker.sh`. Two details that are not cosmetic:

- **Read into arrays** (`read -r -a`), not left as scalars. Each group carries
  several flags, and an empty scalar expanded at the call site hands sglang one
  empty argument rather than none.
- **`none` is the spelling for "neither"**, because agent_sys's variable syntax
  has no way to express an empty default and an empty `--var` is
  indistinguishable from an unset one.

Verified before running: defaults expand to 4 and 4 arguments, `none` to 0 and 0.
The design is identical to `integration-demo`'s copy of the same file — same
two variable names, same sentinel — so the two packages stay comparable.

## 16. The login node OOM-kills a long run, and the symptom names nothing

Run B (`20260902T093432-ee4782`) died with no error line anywhere. What it
looked like — recognise this shape, none of it says "killed":

- the run log stops mid-graph (19 lines), no `done`, no traceback;
- the task sits at `serve_baseline: running` indefinitely;
- **the body outlived the driver.** `round.baseline/mix_up.log` shows
  `MIX_UP_OK`, the engine served, `mix_smoke.sh` completed all five sections and
  all twelve evidence files were written — then nothing. The body took SIGPIPE
  writing to the dead parent's captured stdout, immediately before the handoff
  assembly, so the store holds a `claim` and an **empty `content/`**;
- no `agent-sys` process remains. `pgrep -f "agent-sys run"` is a trap here — it
  matches your own shell's command line. Use
  `ps -eo pid,cmd | grep agent-sys | grep -v grep`.

Cause: `crs-m2m-cpu-spur-012` at the time was **62 GB total, 1 GB free,
18 GB available, load average 30.41, 169 users**. Nothing in the package
changed between the run that passed and the run that died.

**Fix: run `agent-sys` on the compute node** (2751 GB RAM, 236 cores) and give
`on()` a `local` mode — commit `8274a08`, `PD_TRANSPORT=local`, `on()` becomes a
plain `bash -lc`. `auto` never picks it on purpose: "neither transport binary is
present" is not the same fact as "I am on the node", and guessing wrong runs
every GPU command on the login node.

### Two gotchas when moving a run onto the node

1. **`agent-sys` refuses to start with "the 'claude' backend is not on PATH"
   even for a package with no AI agent at all** — every closure in
   profiling-demo is `kind: program`. The check is unconditional. Export both:

       export PATH=/home/yihou/.local/bin:/home/yihou/miniconda3/bin:$PATH
       export HOME=/home/yihou

2. **Detach on the node side**, or the run dies when the `spur exec` returns:

       spur exec <job> bash -lc 'cd <dir> && setsid nohup bash run.sh \
         > run.log 2>&1 </dev/null &'

   The `cd` runs inside the backgrounded subshell, so a follow-up command in the
   same `spur exec` still has `pwd=/` — `tail run.log` fails with "No such file"
   and looks like the launch failed when it did not. Use absolute paths.

Note this does not contradict the user's directive to keep workspace, playground
and handoff roots on `/shared_nfs`: the run root stays there and both hosts see
it. Only the *driver process* moves.

## 17. Final state — three consecutive clean passes, run D delivered

| run | driver | notable vars | outcome |
|---|---|---|---|
| `20260902T085925-9f72b7` | login node | GLM parsers left on | PASS, but 0/721 content-bearing |
| `20260902T093432-ee4782` | login node | `parser_args=none` | **killed** — login-node OOM (section 16) |
| `20260902T095359-f04255` | **node**, `transport=local` | `parser_args=none` | PASS, 721/721 content-bearing |
| `20260902T102357-ad761d` | **node**, `transport=local` | same + REPRODUCE fix | PASS — **delivered** |

Three consecutive passes with no intervention; the one failure was the login
node killing the driver, not the package.

**Delivered** to `/shared_nfs/yihou/agent_sys/debugging/profiling/`: all seven
handoffs of run D, the synthesised trace, its generator, and `PROVENANCE.md`.

Run D's evidence, opened:

    traces  2 ranks x 209,609 GPU kernels, spans 11.763 / 11.749 s (10 s window)
    stacks  2 ranks, 4,833,693 / 4,798,112 python_function events
    kernels 124 rows, top 25 = 92.6% of self CUDA time, launchers 46/50,
            24 of top 25 carry a source frame
    aiperf  721 requests per round, 80,686 / 80,685 output tokens,
            721/721 content-bearing

### Cost of a rerun, measured three times

| step | cost |
|---|---|
| Qwen3.6-27B tp=2 cold start off `/shared_nfs` | 360 s first, **247–261 s** warm |
| whole graph | **~22 min** |
| build the engine image (base already pulled) | ~2 min |
| pull `lmsysorg/sglang:v0.5.17-rocm720-mi35x` | ~20 min |

### If you rerun this package here, the short version

    --var transport=local        and run agent-sys ON the node
    --var parser_args=none       for any non-GLM model
    --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B --var tp=2
    --var image=infera/engine-sglang:gfx950-local     (build it; per-node)
    --var magpie_root=/shared_nfs/chaox/Magpie
    --var work_root=/mnt/m2m_nobackup/yihou/profiling
    --var aiperf_trace=<the synthesised trace>
    --var trace_end_ms=180000 --var warmup_s=60 --var window_s=10

and export `PATH=/home/yihou/.local/bin:/home/yihou/miniconda3/bin:$PATH` plus
`HOME=/home/yihou` before `agent-sys`, or it aborts on a `claude` backend it
never uses.

## 18. A delivered handoff is `<hid>/v<N>/`, not `content/`

The first delivery copied only each handoff's `content/` directory. **That is
unreadable by anything.** A store needs `<root>/<hid>/v<N>/manifest.yaml`; with
only `content/` there is no manifest, no `claim/`, no `validation.yaml`, and a
version without a manifest is *unpublished by definition* — the digest and the
validator verdicts are gone with it. `analyze-demo/assets/lib/store.py` pointed
at such a directory returns `versions() == []`, `kind_of() == ''`,
`content_dir() is None`. It is not recoverable by relayout; the files were never
copied.

The shape that reads, with a `store/` level because the delivery directory also
holds `PROVENANCE.md` and the trace and so cannot itself be a store root:

    debugging/profiling/store/<handoff-uuid>/v<N>/{content,manifest.yaml,validation.yaml,claim}
    export AGENT_SYS_DEMO_STORE=.../debugging/profiling/store

Copy with `/shared_nfs/yihou/agent_sys/temp/kernel-opt/relayout_handoffs.py`
(needs `--repo <repo>/agent_sys`; point it at the RUN's `handoffs/` directory,
not at a delivery). It reads uuid and version from each `manifest.yaml`, skips
unpublished versions, verifies source and copy, and chmods **directories only**.

**Why directories only:** `handoff/digest.py:82` computes a git-shaped tree
digest that records each file's exec bit, and `handoff/store.py:284` recomputes
it on **every** consumption and raises `DigestMismatch`. So a `chmod -R 777`
over a delivery does not read oddly — it **fails the consuming task**. A
directory enters the digest as a constant, so chmod-ing directories is free.

### Verify THREE times, and do not collapse any of them

**This section originally said "verify twice". That was wrong, and the third
check is the one that catches a delivered set being invalid.** Found by
`kernel-opt` on 2026-09-02: they copied `integration`'s nine handoffs, verified
digests and layout, and reported "9/9 verified" — while `integration_report`
carried `check_no_regression: result=False` at `strength: strong`. That refused
verdict, not the OOM everyone had settled on, is why their terminal
`integration_packup` never ran.

> **A digest proves the bytes have not changed since sealing. It says nothing
> about whether what was sealed was acceptable.**

So the third check is: **read every `validation.yaml` and look at the verdicts.**

```python
import yaml
from pathlib import Path
for v in sorted(Path("<store>").glob("*/v*")):
    kind = yaml.safe_load((v/"manifest.yaml").read_text())["kind"]
    for e in (yaml.safe_load((v/"validation.yaml").read_text()) or {}).get("verdicts", []):
        print(kind, e["validator"], e["result"], e.get("strength"))
```

Anchor the walk on `manifest.yaml`, not on `validation.yaml`: a version with **no**
`validation.yaml`, or one whose `verdicts:` list is empty, then shows up as
missing rather than being skipped. **An empty verdict list is not a pass** — it
is what an unvalidated handoff looks like, and a check that counts `result: false`
occurrences reports it as clean. The team lead's own sweep had that hole.

All seven of this module's handoffs carry `result=True` at `strength=strong`.

**Why neither producer had this habit, which is the transferable part:** the
check was anchored to a *run report*, which only exists when you ran the thing
yourself. So it vanished silently the moment a delivery was second-hand — which
is exactly the case where it matters. When I copied `analyze`'s five on their
behalf I had no run report and asserted their validity from digest and layout
alone; they happened to be clean. That is the argument for encoding this in the
copy tool rather than in anyone's discipline.

**It now is encoded.** `relayout_handoffs.py` (`a9a6ab5`) prints the verdicts
beside each copy and distinguishes the three outcomes by **exit code**, so a
caller can gate rather than read:

| exit | meaning |
|---|---|
| 0 | copied, intact, every verdict passes |
| 1 | a source or copy failed its digest — the delivery is **broken** |
| 2 | usage: no source, or no published versions |
| 3 | copied and intact, but **not valid** — a failing verdict, or none at all |

`--allow-refused` turns 3 into 0 for knowingly shipping a refused artefact, and
still prints the block. Test `!= 0` for a sound delivery. The codes are kept
distinct rather than collapsed because 1 and 3 are different problems: 1 means
the bytes are wrong, 3 means the bytes are right and what they record is a
failure.

### The first two, and why they are still separate

1. **Integrity** — `tree_digest(content)` against the manifest AND against the
   untouched run-store original. **`tree_digest` takes a bytes path *and returns
   bytes*, and both ends bite:**
   - handing it a `str`/`Path` fails with `TypeError: can't concat str to bytes`
     from inside its own recursion, which looks like a library bug and is not —
     use `os.fsencode(path)`;
   - comparing its **return value** directly to a manifest's hex string is
     always `False`, so a perfectly good tree reports MISMATCH. `analyze` got
     six false MISMATCHes this way. Call `.hex()`.

   The first form is loud and costs a minute. The second is silent and reports a
   wrong answer confidently, which is the more expensive of the two.
2. **Layout** — `analyze-demo`'s `store.py` resolving `versions()` / `kind_of()`
   / `content_dir()`, and `env_mgr.fs.layout.stage()` staging without exception.

`stage()` accepts a mode-damaged handoff and raises nothing, so passing it says
only that the layout is right and **nothing** about integrity. Both results are
recorded in `PROVENANCE.md`; nothing downstream verifies a consumed handoff, so
that line is the only assurance a consumer gets.

All seven of this module's handoffs: digest MATCH against manifest and against
the run store, all seven resolving, all seven staging.

## 19. `check_items` drift audit — clean for all seven kinds

A kind whose `items_schema` has `additionalProperties: false` and lists fewer
items than the producer writes makes `FilesystemStore.seal` refuse **every**
version, and the refusal is filed under `seal_refused`, which has no reader
outside tests — so it surfaces as a task that never finishes, not as an error.

Checked all seven kinds this package declares against the delivered content:

    handoff.content.check_items(content.load(D), content.content_type(ct), items_schema)
    -> deployment_baseline / deployment_profiled / aiperf_baseline /
       aiperf_profiled / torch_trace / kernel_table / profile_packup: no drift

Worth doing even for a package whose runs seal cleanly, because the defect only
bites kinds a run does not exercise. Here all seven were exercised, so this is
a stronger result than a clean run alone.

## 20. Two more hard-coded identifiers, now parameters (commit f4f920c)

Same defect class as the DSA/parser hoist, found by review rather than failure:

- **`container_name` -> `PD_CTR`.** `glm53_mix` (and derived `glm53_mix_etcd`)
  was written straight into `round.sh:30` and `replay.sh:31` — not even as a
  `${CTR:-...}` default. On a node where `docker ps` lists every tenant's
  containers, a name carrying neither owner nor module cannot be reclaimed
  mechanically under a rule like "you may only delete paths naming yourself",
  and two runs of this package on one node tear each other's engine down. The
  downstream scripts already read `${CTR:-glm53_mix}`; only the two bind points
  needed fixing.
- **`context_length` -> `PD_CTX`.** The flag a model substitution cannot survive
  by luck — see section 9. `round.sh` now threads `CTX` to `mix_up.sh` beside
  `DSA_ARGS` and `PARSER_ARGS`.

Verified without another full run: the package loads with both `--var`s, and
`round.sh` driven to its first precondition shows the binding works in both
directions — with `PD_CTR` set it passes the binding line and aborts on the
model check; without it, `line 30: PD_CTR: parameter null or not set`.

## 21. Where this module's files live

Per the user's 2026-09-02 instruction, working files belong under
`/shared_nfs/yihou/agent_sys/temp/profiling/`. Deliverables go to
`/shared_nfs/yihou/agent_sys/debugging/profiling/`, which is a drop, not a
working directory. Earlier runs of this module predate the instruction and were
left in place at `/shared_nfs/yihou/agent_sys_debug/ws2/` rather than moved,
because the instruction was explicitly forward-looking.

**This module claims no exception to keeping its run root on `/shared_nfs`, and
section 16 is why:** a later reader should not "helpfully" move it to local disk
to dodge the `TMPDIR` ROCm segfault. `profiling-demo` ran four complete graphs
with `--demo-root` on `/shared_nfs` — two engine bring-ups and four
torch-profiler captures each — with no segfault, because every kernel launch
happens inside a docker container with its own `/tmp`. The zone's `TMPDIR` never
reaches a process that touches a GPU. That fault is real for packages that run
kernels *in the zone*; this is not one of them.

## 22. Node-local paths do not exist on the login node

`tempfile.TemporaryDirectory(dir="/mnt/m2m_nobackup/yihou")` fails on the login
node with `FileNotFoundError: [Errno 2] No such file or directory`. `/mnt/m2m_nobackup`
is **per-node scratch**; the login node does not mount it. This bit me only
because I lifted a staging check that had run on the node and re-ran it on the
login node.

It is the same class as two other facts in these notes, and they are worth
holding together because each has cost someone time today:

| thing | scope |
|---|---|
| `/mnt/m2m_nobackup` | **per node** — absent on the login node |
| docker images | **per node** — a tag built on `-020` is not on `-079` |
| GPU compute partition (SPX vs CPX) | **per node** — `-079`/`-276` are SPX with 8 devices of 288 GiB; `-080` was CPX with 64 devices of 36 GiB, so "GPU 2" means device 2 on one and devices 16–23 on the other |
| `/shared_nfs`, `/home/yihou` | **cluster-wide** — same bytes on every node and the login node |

The rule that follows: **anything you assert about "the cluster" that came from
one node is a fact about that node until you check another.** Measure the node
you are on. `rocm-smi --showcomputepartition` answers the partition question in
one call and is worth running before binding a device index.
