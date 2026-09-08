# VAR-TABLE — every `--var` this package accepts, audited for THIS host

`smci355-ccs-aus-n04-25`, slurm job 29184, 8×MI355X **SPX** (measured:
`rocm-smi --showcomputepartition` enumerates GPU[0..7]), model
`/apps/data/models/Qwen3-32B`, transport `local`, node_ip `10.235.192.131`.
Derived 2026-09-06 from `shared.yaml` + `steps/*.yaml` — the only two places the
package declares variables — by a script that **skips comment lines**, because
four "variables" in this package exist only inside prose:

> `${mode}`, `${result_type}`, `${packup_skill}`, and the bare `${tp}` are **not
> variables**. They appear only in comments (`m2_profiling.yaml:171`,
> `m5_integration.yaml:457`, `m1_deploy.yaml:159`, `m2_profiling.yaml:112`).
> Four more look like they have two defaults and do not, for the same reason:
> `expect_ranks`, `integration_min_requests`, `served_name` and `measure_gpu`
> each have exactly one real default and one that is a comment quoting a
> spelling the author is *rejecting*. A grep hit is not evidence; the hit has to
> be read.

**101 declared variables.** Everything below is grouped by what it costs you.

---

## A. Blockers — no default, or a default that does not exist on this host

Each of these fails *after* something expensive, which is why they are first.

| var | declared default | why it is broken here | what to do |
|---|---|---|---|
| ~~`aiperf_image`~~ | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | ~~absent~~ — **RESOLVED, see the changelog below** | nothing to pass; the default is now satisfiable |
| `aiperf_trace` | *(empty)* → `${E2E_AIPERF_TRACE:?}` | first-cluster path; `/shared_nfs` is **empty** here | `--var aiperf_trace=/data/yihou/e2e_verify_20260906/m2/materials/conversation_trace.jsonl` — produced and verified, see that dir's README |
| `gsm8k_data` | *(empty)* → `${E2E_GSM8K_DATA:?}` | same | `--var gsm8k_data=/data/yihou/e2e_verify_20260906/m2/materials/gsm8k_test.jsonl` — 1319 rows, verified against `GSM8KEval` |
| `work_root` | `/mnt/m2m_nobackup/yihou/e2e_flow` | **`/mnt/m2m_nobackup` does not exist on this host** | `--var work_root=/data/yihou/e2e_verify_20260906/<line>/work` |
| `scratch_root` | `/mnt/m2m_nobackup/yihou/e2e_flow/kfo` | same mount, **and it does not follow `work_root`** — this is the pair recorded as bug 30, where a fallback moved one and not the other | set it explicitly whenever you move `work_root` |
| `validate_work_root` | `/mnt/m2m_nobackup/yihou/e2e_flow/validate` | same mount, third member of the same family | set it explicitly |
| `mock_root` | `/shared_nfs/yihou/agent_sys/cheat_for_mock` | **does not exist here.** There is no sealed corpus on this cluster | **any run with `mock_stages` set will fail.** `mock_stages` defaults to `all`, so a bare launch is already in this state |
| `magpie_root` | `/shared_nfs/chaox/Magpie` | does not exist here | m3's problem; flagging it so it is not discovered mid-run |
| `measure_gpu` | *(empty)* → `${E2E_MEASURE_GPU:?}` | no default and no source; a card must be named | m3/m4 |
| `image` | **none** | | the servable image on this node |
| `jobid`, `node`, `node_ip` | **none** | | `29184`, `smci355-ccs-aus-n04-25`, `10.235.192.131` |
| `model_name`, `model_path` | **none** | | `Qwen/Qwen3-32B` (or the name the engine serves), `/apps/data/models/Qwen3-32B` |

**Present and fine:** `etcd_image` (`quay.io/coreos/etcd:v3.5.14` **is** local),
ports `8101`/`8102`/`8103`/`8140` all free (`ss -ltn`).

### Changelog for §A — because a table that is edited silently cannot be trusted

**2026-09-06, `aiperf_image` struck out.** It was listed as a blocker on a
correct measurement (156 local images, zero matching) that the leader's
`docker pull` at 06:34Z had already overtaken — our clocks raced. Verified
first-hand after the fact, not taken on report:
`docker inspect` returns the image; `aiperf --version` → `0.12.0`.
**Nothing else in this section changed: no other row edited, no default
re-graded, no count touched.**

**And the follow-on caution turned out not to apply to this package.** The
image's default entrypoint is **not** `aiperf` — measured,
`Entrypoint: ["/bin/bash","-c"]`, `Cmd: null` — which is why a bare
`--version` needs `--entrypoint aiperf`. **All three call sites already assume
exactly that** and pass the whole invocation as one string:

```
assets/bench/aiperf_replay.sh:117    "$AIPERF_IMAGE" "$CMD"
assets/bench/aiperf_synthetic.sh:126 "$AIPERF_IMAGE" "$CMD"
assets/load/aiperf_replay.sh:119     "$AIPERF_IMAGE" "$CMD"
```

and each says so in a header comment: *"Passing it as argv silently runs
`aiperf` with the rest as positional parameters instead of flags."*
**So the hazard here is the reverse of the one to look for: the call sites are
right, and the failure mode is somebody "fixing" them to pass argv.**
`User: nvs` (uid 1000) and `WorkingDir: /app` likewise match the `--user` and
`HOME=/tmp` the call sites already set.

---

## B. Defaults that are wrong for *this host* even though they resolve

| var | default | this host | why |
|---|---|---|---|
| `context_length` | `262144` | **≤ 40960** | `Qwen3-32B/config.json: max_position_embeddings = 40960`. A bring-up asking for 262144 does not start |
| `transport` | `auto` | **`local`** | must be passed explicitly. On the first cluster omitting it cost m3 a run two hours in: m1 mints the resolved value into the record and `_agree_or_die` refuses later, *as a message about transports* |
| `tp` | `8` | **`4` — pass it; the default is wrong for this run** | 8 GPUs are present (SPX), but m1 brought up on **four**: `--var tp=4 --var gpu_devices=0,1,2,3`, cards 0-3 at 3 % VRAM and 4-7 at 0 %. **The hardware count is not the TP.** See the §B changelog |
| `trace_end_ms` | **`180000` in `shared.yaml:149`, `60000` in `m5_integration.yaml:126`** | pick one and pass it | **one variable, two different defaults.** m2's replay window and m5's bench window silently differ by 3×. Nothing warns |
| `needle_tokens` | `31000` | fits under 40960, but leaves 9960 for everything else | m5 |

### Changelog for §B/§C — 2026-09-06, `tp` and `expect_ranks` corrected

**I had `tp` as "`8` — correct, omit" and `expect_ranks` as "default 8 is
right". Both were wrong, and the error was mine to make**: I measured that this
host has **8 GPUs in SPX** and let that stand in for the deployment's TP. It is
not the same fact. m1 chose **TP 4** (matching the first cluster for
comparability, and faster to load), and the cards agree — 0-3 at 3 % VRAM, 4-7
at 0 %.

> **A hardware count is not a parallelism degree. `rocm-smi` answered the
> question I asked it, and I asked the wrong one** — the same shape as every
> other instrument note in this file.

**Changed:** the `tp` row in §B and the `expect_ranks` row in §C, both now
naming `4`. **Not changed:** no other row, no count, no default re-graded, and
the §A changelog above is untouched.

### The one with no knob at all

`E2E_EVAL_THINKING` has **no `--var`**. `measure.sh:193` hardcodes the fallback
`--thinking-mode glm-45`, and `lm_eval.sh`'s own comment says the mismatch
"fails as a NUMBER rather than an error" — *"Without the matching
chat_template_kwargs a healthy deployment scores 0.00."* This host serves
**Qwen3**, not GLM. There is no launch-line fix; it needs an edit to
`m5_integration.yaml` to expose the variable (or `none`, which `lm_eval.sh:47`
accepts and turns into an empty flag). **Reporting, not fixing — m5's call.**

---

## C. The mock/real class — one value when the consuming stage is mocked, another when it is real

This is the class that cost the first round three launches, one variable at a
time. **Read every row against your own `mock_stages`; do not fix the one that
just refused.**

| var | mocked stage | real stage | on this host |
|---|---|---|---|
| `expect_ranks` | the **sealed capture's** TP | the **deployment's** TP | **`4`, and all three candidate values are different on this run** — declared default `8`, mock value `2` (which is what m1's recorded launch line carries), actual TP `4`. It does **not** track `${tp}`, deliberately (`m2_profiling.yaml:112-118`). Full reasoning in `PRE-REGISTER-m2.md` §1 |
| `adhoc_cases` | `0` — a mock cannot produce ad-hoc cases | `3` (the default) | **omit** when m5 is real; pass `0` when it is mocked. Note the direction: the *default is the real value*, so the mistake here is passing `0` into a real run |
| `bench_rounds` | must match the replay's round count | `1` | m5 |
| `image` | the **sealed kit's** tag | the **real bring-up's** | no sealed kit here, so: the node's |
| `m<N>_agent` | `runner` for every still-mocked stage | **absent** for the promoted stage | `m3_agent` and `m5_agent` each have two declared defaults (`runner`/`workset_builder`, `runner`/`e2e_integrator`) — that is the mock/real split, not a conflict |
| `integration_min_requests` | n/a | omit — default `50` | m2/m5 |
| `workset_operator` | may be empty | **required from the rung that makes m3 real** | m3/m4 |
| `forge_fellow` | omit | omit | m3; `run_forge.sh` exits 2 and names it, but only after a campaign fails to start |

**`mock_stages` itself is the head of this class and its default is `all`.** With
no corpus on this cluster, **the only launches that can work here set
`mock_stages` to nothing at all** — i.e. every stage real, or the stages you are
not running simply absent from the graph.

---

## D. Full enumeration — all 101 declared variables

Machine-derived; the "stages" column is which yaml files reference it, not which
bodies read it. `**none**` = no default at all.

| var | default | stages | note |
|---|---|---|---|
| `adhoc_cases` | `3` | m5 | |
| `aiperf_image` | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | shared,m1,m2,m5 | |
| `aiperf_trace` | *(empty)* | shared,m2,m5 | |
| `bench_rounds` | `1` | m5 | |
| `block_size` | `512` | shared,m5 | |
| `container` | `yihou_e2e_flow` | shared,m1,m2,m5 | |
| `context_length` | `262144` | shared,m1,m2,m5 | |
| `deploy_bringup_timeout_seconds` | `3600` | m1 | |
| `deploy_entrypoint` | `scripts/deploy.sh` | m1 | |
| `deploy_load_conc` | `16` | m1 | |
| `deploy_load_in` | `1024` | m1 | |
| `deploy_load_out` | `1024` | m1 | |
| `deploy_load_seconds` | `180` | m1 | |
| `deploy_load_slack_seconds` | `900` | m1 | |
| `deploy_mode` | `mix` | shared,m1,m2,m5 | |
| `deploy_teardown_timeout_seconds` | `600` | m1 | |
| `deploy_timeout_seconds` | `5400` | m1 | |
| `dsa_args` | `none` | shared,m1,m5 | |
| `etcd_image` | `quay.io/coreos/etcd:v3.5.14` | shared,m1,m5 | |
| `eval_examples` | `100` | m5 | |
| `eval_max_tokens` | `2048` | m5 | |
| `eval_names` | `gsm8k mixed_prefix_gsm8k` | m5 | |
| `eval_threads` | `32` | m5 | |
| `expect_ranks` | `8` | m2 | |
| `forge_fellow` | *(empty)* | m3 | |
| `forge_max_hours` | `3.0` | m4 | |
| `forge_mock` | `0` | m4 | |
| `forge_model` | `Claude-Sonnet-5[1m]` | m4 | |
| `forge_snr_threshold` | `30.0` | m4 | |
| `gpu` | *(empty)* | m4 | |
| `gpu_devices` | `none` | shared,m1,m2 | |
| `gsm8k_data` | *(empty)* | m5 | |
| `image` | **none** | shared,m1,m2,m5 | |
| `impl_flag` | `--impl` | m4 | |
| `instruction` | `Bring this model up in the requested mode on this host, prove it answers through the router, and hand back a kit somebody else can run.` | shared,m1 | |
| `integration_min_requests` | `50` | m5 | |
| `jobid` | **none** | shared,m1,m2,m3,m4,m5 | |
| `kernel_table_min_launchers` | `10` | common | |
| `kernel_table_min_rows` | `20` | common | |
| `kernelforge_repo` | *(empty)* | m4 | |
| `kfo_python` | `/opt/venv/bin/python3` | m4 | |
| `launcher_top_n` | `50` | shared | |
| `m1_agent` | `e2e_deployer` | m1 | |
| `m2_agent` | `runner` | m2 | |
| `m3_agent` | `runner` / `workset_builder` | m3 | |
| `m4_agent` | `e2e_kernel_optimizer` | m4 | |
| `m5_agent` | `runner` / `e2e_integrator` | m5 | |
| `magpie_root` | `/shared_nfs/chaox/Magpie` | shared,m3 | |
| `max_conc` | `32` | shared,m5 | |
| `max_error_rate` | `0.05` | m5 | |
| `measure_container` | *(empty)* | shared,m3 | |
| `measure_gpu` | *(empty)* | shared,m3,m4 | |
| `min_requests` | `50` | m2 | |
| `min_resolve_ratio` | `0.0` | m3 | |
| `mock_premise` | `matched` | shared,m4 | |
| `mock_report` | `refused` | shared,m5 | |
| `mock_root` | `/shared_nfs/yihou/agent_sys/cheat_for_mock` | shared,m1,m2,m3,m5 | |
| `mock_stages` | `all` | shared,m1,m2,m5 | |
| `model_name` | **none** | shared,m1,m2,m5 | |
| `model_path` | **none** | shared,m1,m2,m5 | |
| `needle_depths` | `0.02,0.5,0.98` | m5 | |
| `needle_frontier_tokens` | `0` | m5 | |
| `needle_tokens` | `31000` | m5 | |
| `node` | **none** | shared,m1,m2,m3,m4,m5 | |
| `node_ip` | **none** | shared,m1,m2,m3,m5 | |
| `package_commit` | `unknown` | shared,m3 | |
| `parser_args` | `none` | shared,m1,m5 | |
| `port_etcd` | `8103` | shared,m1,m2,m5 | |
| `port_router` | `8101` | shared,m1,m2,m3,m5 | |
| `port_worker` | `8102` | shared,m1,m2,m5 | |
| `remote_home` | *(empty)* | shared,m1,m2,m3,m5 | |
| `report_flag` | `--json` | m4 | |
| `req_timeout` | `900` | shared,m5 | |
| `require_runtime_marker` | `true` | m5 | |
| `scratch_root` | `/mnt/m2m_nobackup/yihou/e2e_flow/kfo` | m4 | |
| `served_name` | *(empty)* | shared,m1,m2,m5 | |
| `snr_threshold` | `30.0` | shared,m3 | |
| `stack_ranks` | `2` | shared,m2 | |
| `stack_window_s` | `3` | shared,m2 | |
| `stage` | *(empty)* | shared,m1,m2,m4 | |
| `stock_vs_m2_tolerance` | `0.10` | m5 | |
| `teardown_entrypoint` | `scripts/teardown.sh` | m1 | |
| `top_n` | `25` | shared | |
| `tp` | `8` | shared,m1,m2,m5 | |
| `trace_end_ms` | `180000` / `60000` | shared,m5 | |
| `trace_min_kernels` | `1000` | m2 | |
| `transport` | `auto` | shared,m1,m2,m3,m4,m5 | |
| `transport_env` | *(empty)* | m1,m3,m4 | |
| `transport_path` | `/usr/local/bin:/usr/local/sbin` | m1,m3,m4 | |
| `validate_port_base` | `8140` | m1 | |
| `validate_work_root` | `/mnt/m2m_nobackup/yihou/e2e_flow/validate` | m1 | |
| `verify_ranks` | `1` | m2 | |
| `warmup_s` | `60` | shared,m2 | |
| `window_s` | `10` | shared,m2 | |
| `work_root` | `/mnt/m2m_nobackup/yihou/e2e_flow` | shared,m1,m2,m4,m5 | |
| `workers_max` | `8` | shared,m5 | |
| `workset_id` | `workset` | shared,m3 | |
| `workset_max_rsd` | `0.10` | m3 | |
| `workset_min_pass_ratio` | `0.5` | m3 | |
| `workset_operator` | *(empty)* | shared,m4 | |
| `workset_reverify_shapes` | `1` | m3 | |

---

## E. What this audit does NOT establish

- It says which variables are **declared**, from the two files that declare
  them. It does **not** prove every declared variable reaches a body. **A row in
  this table is not evidence that the flag works.** Checking that per-variable
  is a separate, larger job — one variable was checked, below.

### `gpu_devices`, traced end to end 2026-09-06 — inert on one path, plumbed on the other

The first cluster spent five launches on this flag. Here it splits by path:

**m5's serve path — INERT, and unoverridable from the launch line.** Not one
line but an **omission**, which is why grepping `GPUS` does not find it:

```
mix_up.sh:106       docker run … --device=/dev/kfd --device=/dev/dri     (all cards visible)
                    …and no  -e GPUS  at container creation
mix_up.sh:157-165   docker exec -d "$CTR" env MY_IP=… TP="$TP" … bash /mix_worker.sh
                    ^ a CLOSED, explicit env list. GPUS is NOT in it.
mix_worker.sh:26    GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"    -> always the default
mix_worker.sh:100   HIP_VISIBLE_DEVICES="$GPUS"
```

The one assumption in that chain was checked rather than assumed: the engine
image defines **no** `GPUS`, `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`
(`docker inspect -f '{{json .Config.Env}}'`). So `GPUS` is unset inside the
container and the default stands. **An m5 arm takes `0..TP-1`, always.**

> **On this host that is `0..7` — all eight cards — which is what anyone would
> have asked for. The inert answer and the correct answer are the same numbers
> again, exactly as on the first cluster.** The consequence worth carrying: an
> m5 arm cannot be co-tenanted on this node, whatever the launch line says.

**m2's load path — PLUMBED.** `line.sh` runs *the kit's* `scripts/deploy.sh`,
not `mix_up.sh`, and maps the variable explicitly with a documented three-branch
precedence (`line.sh:169-184`), announcing which branch it took in the log. The
remaining half — whether the kit's own `deploy.sh` honours
`E2E_KIT_GPU_DEVICES` — is a property of **m1's kit, which does not exist yet**,
so it is unverified rather than verified. `line.sh:154` notes the kit's default
is the hardcoded literal `0,1,2,3` and that *nothing in the kit reads a card*.
- It grades defaults against **this host's facts as of 2026-09-06**. The mounts,
  the image list and the free ports were read today and will expire.
- The mock/real column in §C is **the set someone has already been bitten by**,
  carried over from `RUN-PLAN.md`, plus what today's audit added. It is not a
  proof that the remaining 90-odd variables are mock-invariant.

---

## CORRECTION 2026-09-06T22:13:50Z — this table missed `gpu`, and run 8's m4 died on it

**`gpu` is a `--var`, its default is empty, and an empty value aborts.**

```
steps/m4_kernel_opt.yaml:359                 HIP_VISIBLE_DEVICES: '${gpu:-}'
optimize_kernel.task/steps/run_in_container.sh:136   [ -n "$HIP_VISIBLE_DEVICES" ] || exit 1
```

Run 8, 22:04:08, `m4/20260906T220149/logs/step4_attempt.log`:

```
run_in_container: HIP_VISIBLE_DEVICES is empty and this host is shared.
  Pass --var gpu=<n>.
```

**Why I missed it: the line carries `measure_gpu=4`.** Two variables, two
consumers, similar names — **having one of the pair made the other look
covered.** Identical to the `remote_home` / `transport_env` pair three hours
earlier, where one of two needed variables was present and the other was not.

### The whole empty-default class, swept

Every `${name:-}` in `steps/*.yaml` + `shared.yaml` absent from the launch line:

| var | status |
|---|---|
| `gpu` | **ABORTS — demonstrated by run 8** |
| `kernelforge_repo` | real campaign only; `forge_mock=1` skips it |
| `forge_fellow` | same |
| `measure_container` | auto-named in practice (`yihou_m3_measure_<pid>`, run 8) |
| `served_name` | documented: empty means `$E2E_MODEL_NAME` |
| `stage` | no abort found |
| `workset_operator` | no abort found; m4 pinned gemm without it |

**Only `gpu` has a demonstrated abort. The other six are un-refuted, not proven
harmless.** The check was a grep for die/exit on their env keys, which is weaker
than running them.

### The command that regenerates this sweep

```sh
cd <package>
grep -rhoE "\\\$\{([a-z_0-9]+):-\}" steps/*.yaml shared.yaml | sed 's/[${}]//g; s/:-//' | sort -u > /tmp/empty.txt
awk '/^```sh$/{f=1;next} /^```$/{f=0} f' LAUNCH-CHAIN.md | grep -oE '\-\-var [a-z_0-9]+' | sed 's/--var //' | sort -u > /tmp/have.txt
comm -23 /tmp/empty.txt /tmp/have.txt
```

**Run this before every launch.** It is two commands and it would have caught
this one; reading the table again would not have, because the table was wrong.
