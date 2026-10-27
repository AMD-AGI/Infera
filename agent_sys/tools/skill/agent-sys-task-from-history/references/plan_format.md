# The plan `astk scaffold` reads

A plan is the smallest description of a package; `scaffold`
(`lib/skillkit/scaffold.py`) derives the rest. Every field it reads:

```yaml
package: qwen_serve_bench            # a name for descriptions
description: Serve a model with sglang on one node and measure it under a fixed load.
goal: Serve the model and measure throughput and latency under a fixed load.   # <= 100 chars

tasks:                               # listing order = execution order (see § one sink)
  - name: serve_and_bench
    kind: ai                         # ai: Claude follows the readme | program: `entry` runs
    goal: Bring up the server, prove it answers, run the load, publish the result.  # <= 100 chars
    description: One AI task; the readme carries every step and trap from the history.
    inputs: []                       # kinds read
    outputs: [bench_result]          # kinds written (at least one)
    froms: []                        # optional ordering-only edges to EARLIER tasks
    env:                             # optional; reaches this task's body only
      MODEL_PATH: '${model_path}'    # required --var at load time
      IMAGE: '${image:-lmsysorg/sglang:v0.5.16-rocm720-mi30x}'
    resources: {gpu: 1}              # optional
    readme: |                        # ai: the prompt | program: what it does
      # serve_and_bench
      ... (written from references/readme_template.md) ...
    # entry: |                       # program only: POSIX sh body (`set -eu` is added)

kinds:
  - name: bench_result
    content_type: reproducible       # reproducible | code | structured_text | text
    description: The load result with the command that produced it and the environment.
    items: []                        # extra required items beyond the content type's
    check:                           # args of the generic validator (templates/handoff_check.py docstring)
      require_items: [result, env, command]
      json_files:
        'items/result/summary.json': [throughput_tok_s, ttft_ms.p50, requests_ok]
      numeric:
        - {file: 'items/result/summary.json', key: error_rate, max: 0.05,
           why: 'a zero bar is a flake detector; 5% matches check_bench_report'}
      executable: ['items/command']
      shell_parses: ['items/command']
      min_lines: {'README.md': 12}   # a floor ~1/3 of a real README: catches stubs only
    calibration: |                   # why each bar is where it is, measured on the real result;
      ...                            # rendered into the validator readme, survives --force
    # fixture: /abs/content/dir      # only for an input no task produces
    # validators: [...]              # hand-written ones, in addition (§ custom)
```

## One sink

The last task is the subgraph's one end (`is_end`): agent_sys finishes `main`
when it completes and exports only its outputs. So every output of every other
task must be an input of a later task; `froms` orders tasks but exports
nothing, so it does not count. `scaffold` refuses a plan that breaks this.

## Two tasks, with an external input and a hand-written validator

```yaml
tasks:
  - name: build_kit            # program: a script from the history does it end to end
    kind: program
    goal: Build the deploy kit from the recorded recipe.
    inputs: [model_card]
    outputs: [deploy_kit]
    readme: "# build_kit\nRuns build_kit.sh, which renders the recipe into items/codes/.\n"
    entry: exec bash "$AGENT_SYS_TASK_PACKAGE/assets/build_kit.task/build_kit.sh"
  - {name: tune, kind: ai, inputs: [deploy_kit], outputs: [tuned_kit], readme: ...,
     goal: Tune the kit's serving flags under the fixed load and publish the best.}
kinds:
  - name: model_card           # produced by no task -> published from a fixture
    content_type: structured_text
    fixture: /abs/path/to/a/content/dir     # README.md + items/text.json
    check: {json_files: {'items/text.json': [model_path, tp]}}
  - name: deploy_kit
    content_type: code
    check: {require_files: ['items/codes/deploy.sh'], executable: ['items/codes/*.sh']}
  - name: tuned_kit
    content_type: code
    check: {require_files: ['items/codes/evidence/*.json']}
    validators:
      - name: check_tuned_reproduces
        brief: Re-running the recorded load reproduces the claimed throughput within 10%
        dimension: trustworthiness            # completeness | usability | trustworthiness
        strength: strong                      # strong | long_term_strong | weak
        cost: gpu_hours                       # seconds | minutes (default) | gpu_hours
        logic_source: external_dynamic        # external_static | external_dynamic (default)
        args: {tolerance: 0.10}
        check_py: /abs/path/to/check_tuned_reproduces.py   # copied to assets/<name>.validator/
        calibration: ...
```

`scaffold` adds a `publish_inputs` program leaf that copies `model_card`'s
fixture into its output slot (validated on publication), wires `build_kit`
after it and `tune` after `build_kit`, and makes `tune` the end.

## Custom — a hand-written validator body

Same contract as every validator (`docs/primer.md` § A validator body); model
`check.py` on `main()` of `lib/skillkit/templates/handoff_check.py`. Read
`inputs.json`, `args.json` (values from `${VAR}` arrive as strings — convert
them) and `materials.json`; write `verdict.json` `{id: bool}` **always**, even
when the check crashes (no file is "nothing decided", not a failure); write
reasons to `validator_report.txt` as "what is wrong — how to fix it".

## Typical `check` per content type

| content_type | typical `check` |
|---|---|
| reproducible | `require_items: [result, env, command]`, `executable`/`shell_parses: ['items/command']`, `json_files` on the result |
| code | `require_files` on the entry points (`items/codes/**/REPRODUCE.md`, `deploy.sh`), `min_lines` floors on README/REPRODUCE |
| structured_text | `json_files: {'items/text.json': [keys the consumer reads]}` |
| text | `min_lines: {'items/content': <floor>}` |
