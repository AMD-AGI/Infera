# The plan `astk scaffold` reads

A plan is the smallest description of a package; `scaffold` derives the rest
(`skillkit/scaffold.py` docstring is the authoritative field list).

```yaml
package: qwen_serve_bench            # a name for descriptions
description: Serve a model with sglang on one node and measure it under a fixed load.
goal: Serve the model and measure throughput and latency under a fixed load.   # <= 100 chars

tasks:                               # listing order = execution order (topological)
  - name: serve_and_bench
    kind: ai                         # ai: Claude follows the readme | program: entry runs
    goal: Bring up the server, prove it answers, run the load, publish the result.
    description: One AI task; the readme carries every step and trap from the history.
    inputs: []
    outputs: [bench_result]
    env:                             # optional; reaches this task's body only
      MODEL_PATH: '${model_path}'    # required --var at load time
      IMAGE: '${image:-lmsysorg/sglang:v0.5.16-rocm720-mi30x}'
    readme: |
      # serve_and_bench
      ... (written from references/readme_template.md) ...

kinds:
  - name: bench_result
    content_type: reproducible       # README: Purpose/How to run/Result/Environment/Watch out
    description: The load result with the command that produced it and the environment.
    items: []                        # extra required items beyond the content type's
    check:                           # args of the generic validator (templates/handoff_check.py)
      require_items: [result, env, command]
      json_files:
        'items/result/summary.json': [throughput_tok_s, ttft_ms.p50, requests_ok]
      numeric:
        - {file: 'items/result/summary.json', key: error_rate, max: 0.05,
           why: 'a zero bar is a flake detector; 5% matches check_bench_report'}
      executable: ['items/command']
      shell_parses: ['items/command']
      min_lines: {'README.md': 12}   # a floor ~1/3 of a real README: catches stubs only
```

## Two tasks, with an external input and a hand-written validator

```yaml
tasks:
  - name: build_kit            # program: a script from the history does it end to end
    kind: program
    goal: Build the deploy kit from the recorded recipe.
    inputs: [model_card]
    outputs: [deploy_kit]
    readme: |
      # build_kit
      Runs build_kit.sh, which renders the recipe into items/codes/.
    entry: |
      exec bash "$AGENT_SYS_TASK_PACKAGE/assets/build_kit.task/build_kit.sh"
  - name: tune
    kind: ai
    goal: Tune the kit's serving flags under the fixed load and publish the best.
    inputs: [deploy_kit]
    outputs: [tuned_kit]
    readme: |
      ...
kinds:
  - name: model_card           # produced by no task -> published from a fixture
    content_type: structured_text
    description: Which model, where its weights are, its parallelism.
    fixture: /abs/path/to/a/content/dir     # README.md + items/text.json
    check: {json_files: {'items/text.json': [model_path, tp]}}
  - name: deploy_kit
    content_type: code
    description: A runnable kit.
    check: {require_files: ['items/codes/deploy.sh'], executable: ['items/codes/*.sh']}
  - name: tuned_kit
    content_type: code
    description: The kit with the winning flags and the evidence.
    check: {require_files: ['items/codes/evidence/*.json']}
    validators:                # custom: the generic check cannot re-run anything
      - name: check_tuned_reproduces
        brief: Re-running the recorded load reproduces the claimed throughput within 10%
        dimension: trustworthiness
        strength: strong
        cost: gpu_hours
        logic_source: external_dynamic
        args: {tolerance: 0.10}
        check_py: /abs/path/to/check_tuned_reproduces.py
```

`scaffold` adds a `publish_inputs` program leaf that copies `model_card`'s
fixture into its output slot (validated on publication), wires
`build_kit` after it and `tune` after `build_kit`, and makes `tune` the end.

## §custom — a hand-written validator body

Same contract as every validator (`docs/primer.md` § A validator body).
Skeleton:

```python
#!/usr/bin/env python3
import json, sys, traceback
from pathlib import Path

ids = json.loads(Path("inputs.json").read_text())
args = json.loads(Path("args.json").read_text()) if Path("args.json").is_file() else {}
materials = json.loads(Path("materials.json").read_text())
verdict, report = {}, []
for hid in ids:
    try:
        root = Path(materials[hid])              # README.md + items/
        problems = []
        # ... decide; append "what is wrong — how to fix it" to problems ...
        verdict[hid] = not problems
        report += [f"{hid}: {'passed' if not problems else 'REFUSED'}"] + [f"  PROBLEM: {p}" for p in problems]
    except Exception:
        verdict[hid] = False
        report.append(f"{hid}: DID NOT RUN\n{traceback.format_exc()}")
Path("validator_report.txt").write_text("\n".join(report) + "\n")
Path("verdict.json").write_text(json.dumps(verdict))
```

Args that came from `${VAR}` arrive as strings — convert them. Always write
`verdict.json`; a body that writes none is "nothing decided", not a failure.

## Content-type cheat sheet for `check`

| content_type | typical `check` |
|---|---|
| reproducible | `require_items: [result, env, command]`, `executable`/`shell_parses: ['items/command']`, `json_files` on the result |
| code | `require_files` on the entry points (`items/codes/**/REPRODUCE.md`, `deploy.sh`), `min_lines` floors on README/REPRODUCE |
| structured_text | `json_files: {'items/text.json': [keys the consumer reads]}` |
| text | `min_lines: {'items/content': <floor>}` |
