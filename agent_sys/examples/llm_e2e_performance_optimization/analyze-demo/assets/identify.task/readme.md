# identify — locate each operator's framework-level entry point

## What it does

For each selected kernel, answer four questions: which repository owns it, what
language it is written in, which files carry it, and which functions are the
editable entry points.

Three resolution levels, tried in order, and the level reached is recorded:

| level | method | evidence |
|---|---|---|
| 1 | `trace_python_stack` | the input table carried a `launcher` block from a `with_stack: true` profile |
| 2 | `symbol_search` | the compound name of the symbol was found in an indexed repository |
| 3 | `agent_recovered` | neither did; direction plus a hint, entry function read from source by `build_workset` |

## Why level 3 is not a failure

The unit of a workset is a **framework-level callable**, not a device kernel
symbol. A Triton or TileLang device symbol is a compilation artefact — there is
no source file to point at, and the editable object is the Python function that
generated it.

AgentKernelArena's own production tasks say this in their own words. Its KDA
task records:

> KDA is Triton-JIT so the trace shows launcher = Not found; the entry points
> below were recovered from the session call stacks.

Its TileLang task points `source_file_path` at `tilelang.py` and
`target_kernel_functions` at `mhc_fused_post_pre_tilelang`, never mentioning the
generated device symbol.

So `main_kernel` is not something this step failed to resolve. It is something
that resolves one level up, and `build_workset` is where the source gets read.

## What it will not do

Report a path it did not verify. `check_identity_resolved` is a
*trustworthiness* validator because the failure that matters here is a plausible
wrong answer: a resolver that names a source file it never opened sends the
whole downstream chain at the wrong code. A claimed path that does not exist
fails the step.

## Watch out

**No host path reaches the handoff.** `image_repo_path` is a path *inside the
serving container* — `/sgl-workspace/aiter` and the like — which is portable and
is what forge-loop needs anyway. Where a repository happens to be checked out on
this machine is deliberately not recorded: `handoff/locality.py` refuses to seal
it, and it would be wrong on any other machine.

`amd_kernel_finder`'s built-in repository list covers `rocm-libraries`, `triton`,
`rocm-systems`, `aiter`, `vllm` and `pytorch` — **not sglang**. Supply a checkout
with `--var sglang_src=...` to index it. With no repositories supplied the
finder runs with `auto_clone=False` and every operator lands at level 3, which
is a supported outcome; auto-cloning tens of gigabytes inside a task attempt is
not something this step does silently.
