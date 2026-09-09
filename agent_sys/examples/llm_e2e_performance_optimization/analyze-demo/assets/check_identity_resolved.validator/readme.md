# check_identity_resolved — trustworthiness, strong

## Why trustworthiness rather than completeness

The failure that matters here is a **plausible wrong answer**. A resolver that
names a source file it never opened sends every downstream step at the wrong
code, and the mistake survives all the way to a forge-loop run that optimizes
something nobody asked about. A missing field is loud; a confident wrong path is
not.

So every rule below asks whether a claim was checked, not whether a field is
populated.

## What it checks

1. Every operator carries a resolution method from the known set.
2. An operator claiming `trace_python_stack` or `name_grep` names at least one
   source file. One claiming `agent_recovered` names a non-empty hint instead.
   Neither may be silently empty.
3. Source paths are repository-relative. An absolute path is either a host path
   — which the seal refuses anyway — or a container root in the wrong field.
4. `image_repo_path`, when set, starts with an allow-listed container prefix.
5. Every operator carries at least one workload case.
6. The share resolved by evidence clears `min_resolve_ratio`.

## Why rule 6 is a ratio

Triton-JIT and TileLang operators reach `agent_recovered` **by construction**:
their device symbols are compilation artefacts with no source file. Two of
AgentKernelArena's own production tasks live there. Demanding a ratio of 1.0
would fail on a correct answer.
