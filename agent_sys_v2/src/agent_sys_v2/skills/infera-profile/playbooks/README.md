# Engine playbooks

One file per engine, `<engine>.md` (atom, sglang, vllm). A profile task in standard mode reads the one of its
project's engine; an onboarding task writes it. Sections, in this order:

1. **Control plane**: how to start and stop a trace on a live server (URL, body, roles), what blocks or crashes,
   how to tell the trace is on disk.
2. **Adapter**: the metric names and log lines for batch size, step time, queues (waiting / transfer / prealloc),
   KV use, prefix hits, MTP acceptance; how often they are emitted.
3. **Trace layout**: directory and file names per rank and role; how a step is cut (graph launches, ProfilerStep,
   engine labels); verify / draft phases with MTP.
4. **Kernel taxonomy**: the regular expressions of moe / attention / gemm / comm / norm / copy for this engine and
   model family.
5. **Benches**: the decode step bench (fixed batch) and the prefill bench for this engine, as commands.
6. **Known traps**, each with its symptom and the way around it.
7. **Provenance**: which project or onboarding task this came from, and when it was last checked.

| Engine | Playbook | State |
|---|---|---|
| ATOM | `atom.md` | partial: the known facts only, the sections still to be written |
| SGLang | — | none yet: the first SGLang project runs an onboarding |
| vLLM | — | none yet: the first vLLM project runs an onboarding |
