# ATOM playbook

Partial: only the facts below are known so far; the sections of `README.md` are not written yet. A standard profile
on ATOM starts from these facts and notes in its report what it had to work out itself (trace window control, step
bench, kernel breakdown), so that an onboarding can write the sections later.

Known so far:

- The torch profiler is switched on only by the CLI `--torch-profiler-dir` (`ATOM_TORCH_PROFILER_DIR` is overridden);
  stop blocks the API synchronously; turning it on for a PP prefill crashes the engine (heap corruption).
- Trace directories are `pp{k}_rank_0` / `rank_{r}`; a step with MTP K3 has three graph launches; phases are cut on
  the `ATOM::forward`, `decode[...]`, `propose_eagle[i/K]` labels.
- Metrics: `atom:decode_batch_size`, `atom:gpu_forward_seconds`, `atom:decode_context_tokens`; workers are not
  federated: scrape each worker's `/metrics`.
