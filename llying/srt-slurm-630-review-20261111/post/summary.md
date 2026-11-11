Thanks @nv-kmcgill53 and Devin for the reviews. All points are addressed in afefb95, on top of a merge with current `main` (#661), plus two small follow-ups to the load-time checks in 82e808f and 72c4d04. Replies are in the threads; in short:

- **KV events (Devin's blocker):** the decision now comes from `Backend.get_kv_events_config_for_mode()`, the same method the backend uses for `--kv-events-config`, so a role without `kv_events` gets `--no-enable-kv-events` and Infera's launcher no longer opens a port of its own.
- **Worker hooks:** `direct_worker_launcher()` is replaced by `worker_module()`, `worker_launcher_args()` and `worker_environment()`. The backend builds `python3 -m <module>` itself, and `INFERA_READINESS_PORT` moves from an `env` prefix into the worker environment.
- **Readiness:** an entry of `/v1/workers` without `status` or `disagg_mode` no longer counts; the tests pin v0.3.3's payload.
- **Load-time checks:** kv-aware routing (also `infera.server`'s default) with no role publishing KV events is rejected. The exception is `decode-input-demand`, which runs only under kv-aware and needs no events, so with it on, loading logs a warning instead. Different `INFERA_VERSION` values for the router and the workers, and a `grpc-mode` role, are rejected too.
- **Smaller items:** the install check no longer imports a private Infera helper; docs cover readiness semantics, the release pin, what the roles must agree on, and why SGLang only.

Validation on afefb95:

- **Checks and tests:** `ruff check`, `ruff format --check`, `ty check`, `srtctl schema-docs --check` and `mkdocs build --strict` pass, and the regenerated launch snapshot changes only `sglang__infera-disagg.txt`. The infera, frontend, design-rule, launch-snapshot and backend tests pass (222). The full suite has 5529 passed and 3 failed, in `test_dry_run.py` (2) and `test_schema_docs.py` (1): on our nodes rich wraps output at 80 columns and splits the asserted paths. The same 3 fail on the previous head (5bf86633), so they come from the environment, not this change.
- **Hardware smoke (AMD MI355X, Slurm + Pyxis, 2 nodes):** the example recipe adapted to the cluster (`gpu_type: mi355x`, stock `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`, Mooncake over the ionic rails), Qwen3-0.6B, amd-infera 0.3.3 installed by `install-infera.sh`, srtctl at afefb95. Prefill and decode on separate nodes, both registered, and `/v1/workers` readiness passed.
  - The decode registered with `kv_events_endpoint: null`, and Infera spawned its SGLang without `--kv-events-config`.
  - `INFERA_READINESS_PORT` showed in the per-worker `Env:` lines, and the launchers opened readiness on the allocated 7500 and 7501.
  - The router subscribed only to the prefill's allocated `tcp://<prefill>:3000`.
  - Three chat requests returned correct answers.
- **Production recipes:** our GLM-5.2 MXFP4 recipes (4+4, 1P1D, and 2P1D with two prefills on one node) load and render through the mock orchestrator with afefb95. Decode gets `--no-enable-kv-events`, and the colocated prefills get distinct readiness ports (7500 / 7501) and KV-events ports (3000 / 3001).

The two follow-ups change only `validate()`, its tests and docs/infera.md, not the commands or ports. `ruff check` and `ruff format --check` pass on them; their test run is queued on our cluster, and I'll post here if anything fails.

On the two other points in the review summary:

- **Snapshot addresses:** right, the `127.0.0.1` etcd and advertise addresses come from the mock orchestrator. Real addresses come from `get_hostname_ip`; the hardware run above had etcd and the workers on the nodes' own IPs.
- **`get_hostname_ip` per worker command:** it is the cached `get_hostname_ip` (`@functools.cache`), and #651 is in this branch.

@nv-kmcgill53, could you take another look when you have time? The CI and Copyright workflows for 72c4d04 are waiting for approval again (fork PR); could a maintainer approve them? Thanks!
