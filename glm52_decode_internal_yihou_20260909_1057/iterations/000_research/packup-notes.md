# Interpretation, wrong turns and limits

## What the result does and does not show

The 2026-09-08 task built/started an aggregated TP8/EP1/DP1 GLM-5.2-MXFP4 server, completed real-verification curl/chat/full GSM8K, and measured one c1 AgentX point with acceptance simulation3.61. Correctness and performance are **separate phases**. GSM8K was1277/1319=0.9681576952236542 for both extraction filters. The simulated performance number is2184.42030125 total(input+output)tokens/s/chip, not2184 output tokens/s/chip (output is16.41346625).

The approximately+0.54% total-throughput difference and approximately+0.26% interactivity difference against2026-09-07 are observations only. There is one run per condition, no variance estimate, no predefined equivalence tolerance, and no evidence that identical model-directory names mean identical weight files.

## Time and historical record

`history/work.checkpoint.summary.md` is preserved byte-for-byte, all **14** checkpoints and the final correction. Its historical claims are not silently rewritten or all endorsed. Main work occurred2026-09-08; full eval artifact is stamped09:37:13.323592, performance completion was around11:16Z and collection around11:22–11:25Z. Correctness artifact elapsed time2917.369245s is a monotonic duration, not an epoch timestamp. Source mtimes and original names are in `provenance/source-files.json`.

The recorded task start marker07:58:51Z differs from the builder's07:38Z narrative. Checkpoint1 has a disputed earlier SGLang pin; Dockerfile/build logs establish402df1e1 instead. The checkpoint gap and later stale state files are preserved, not filled with invented progress. Use raw artifacts over prose when they disagree.

Cancellation history is explicitly uncertain. The log reports jobs120922/120923 as CANCELLED at14:04:08Z on2026-09-08, before the expected15:56:46Z expiry. Checkpoint14 initially attributes this to a13:20Z decision; its final correction retracts that, instead reporting a user decision around02:30Z on09-09 and a subsequent failed cancellation attempt after the jobs were gone. This account is historical testimony, not a fresh audit of the user conversation or scheduler. `Terminated`/CANCELLED do not identify who acted, which command/API caused it, or authorization. The correction's “external canceller” inference is **not** independently proven here. Exact decision timing, actor and mechanism remain unverified; none changes the saved measurement.

## Environment/operational pitfalls (what, why, how, context)

| What happened or was claimed | Why it matters | Reproduction handling / evidence limit |
|---|---|---|
| BuildKit attempted `/opt/spur/.docker` in a read-only HOME | Image build failed before experiment work | Writable client `DOCKER_CONFIG`, `DOCKER_BUILDKIT=0`, omit legacy-incompatible `--progress`. This is an environment workaround, not a server code fix. |
| Slow initial startup, then faster restart | AITER JIT and cache warmth affect readiness time | Original launcher keys a node-local JIT cache by image ID. Recorded readiness1710s cold/540s warm; do not share a manually overridden cache across image builds. Timings are launcher polling reports, not isolated kernel timing. |
| HBM remained high after graceful stop | Starting a new server on still-busy devices can fail/hang | Stop your server before its launcher; inspect and wait for the original drain gate. Approximately28minutes was reported. Namespace-local `rocm-smi --showpids`, `/proc` or `pgrep` cannot establish absence of host processes; no kernel-deadlock diagnosis or fix is proven. |
| Readiness with silently declined acceleration can look healthy | Numbers may come from unintended kernels | Inspect decode/indexer engagement at readiness and prefill engagement after traffic. Full performance server log has8 occurrences of each marker. Original readiness helper only warns, not fails, when a marker is absent. |
| `ACC_LEN` defaults to3.61 | Drafted tokens can be committed without real verification | Explicitly pass `ACC_LEN=` for correctness and restart with3.61 for performance. Never mix the two phases' claims. |
| Historical eval `bench = untracked` | Weaker automatic correctness provenance | Keep the original stamp and `results/correctness/PROVENANCE.txt`. That file attributes it to root/container Git ownership; current checked-in eval.sh stamps on the host before docker exec, so the exact historical invocation leading to the stamp is not fully reconstructed. No blanket global safe.directory change is required or made. |
| Three-turn chat log does not contain actual input lines | Cannot reproduce identical dialogue from output alone | The runnable recipe supplies explicitly labeled reconstructed prompts; original outputs remain intact. |
| AIPerf needs Python>=3.11, server uses3.10 | Installing into server site-packages or wrong interpreter breaks setup | The pinned helper creates an isolated venv, selects3.11 and installs its pinned source. Original package resolver remains partly floating; observed constraints are evidence, not a complete lock. |
| Original eval dependency installer tolerates some install failures | A nominal pin may not equal installed commit | Intended lm-eval commit is recorded, but result git_hash is null. Capture future installed versions and failures; do not infer a historical commit stamp that does not exist. |
| Original wrapper prefixes repo paths | Absolute OUT_DIR can form incorrect/doubled paths | Keep OUT_DIR relative; absolute paths are appropriate for SERVER_LOG. Use a dedicated run directory and unique container name; original launcher/wrapper can remove their named container. |

## Corrections to storage and memory conclusions

- The old log/spec claims node153 had67GB free, node168 had22GB free, all base layers already existed, and a reload required92GB. Namespace `df` plus a flat watcher sample is **not evidence of the host Docker backing filesystem, no-download bytes, expanded image footprint or a universal free-space requirement**. Preserve those observations as historical reports, not host facts. No claim that node168 was unusable is supported by this evidence alone.
- `free -g` reports **GiB**, so the reported2751 is not2751 decimal GB. The raw historical complete memory snapshot is missing. Metadata `allocated_cpu_dram_gb=3023` comes from script configuration, not measured RAM.
- Source inspection supports that the metadata value does not size the launcher HiCache. It does **not** prove physical DRAM differences had no performance effect. Full reported pool allocation (235.47 per rank in the original log's units) cannot exclude paging, bandwidth, NUMA, pressure or baseline-host differences. Those effects remain unmeasured.
- The original archive was integrity-tested but not loaded during the09-08 mix task. A separate09-09 validation now records a successful load and matching identity; it closes that specific later round-trip gap, **not** historical host-storage uncertainty. See `provenance/later-image-validation.md`.

## Packaging provenance and scope

All original source/evidence is read-only. Canonical `results/perf` and full correctness evidence are copied; only the superseded partial `results/20260908-repro` copy is omitted. Original workspace logs are preserved losslessly as gzip, even when duplicated in result trees. The23,912,216,852-byte image and model weights were not copied. User explicitly approved compressed raw logs and artifacts over4MB.

The existing dependency tar files were created from file content: directory symlinks were omitted and a file symlink was dereferenced. They remain unchanged. `provenance/git-snapshots.json` records original Git blob/mode/target/tree/commit identity and `scripts/bootstrap_repo.py` reconstructs the authentic source including all4 symlinks. Offline tree/commit checks are stronger than merely assigning the desired commit label to an unrelated checkout. Parent Git history/config/credentials are not shipped. One original InferenceX B200 runner symlink targets a missing upstream file; it is preserved for exact tree identity, is unrelated to this MI355X recipe, and is listed in `provenance/symlinks.json`. The retained `audit-work/offline-repo` is generated offline verification material, not another experiment; do not run it as a benchmark workspace.

`patches/pin-external-inputs.patch` is an explicitly new packaging adaptation (base digest and dataset revisions). It was not used for the historical measurement. No new inference kernel patch or bug fix was invented. Original inline fixes are listed in `patches/README.md`.

Secret checks scan common key/token signatures and avoid credential files. They cannot guarantee absence of arbitrary secret strings; per-sample model/trace data is intentionally preserved under user approval. Do not publish this package beyond the approved audience without data/license review.

## Gaps left explicit

Historical host GPU kernel-driver version, RDMA fabric/rails/driver/IPs, raw host RAM/Docker backing-store snapshots, baseline hardware/weight identity, full weight hashes/revision, exact chat prompts, complete wheel-hash lock and independently stamped installed lm-eval commit remain missing. No new GPU run, kernel fix validation, statistical test, live cold replay or cold image rebuild was performed for packaging. Archive-load validation is separately attributed. Read `audit.md` for exactly what passed offline.
