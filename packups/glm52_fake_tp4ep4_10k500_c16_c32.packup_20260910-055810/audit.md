# Offline audit and completeness record

Packaging performed no GPU/Docker/Spur invocation, host configuration change, Git branch/commit/push, source edit, move or deletion. All work was inside this new package. Source instructions were copied to `CLAUDE.fake-tp4ep4-short.20260910-0558.md.bak` using the actual UTC creation time; fresh package instructions are separate.

## Executed audit

Initial full audit at2026-09-10T06:12:37Z completed successfully using Python3.14.6 on the packaging host (not the experiment Python environment). The machine-readable repeated assembly audit is `provenance/audit-results.json`. Final manifest verification runs after assembly without `--before-manifest`.

| Check | Result / scope |
|---|---|
| Full source inclusion |All59 non-cache source files preserved; no prior long-run tree copied|
| Copy/gzip integrity |68 provenance records, source SHA256 and decoded byte lengths exact|
| Source immutability |Included original source bytes and mtime_ns unchanged|
| Patch application |3/3 applied with `patch --batch --fuzz=0 -p1`; all exact modified-file matches|
| Syntax |15 Python AST parses and4 shell `bash -n` checks passed|
| Guard regression |Original index/topk tests exit1; patched tests exit0,72 and48 subcases|
| Proposal runtime regression |Not rerun: local Torch/SGLang absent and Docker prohibited; original4-case test shipped|
| Relocation |Helper works from a relocated cold kit; generated scripts equal exact allowed substitutions; empty caches|
| Destination protection |Existing target, kit descendant and original experiment descendant rejected with exit2|
| Tokens and throughput |Both128 successes/zero errors,128×10000/500,64000 outputs;1803.1454819620747 and2569.0979197579804 tok/s exactly recomputed|
| ITL |P50/P90 exactly recomputed from full arrays|
| TPOT |Reported P50 preserved; ITL estimate differs by-0.00001214405443ms/-0.00001043025292ms; not exact per-request-latency reconstruction|
| Acceptance |Matches embedded server-info average; raw counter reconstruction unavailable|
| Scheduler/backend evidence |114×16 versus56×32+1×16; retractions0; sampled graph true; C32 seq192/1..96 decline present|
| Recognizable secret scan |No private-key/GitHub/AWS/Bearer-pattern hits; no suspicious credential env names; heuristic, not a proof over arbitrary text|
| Final checksums |`MANIFEST.sha256` covers every deliverable file except itself; verify with commands below|

```bash
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/audit_package.py
sha256sum --check MANIFEST.sha256
# Optional while source paths remain available; strictly reads sources:
python3 scripts/audit_package.py --verify-sources
```

The manifest intentionally does not hash itself. No evidence files are omitted from checksum coverage. The assembly report predates the final checksum freeze and explicitly labels that stage; final audit output is produced on demand rather than modifying the hashed report every run.

## Cold-read and completeness review

- Date, actual node/job, requested fixed lengths/concurrency,16 warmups/128 measurements and no specified performance threshold are documented.
- Original task/process/checkpoint/state history is included verbatim. Stale249-only source instructions are retained as history, not applied to measured-node attribution.
- README leads to reproduction, actual environment, three patch explanations, full two-point JSON/logs, provenance and caveats. Internal Markdown links resolve within the kit.
- Required runtime scripts, inputs, three modified files, three exact originals, tests, Dockerfile and metric implementation are present. Historical scripts are unchanged. New relocation helper is labeled and does not run workloads.
- External model and image archive paths, exact image ID, source SHAs and credential requirements are explicit. No images, weights or caches included; full logs authorized and losslessly compressed.
- Host/device/driver/RAM/library facts come from captured source; CPU, base registry digest, exact Python/FlyDSL/transitive versions and model shard hashes are explicitly missing, not inferred from this host.
- C32 fallback, synthetic semantics, true-PD non-claim, unhealthy249/failed stop, untouched056, KFD-zero-use caution and only single-run evidence are explicit.
- Reproduction requires a new authorized allocation and workspace; no historical workload-stop command is prescribed. Original and prior packages remain untouched.

## Known gaps, not hidden failures

No GPU replay during packaging. Proposal test requires unavailable runtime imports; it is not claimed freshly passed. Exact TPOT cannot be recreated from missing per-request latency, and acceptance cannot be derived from missing raw counters; both limitations are separately labeled. Archive hash comes from prior provenance, not a new23.9GB read. Base-image digest, CPU/model-shard identities and full software lock were never captured. No true-PD or AgentX compatibility implementation/testing is claimed.
