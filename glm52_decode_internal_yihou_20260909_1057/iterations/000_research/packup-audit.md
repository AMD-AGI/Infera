# Offline cold-read self-audit

**PASS for the documented offline checks.** Machine-readable evidence: `provenance/audit-results.json`; reviewed secret scan: `provenance/final-secret-pattern-scan.json`; delivered bytes: `MANIFEST.sha256`. This is a self-audit, **not a cold GPU rerun**.

## Checks actually executed

- **3736 source mappings**,500,163,155 original/decompressed bytes: every included verbatim/gzip/tar member matches the recorded SHA256 **and current read-only original bytes**. Original task backup was separately byte-compared (`provenance/spec-provenance.json`). All14 checkpoint sections and final correction are retained.
- Offline bootstrap reconstructs original rocm-llm-bench/InferenceX/aiperf blobs, executable modes, symlinks, trees and authentic commit objects. All3 HEAD/tree pins, clean status and `git fsck --full --no-reflogs` pass. No clone/fetch/download is used. Original source archives are unchanged.
- New four-file external-input pinning patch passes a clean `patch --dry-run` against the restored source. Its runtime effect was **not GPU-tested** and it was not used in the historical run.
- Original collector recomputes the one-row performance CSV exactly from the saved JSON, including249/260/0 accounting and throughput/interactivity conventions.
- Full GSM8K raw JSONL has2638 filter records:1319 distinct questions in each of strict-match and flexible-extract. Both recompute0.9681576952236542;5-shot/full-test configuration agrees with summary JSON.
- Performance server evidence contains8 decode,8 fused-indexer and8 prefill engagement markers.
- `bash -n`:9 packaged shell scripts and8 reproduction command blocks. `ast.parse`:8 packaged Python scripts, without running their benchmark code.
- Entry-document Markdown links:17 checked, zero missing at audit time. Manual cold-read also checks script/dependency paths, relative OUT_DIR handling, immutable originals, prerequisites, simulator separation and clearly labeled external/network dependencies.
- All4 restored symlink targets remain inside the package. Three resolve; one pre-existing broken upstream B200 runner alias is preserved for source identity and is unrelated to the MI355X recipe (`provenance/symlinks.json`). It is not a dangling reproduction dependency.
- Common secret-signature scan reviewed7990 current regular files (including expanded source). One match is a synthetic SigV4 example in pinned AIPerf `test_redact.py`, not an experiment credential; no unreviewed match remains. Original pre-compression scan identifies the same fixture. No secret values are printed, credential configurations are excluded, and heuristic scanning is not an absolute secrecy guarantee.
- External image was read-only rehashed:23,912,216,852 bytes; SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2`. It was **not copied**.
- Two explicitly supplied later image-load validation files match copy hashes. They are separately attributed; no later experiment metrics/source were folded into the09-08 mix evidence.

## Skill checklist coverage

| Requirement | Coverage / remaining gap |
|---|---|
| Time | Absolute experiment/packaging dates, original timestamps/mtimes,14 checkpoints plus correction; disputed decision/start timestamps explicitly qualified. |
| Purpose/spec/criteria | Original CLAUDE copied verbatim; README goal/results; exact one-point scope; no invented “sane accuracy” threshold or equivalence tolerance. |
| Hardware | Historical node/role,8 MI355X,gfx950,CPU visibility/kernel and memory-unit correction; host driver/RDMA/raw host RAM/storage snapshot **missing**, not guessed. |
| Software | Top repo branch+SHA, nested SHAs, source snapshots, image ID/base digest, Dockerfile, observed runtime versions; no complete wheel-hash lock or independently stamped installed lm-eval commit. |
| Commands/scripts | Original tracked scripts preserved; complete ordered recipe, offline bootstrap, exact knobs; chat inputs explicitly reconstructed, since original text is missing. |
| Fixes | Inherited inline fixes explained; packaging-only pinning diff labeled what/why/how/context; no new kernel-fix claim. |
| External dependencies | Absolute image/model paths, pinned public dataset acquisition, model identity limitations, no undocumented host-injected library. Runtime downloads explicitly declared. |
| Secrets | Required access/providers listed without values; no credential stores packed; synthetic source-test match reviewed. |
| Notes | Failure modes, simulation/real-verification split, namespace/proc/storage, DRAM, cancellations and uncertainty preserved. |
| Results/evidence | Full machine-readable summaries and raw compressed records/logs; existing plots copied, not regenerated; partial duplicate omitted. |
| Hygiene | Source unchanged, no directory removed, heavy/full compressed logs explicitly approved, image excluded, all work in package; no new GPU/Docker/environment action. |

## Re-run these checks without downloads

```bash
KIT=/shared_nfs/yihou/packups/glm52_mix_repro.packup_20260909-100729
(cd "$KIT" && sha256sum -c MANIFEST.sha256)
PYTHONDONTWRITEBYTECODE=1 python3 "$KIT/scripts/audit_package.py"
```

`--verify-source` additionally re-reads declared original paths; `--verify-image` additionally streams the external23.9GB archive. Neither option modifies those sources. The no-option audit uses only packaged files and the retained generated offline checkout. Do not redirect output over the signed/manifested evidence unless deliberately creating a new package revision.

## What has NOT been validated

No new GPU inference, image rebuild, end-to-end cold replay, driver/fabric equivalence, weight identity, statistical equivalence or kernel-deadlock fix. Later image-load success is separate evidence in `provenance/later-image-validation.md`. Missing historical provenance remains in `environment.md`/`notes.md`; completing packaging does not close those scientific/environment gaps.
