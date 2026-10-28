# Reproduction guide validation — 2026-10-28

This is validation of the published source/configuration assembly, not a new
image build, GPU test or reproduction of model accuracy/performance.

| Check | Result |
|---|---|
| Workspace protection before pull/rebase | Backed up 157 pre-existing untracked files. Git protected 82 incoming same-path files; 81 matched upstream, and INDEX.md differed only by our live-status link. After rebase/restoration all 157 original file hashes matched. Local pre-rebase commit retained in a dedicated backup branch. |
| Public availability | Anonymous public APIs returned the SGLang image manifest, Infera repository, InferenceX commit, ungated model and ungated dataset. Pins recorded in SOURCE_PINS.json; full downloads were not performed. |
| PR states | Refreshed 44 PRs from GitHub; heads/bases, state, draft flag and timestamps are in PR_SNAPSHOT.json. Eleven owned core PRs remain open: five ready, six draft. |
| File assembly | Prepared a fresh directory from the complete base kit, c104 worker/common, c104 site, two CSVs and K01. Each resulting file has a source/hash record in generated ASSEMBLY.json. |
| K01 reconstruction | All eight added files applied with zero fuzz. Python AST parsing passed without importing GPU modules. The reconstructed hook plus the later llr42 alias patch equals the separately archived llr42 hook byte-for-byte; that alias is not applied to the llr37 assembly. |
| CSVs | Parsed 75 and 49 data rows, matching the archived DPA/K3 and PTPC tables. No retuning or numerical equivalence claimed. |
| Shell syntax | All assembled shell files and both generated environment files pass bash -n. |
| Worker command contracts | Four invocations (historical/control × prefill/decode), using explicit shell mocks for ip/docker, confirm PP4/TP1 prefill, TP4/attention-DP4 decode, 3/1/4 MTP, index sharing, chunk 8192, memory settings, S13/KF/CSV mounts and RDMA selection. No Docker process was invoked. |
| Historical/control distinction | Historical commands include simulated acceptance 2.99, PP overlap and custom AR; control commands remove simulation and disable both switches. The control is not labeled correctness-qualified. |
| No-overwrite behavior | A second assembler call to the same destination exits with an error; all destination file hashes remain unchanged. No deletion is implemented. |
| Optional Infera #218 backport | Production-only diff passes git apply --no-index --check against the fdbafb60 build script, applies successfully, and passes bash -n. The build script was not executed. |
| Documentation / artifact integrity | Relative links and the published SHA256SUMS were checked; repository lint was run on newly maintained docs/config/helper files. Archived third-party diffs retain their bytes. |

The performance artifacts remain untouched. The original JSON's prefill topology
metadata is inconsistent with the launcher; the guide explicitly records the
actual PP4/TP1 topology and the correct eight-GPU denominator instead of modifying
that JSON. The older base pack-up's environment.md describes a different DCP
stack and is not used as llr37's source of truth.

Outstanding experimental limits are in PUBLIC_REPRODUCE.md: historical weight and
trace revision binding, immutable integrated image publication, floating build
dependencies, correctness of the final stack, and clean upstream integration of
the KF/CSV artifacts. Historical tests remain useful evidence within their scope.

Local backups, command captures, CI logs and caches are deliberately excluded
from this publication; public PR comments and the archived pack-ups carry the
shareable evidence. No new GPU node was allocated or accessed for this guide.
