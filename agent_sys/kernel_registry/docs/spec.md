| | |
|---|---|
| Status | Normative for kernel artifact catalogs |
| Version | 2 |
| Updated | 2026-10-28 |
| Summary | Keep kernel code, integration instructions and test history together in a simple Git directory. |

## 1. Purpose

Let an engineer or AI agent preserve and hand over an optimized operator with
its original implementation, naive reference, integration method and historical
test results. A customer can use the format in their own repository.

## 2. Design principles

Keep the system simple: ordinary files, readable names, Git for version control,
and a small optional CLI. Missing historical information must remain visible,
but must not prevent useful artifacts from being registered. A measurement is
an observation, not a deployment recommendation.

## 3. Required information

Each kernel records its purpose and constraints, original implementation, naive
reference, optimized versions and their parent versions, integration instructions,
documentation and provenance. Version labels are readable. Actual source bytes
are identified by SHA256. The version used in an overlay may differ from the
best candidate and must have its own label when the source differs.

Each microbenchmark or e2e test records the tested versions, exact source hashes,
baseline, environment, command/configuration, workload, correctness result,
performance measurements and raw evidence. Results retain scope, metric units,
aggregation and limitations. One e2e result can name several kernels. A failed
or inconclusive test stays in history; synthetic correctness, live kernel checks
and full-model correctness are distinct statements.

Environment snapshots record the container identity, hardware and driver,
OS/kernel, Python, package versions and source revisions. Workload records include
input/model identity and generation settings as applicable. Historical gaps are
named explicitly. Completeness describes available reproduction information,
not whether a new rerun has succeeded.

## 4. Versioning and validation

Git is the version history and review mechanism. Published version labels and
run names should be retained; fixes get new labels and explain their relationship.
The tool does not build a second version-control system or require a privileged
write API. Engineers can edit the documents normally.

Validation rejects malformed records, dangling local references and byte/hash
mismatches. Missing optional historical facts produce actionable warnings.
Files are local to the catalog or external with a URI, SHA256 and byte size.
Large artifacts need not be committed. External availability is reported, not
assumed. No files, code or network locations in a catalog are executed by validation.

Shared artifacts omit personal paths, machine identities, private endpoints and
credentials. A redacted source is explicitly distinguished from its historical
original; its new digest cannot inherit a claim of measured GPU correctness.
Public software versions, technical results and required attribution remain.

## 5. What this module does not own

Optimization, benchmark execution, GPU scheduling, deployment and automatic
selection of a fastest kernel remain with existing tools. The agent work system
can carry a catalog as ordinary handoff content; its lifecycle is unchanged.

## 6. Acceptance criteria

1. A readable example demonstrates original, naive and optimized sources, lineage,
   integration and both microbenchmark/e2e history with environment references.
2. Validation detects changed file bytes, missing versions, malformed JSON and
   paths escaping the catalog, while incomplete history remains usable.
3. History shows every observation with correctness, scope, reproduction gaps and
   comparability; failed or confounded results are never presented as proven gains.
4. Ratios use the declared direction and aggregation; mean case speedup is not
   replaced by a ratio of totals.
5. Environment capture records actual available information and lists missing
   facts without importing GPU libraries or dumping credentials.
6. The installed CLI and skill work outside the checkout. A new user can start
   from a small template, edit it and validate without a running service.
7. Existing artifacts can be imported with exact source bytes or explicitly
   marked redacted representations, meaningful historical records and limits,
   without claiming a new GPU rerun.
8. The skill guides an explicitly requested e2e integration using the archived
   version, a compatible environment, actual activation evidence, completed
   requests, scoped correctness checks and rollback. It can use a sanitized
   successful historical packup without treating that history as a new run.

## 7. Open questions

Long-term hosting of large external artifacts belongs to the customer's storage
policy. This version records their identity and location without managing storage.
