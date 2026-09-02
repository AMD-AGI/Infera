# agent_sys — the design stage

| | |
|---|---|
| Status | Working record of stage two. Updated per module, step 2 of four |
| Scope | The task, the module order, what is frozen, and what each module's research settled |

**Binding rules live in [`../engineer_principle.md`](../engineer_principle.md).**
Read those first; this file is the state of the work, not the rules.

Tracked deliberately. `.gitignore` treats a `CLAUDE.md` anywhere as local
scratch, and the per-module research below is not scratch — it is the only
written record of what was measured, and the probe scripts it cites live in
`scratch/`, which is gitignored.

---

## Task

The nine specifications are written and reviewed. **This stage turns them into
design documents — still no code.** One module at a time, main spec first, then
each subordinate module in dependency order.

**Each module is its own complete task**, run to the same four steps:

| | |
|---|---|
| 1 | **Deep research.** Independent, per module. Prior art, first-hand measurement, official sources |
| 2 | **Update this file** with what that research settled |
| 3 | **Plan**, and get it agreed |
| 4 | **Write the design document**, then **stop and report to the user** |

The user compacts between modules. Do not begin the next module unprompted.

### Module order

| # | Module | Deliverable | Status |
|---|---|---|---|
| 1 | **main / architecture** | `agent_sys/docs/design.md` — the loader, the five schemas, the registries, the composition root | **done, rev. 1** |
| 2 | **handoff** | `handoff/docs/design.md` | **done, rev. 1** |
| 3 | **validator** | `validator/docs/design.md` | **done, rev. 1** |
| 4 | **task_graph** | `task_graph/docs/design.md` rev. 10 → 11 (spec rev. 7 → 12) | **done, rev. 11** |
| 5 | **agent** | `agent/docs/design.md` | **done, rev. 2** |
| 6 | **closure** | `closure/docs/design.md` | **done, rev. 1** |
| 7 | **env_mgr** | `env_mgr/docs/design.md` | **done, rev. 1** |
| 8 | **demo** | `cli/docs/design.md` | **done, rev. 1** |
| 9 | **monitor** | `monitor/docs/spec.md` **and** `monitor/docs/design.md` — the only module with no stage-one spec | **done**, spec rev. 14, design rev. 2 |

A design document **turns the spec into files, classes, and interfaces. It adds
no requirements.** Where it makes a choice the spec left open, it states the
choice; where implementing the spec exposed a contradiction, it says so rather
than papering over it.

**House style: `task_graph/docs/design.md` rev. 10.** Header table with
Status / Revision / Implements / Language; a layout-and-import-graph section; a
per-module build-versus-adopt table; a test plan mapping every acceptance
criterion to a named test; an implementation order; a **deviations** section;
and a **new open questions** section. It specifies *interfaces, not bodies* — a
method is a signature plus a sentence, and a body appears only where the
ordering of steps *is* the design decision.

## What is already frozen

**The spec set is agreed. A design document does not amend it.** If the design
needs the spec to change, that is a finding to report, not an edit to make.
That happened exactly once, and the report was accepted — see "The no-agent
question, closed" below; the three revisions it produced are in the table.

| Document | Rev. | Criteria |
|---|---|---|
| `docs/spec.md` | 9 | §8 — 15 |
| `handoff/docs/spec.md` | **5** | §9 — 17 |
| `validator/docs/spec.md` | **7** | §11 — 21 |
| `task_graph/docs/spec.md` | 12 | §11 — 54 (1–35 built and green) |
| `agent/docs/spec.md` | 4 | §8 — 16 |
| `closure/docs/spec.md` | 8 | §5 — 11 |
| `env_mgr/docs/spec.md` | 3 | §10 — 22 |
| `cli/docs/spec.md` | 6 | §6 — 16 |

172 criteria. `docs/spec.md` §9 is the index. **The design's test plan must map
every one of its module's criteria to a named test** — that is what "done"
means, exactly as `task_graph/docs/design.md` §11 does today.

Two components are implemented and must not regress: `env_mgr` (recipes,
PR #123) and `task_graph` rev. 7 (423 tests, PR #124). `pytest agent_sys`
is 423 green and stays that way.

## Research findings that decide design questions

> **Nine findings below were superseded by the user-interface stage (2026-08-29),
> which deleted jsonnet.** They are **kept verbatim and tagged inline**, never
> edited or removed: this file says above that it is *"the only written record of
> what was measured"* and that the probes live in gitignored `scratch/`, so
> rewriting a measurement here destroys the evidence rather than updating it. A
> superseded finding was **true when measured** — what changed is the thing it was
> measured about. Each tag names where the current statement lives; `docs/design.md`
> rev. 3 §3.2 carries the three that still teach something and says which of their
> *conclusions* transfer and which are unmeasured for YAML.

Recorded in `agent_sys/scratch/design/findings-main.md` (gitignored), with the
probe scripts beside it. The load-bearing ones:

### Measured here, first-hand

- **[SUPERSEDED — the module is deleted; the conclusion is UNMEASURED for YAML. `docs/design.md` §3.2 finding 1]** **`_jsonnet` is thread-safe and releases the GIL.** 8 threads × 200 renders
  0.95 s vs 6.98 s serial. Rendering may be parallelised.
- **[SUPERSEDED — same. `load_package` is serial today and nothing measures its cost]** **Per-render cost is a fixed ~23 ms regardless of spec size** — `{}` costs the
  same as a 2000-element tree. It is per-call VM construction, not warm-up. 100
  specs ≈ 2.3 s serial. The loader must not re-render, and parallelising pays.
- **[SUPERSEDED — there are no imports. The containment QUESTION survives as `docs/design.md` O3, restated against `_scan`]** **Cross-package `import` by relative path works out of the box.**
  `import_callback(base, rel)` gets the *importing file's directory*; supplying
  a callback **completely disables `jpathdir`**, so a callback means owning
  resolution yourself. Since 0.19.0 it must return **bytes**. Callback
  exceptions are flattened to `RuntimeError` — the type is lost.
- **[SUPERSEDED — nothing renders. The Norway trap is now closed by `ruamel.yaml`'s round-trip loader being YAML 1.2, not by the source format: `docs/design.md` §4.4]** **Rendered JSON is YAML-equivalent**, verified over `null`/`true`/`1e3`/
  `"NO"`. It also **structurally avoids the YAML 1.1 Norway trap** — an argument
  main spec §4.4 does not currently make.
- **[SUPERSEDED — measured 2026-08-29: YAML integers are Python `int`, exact, under both parsers. `docs/design.md` O4]** **jsonnet numbers are IEEE 754 doubles.** `12345678901234567890` round-trips
  as `…567168`. Any spec field needing an exact large integer must be a string.
  Nothing in the spec set says this.
- **`jsonschema` produces exactly the messages main spec criterion 3 claims**:
  `'reproducible' was expected` (const),
  `Additional properties are not allowed ('sneak' was unexpected)`.
- **Error multiplication is reachable from our own schemas.** `$ref`-ing the
  metaschema to check `handoff.items_schema` turns one fault into **8 identical
  errors**; `check_schema` as a named step, or `best_match`, gives 1.
- **pydantic `model_json_schema()` emits `const` + `additionalProperties: false`
  + `description`** — the exact three mechanisms §4.4 names. So hand-written vs
  generated schemas is a real open choice, not a forced one.
- **Landlock ABI 3 blocks all five documented defeats** on this machine, with
  **no `bwrap` present**: scripted read/write outside the zone, a `zone-EVIL`
  sibling, an in-zone symlink pointing out, `..` traversal, and a `bash -c`
  child. In-zone access still works.
  - **EXECUTE is bit 0, not bit 4.** Granting the wrong bit makes the
    interpreter itself unexecutable, and the error blames `python`, not the
    policy.
  - **The interpreter's own prefix must be granted.** `/usr` is not enough for a
    conda or venv install.

### From the prior-art surveys

- **Render → parse → validate, with the parse step injected.** The shape in
  `check-jsonschema` (`ParserSet`), `cfgv` (`load_strategy`), and `jsonargparse`
  — which is the closest prior art to our whole pipeline and also parses
  rendered JSON with a YAML loader.
- **Do not generate pydantic models and drop the schema pass.**
  `datamodel-code-generator` publishes the keywords generated models silently
  fail to enforce — `if/then/else`, `not`, `unevaluated*`, `propertyNames`,
  `dependentRequired`, `patternProperties`, `uniqueItems`. And JSON Schema
  checks structure while pydantic **coerces**: validate-then-`model_construct`
  leaves typed fields holding raw strings.
- **Four registries stay separate, and the real argument is the collision
  policy.** Every canonical "one generic registry" decomposes into N typed
  containers: `runtime.Scheme` is 7 maps, 3 key types, and **three different
  duplicate policies** (panic / silent / silent). Airflow's is ~25 containers.
  A generic registry cannot hold one policy; it grows N inconsistently.
- **The component `Registry` keeps overwriting; the four spec registries must
  reject duplicates.** Different objects, different jobs. `fsspec`'s shape is
  the model — error by default, explicit `clobber=True`, identical
  re-registration is a no-op. `pluggy` also rejects the *reverse* collision.
- **Error messages must enumerate candidates.** pytest lists every available
  fixture; dbt prints `expected one of {sorted(...)}`. Ours says only
  `no component registered as 'handof_mgr'`.
- **The closure check is a separate pass after all four registries are loaded** —
  dbt's `ManifestLoader.load()` shape; OPA checks cross-bundle overlap *before
  any mutation*. Resolve-during-load cannot see the far side of a two-way
  binding.
- **Symmetric blame is necessary but not what OPA#7806 is about.** *Corrected in
  module 2*: verified via `gh api`, the pre-fix error **already named both
  bundles**; the issue is `opa build` **silently ignoring `.manifest`** without
  `-b`, defaulting `roots` to `[""]`. The real lesson is better for us — *a
  silently-applied default upstream of a consistency check* is what defeats
  users. Still: name both sides, sort for determinism, add a hint.
- **Add a layering gate**: skip the closure check for a spec that already failed
  its own load checks. Kubernetes CRD validation does this to avoid "error
  messages that are not actionable".
- **Two exception classes**: not-found vs inconsistent. JPMS separates
  `FindException` from `ResolutionException`.
- **No prior art exists for our exact two-way-binding crash.** The survey
  constrains how to *report* it, not whether it is right.
- **Nothing gives phase isolation for free.** Airflow's three phases share one
  `context` dict by reference. **Dagster documents our exact threat model as a
  live gap**: *"if no result is emitted for the check, downstream execution
  proceeds and a warning is logged"* — the graded unit enforces its own gate and
  silence passes. Concourse is the model: a missing input means the container is
  never created, and outputs leave only through designated directories.
- **Backends raise; no capability matrix.** PEP 249 sanctions
  `NotSupportedError`. SQLAlchemy found inherited capability flags to be lies
  and now requires local declaration. LiteLLM's 3040×35 matrix produced #20885,
  where a missing cell read as `False`.
- **Package-escape permissiveness has a price.** Kustomize resolves symlinks
  *then* enforces containment and calls it "an intentional security feature".
  Helm warns and continues — CVE-2025-53547, whose advisory says *"Helm warns of
  the symlinked file but did not stop execution"*. dbt permits arbitrary local
  paths and closed the resulting relocatability bug as `not_planned`. Main spec
  §4.3 currently sits on the permissive branch; that is a **spec** question to
  report, not a design decision to take.

## Module 2 — handoff. What its research settled

`agent_sys/scratch/design/findings-handoff.md` (gitignored), with probes in
`scratch/design/handoff/` and `scratch/design/probes-handoff/`.

### Measured here, first-hand

- **The executable bit is the only mode bit that survives every copy path.**
  Under `umask 077`, `cp -r` turns 0644 → **0600** and 0755 → **0700**; the
  x-bit survives all six copy × umask combinations. git's `fsck.c` whitelist and
  Nix's `archive.cc` converged on the same answer independently. A digest over
  the full mode would fail criterion 6.
- **`shutil.copytree` defaults dereference symlinks** and change the mode.
  `cp -r`, `cp -a`, tar and zip round trips all preserve them. The consumption
  protocol must name its copy mode.
- **A git-shaped tree digest satisfies criterion 6 and 7 as measured**: `cp -r`
  reproduces it; a sibling `validation.yaml` does not move it; mtime does not
  move it; dropping the x-bit does. Recomputation costs **17 ms for 1000 files /
  4 MB**, so it need not be cached.
- **Sort byte-wise over `os.fsencode()`, never `sorted(str)`.** A file named
  `b"\x80name"` surrogate-escapes to U+DC80 — a *high* codepoint — so `str` sort
  puts it after `éname` and byte sort before. `pathlib.Path` **rejects bytes
  paths**, and `os.scandir(bytes)` yields bytes names, so a name comparison
  against a `str` literal is silently `False`.
- **Locale collation is a trap for the shell, not for Python.** `sorted()` is
  codepoint-based and unaffected by `setlocale`; only `strxfrm`, `sort(1)` and
  globs follow the locale.
- **Git's trailing-slash sort rule cannot be reproduced by plain `sorted()`** —
  `tree.c::base_name_compare`. Nix uses plain name order. Two consistent orders;
  whichever we pick must be **enforced on read**, as `fsck.c::verify_ordered` and
  `archive.cc` both do.
- **Empty directories survive every copy mechanism we use; only git loses
  them.** Adopting git's tree format wholesale would drop a real distinction.
- **[PREMISE SUPERSEDED, CONCLUSION UNTOUCHED — no backend remains, but *"digesting rendered text binds the digest to the backend"* generalises to any serialiser and `handoff/digest.py:144` cites it. Not `spec_loader`'s to re-decide]** **The two jsonnet backends do not agree byte-for-byte** — `_jsonnet` 0.22.0
  emits `0.10000000000000001` where `rjsonnet` 0.5.6 emits `0.1`; indentation
  and trailing newline also differ. **Digesting rendered text binds the digest to
  the backend.** They agree after `json.loads`.
- **[SUPERSEDED — duplicate keys are now rejected by `ruamel.yaml` with a position, `docs/design.md` §4.4]** **jsonnet is already a canonicaliser, and not JCS**: `1.0` → `1`, `-0.0` →
  `-0` → Python `int` 0, `1e300` → 301 positional digits where JCS mandates
  `1e+300`. It **rejects duplicate keys statically**; `json.loads` keeps the last.
- **[PARTLY SUPERSEDED — the three libraries still behave as measured; *"one value we can produce"* no longer holds, since `12345678901234567168` was jsonnet's rounding of an exact YAML integer]** **Three JCS libraries, three behaviours on one value we can produce.**
  `12345678901234567168`: `rfc8785` raises `IntegerDomainError`, `jcs` silently
  rounds to `…567000`, `canonicaljson` passes through. **Refuse, do not coerce.**
- **`startswith(zone)` is the `/a/b`-matches-`/a/bc` bug**, and
  `Path.is_relative_to` is not the fix either — it is purely lexical, so `..` and
  an in-zone symlink pointing out both pass. `os.RESOLVE_BENEATH` is **not
  exposed** by CPython 3.13.
- **`/tmp` and the working tree are different filesystems here**, so a staging
  directory must be a sibling of its destination. `rename` onto a **non-empty**
  directory fails `ENOTEMPTY`, and `os.mkdir(vN)` is a free atomic allocator.
- **The naive markdown-heading regex is wrong in both directions** — it misses
  setext and ≤3-space indents, and matches headings inside a code fence.
  `markdown-it-py` 4.0.0 gets all seven cases right and is **already installed**,
  via `rich`.
- **Criterion 17's false-positive rate, measured on this repository**: 650
  absolute-path matches over 276 files, of which **23 are genuinely local and 627
  (96%) need a suppression rule** — system paths, URLs, prose fragments.

### From the prior-art surveys

- **Nobody detects locality dependence by a path's *shape*.** lintian, rpmlint,
  conda-build and Nix all match a prefix supplied by an oracle; Bazel prevents
  the path existing. Debian #1002451 proposed a tighter regex and it was
  **rejected** — *"the shape is a configurable parameter"*. rpmlint #1350 is the
  controlled experiment: oracle replaced by a shape regex, false positives within
  one release.
- **Digest the `content/` subtree; put the verdict file beside it.** Debian's
  `Release` names 772 digests and never itself; PyPA's `RECORD` *"cannot contain
  a hash of itself"*. **Criterion 7 is a structural fact, not a convenience.** The
  name-exclusion alternative is not stable.
- **REAPI's inclusion test is the rule for what belongs in a digest**: *does
  omitting this let a wrong answer masquerade as a right one?* Producer,
  timestamps and verdict all fail it — correctly outside. Git includes committer
  dates and pays for it with `git patch-id`.
- **Make the digest a map, `{"sha256": …}`, not a string** — in-toto's algorithm
  agility, the only cheap migration path. **Namespace it**: W&B prefixes its
  manifest digest, DVC did not and paid with a permanent second cache and a
  `dvc cache migrate` command.
- **If a digest is recorded durably, the rule and its enforcement must land in
  the same change.** Matrix's canonical JSON does not implement its own spec, and
  fixing it needed a **new room version** because non-conforming digests were
  already persisted.
- **`checksumdir` and `dirhash` are both unusable.** The first ignores filenames
  — two trees with contents swapped hash identically. The second has no
  executable-bit option.
- **Prior art does not support our two-way binding crash.** SQLAlchemy
  `back_populates` **does not check that the two sides agree** (verified on
  2.0.44: a flat contradiction configures silently); GraphQL Federation
  **deleted** the requirement in Fed 2; Kubernetes treats a dangling ownerRef as
  absent. But SQLAlchemy 2.0 moved *away* from deriving one side — `backref` is
  *"legacy"* — for our reasons: explicit, statically visible, PEP 484. So
  **explicit two-way declaration has precedent; checking agreement does not.**
- **Rust's orphan rule is the strongest argument against**, and its force
  survives the three disanalogies (we have identical assertions not conflicting
  behaviours; a closed world; both spans available): the *possibility* of two
  declarations is itself the defect.
- **RFC 9535 §2.5.1.2 forbids a valid JSONPath query from erroring** — an
  out-of-range index *"simply result[s] in fewer nodes being selected"* — so the
  library can never say "you addressed nothing". Only the caller can.
  **`jsonpath-ng` leaks 19 stdlib exceptions from validly-parsed queries**
  (`IndexError`/`KeyError`/`TypeError`/`ValueError`); `python-jsonpath` leaks 0.
  **JSON Pointer separates parse-error, no-match and null-value; JSONPath cannot.**
- **A presence-only README check on an agent-authored artefact is theatre by
  construction.** Hugging Face returns HTTP 200 `OK` for a card whose entire
  prose is `[More Information Needed]` — a string its own template emits 39
  times, and which appears in **636,321** repos. Kubernetes' KEP linter does the
  right check with a real AST and then `exit 0`s.
- **Set-membership, not sequence-matching.** markdownlint #394 is a wildcard
  matcher that failed **open**, and MD043 provably cannot express per-kind
  requirements (#32) — our four-content-types shape exactly.
- **The store interface leaks in seven named places** — atomic directory rename
  (Arrow's S3 refuses outright), empty directories, append, `stat`, locking,
  listing, delete. Capability goes in **mixins and a conformance suite**, never a
  flags dict — MLflow's four ABCs, Arrow's 13 *test* predicates.
- **MLflow's `download_artifacts(dst_path=None)` returns the store's own path,
  not a copy.** An agent handed that edits the store in place. `dst_path` must be
  mandatory.
- **A string containment check is defensible under five stated assumptions**:
  the threat is mistakes not attackers (Kustomize says so, and ships a disable
  flag); a kernel layer is the real boundary (runc/k8s run the string check
  *inside* an `openat` loop); `..` is rejected by policy; the name space is a
  mint-time allow-list that cannot express an escape (Nix's `checkName`); and
  symlink *creation* is prevented, not detected. **No precedent pairs a string
  precheck with Landlock specifically** — the precedent is openat2 and fds.
- **`NAME_MAX` is 255 bytes, not characters.** Nix's 211 limit is a derived
  budget reserving room for decorations. Bazel #23576: **provide one symbolic
  accessor and treat the on-disk path shape as private.**

### Gaps the research opened, none yet decided

- **`jsonpath` vs JSON Pointer is now a live question.** handoff §5.1 and
  validator §4.1 both say jsonpath; the failure semantics argue for Pointer. A
  **spec** question to report.
- `jsonpath-ng` is required by handoff §5.1 and validator §4.1, is **not
  installed**, and is **not declared**. `jsonpointer` 3.1.1 and `markdown-it-py`
  4.0.0 **are** installed — as transitive dependencies of `jsonpatch` and `rich`,
  neither declared by us.
- **Nothing in the spec set names the tree-digest algorithm.** in-toto registers
  `dirHash` with an exact shell equivalent precisely so two implementations
  cannot disagree silently.
- **Garbage collection between an artefact and its verdict is unsolved
  everywhere.** OCI distribution-spec#378 open since 2023; zot#4271 is one
  dangling reference that silently disabled GC estate-wide. Nix's direction —
  roots point *at* content — makes the harmful case unrepresentable.
- **[STALE — the file has grown since; jsonnet must now come *out* and `ruamel.yaml` go in. Not re-measured in the 2026-08-29 pass]** `agent_sys/pyproject.toml` declares only `pyyaml`, `packaging`, `pydantic`.
  **jsonnet, jsonschema, and jsonpath are all missing.**
- `claude-agent-sdk`'s wheel is **103 MB** — hard dependency or extra is undecided.
- **[SUPERSEDED — `docs/design.md` O2 is closed by deletion; neither binding is imported anywhere]** `_jsonnet` ships **no Linux aarch64 wheel**; `rjsonnet` ships fourteen.
- `agent-task-graph-prior-art.html` in the materials directory is **`task_graph`'s**
  prior art (RCPSP, the two waiting sets, HiveMind, A2A). Belongs to module 4.

## Module 3 — validator. What its research settled

Two surveys plus my own probes:
`agent_sys/scratch/design/findings-validator.md` (mine),
`findings-validator-isolation.md` (C10/C11/C21/C19/C5),
`findings-validator-registry.md` (registry, composition, skips, cost). Probes in
`scratch/design/validator/`, `probes-validator/`, `probes-validator-registry/`.

### Four places the spec is contradicted by the source it cites

**Findings to report, not edits to make.** None changes an acceptance criterion;
all four were verified first-hand against the implementation the spec names.

- **§6.2: "Inspect AI … Nesting is not permitted" — it is.** `_multi.py`
  (0.3.260) is 69 lines with **no depth check**; a 3-deep composite constructs
  and executes. The one-level rule survives, but rests on **OpenAI** (which
  forbids it *structurally*, by omitting `Multi` from the nested union in the
  schema) and **Gatekeeper**, not on Inspect.
- **§6.2: "DeepEval's `DAGMetric` is the model" for one level deep — it is
  unbounded.** `DeepAcyclicGraph.__init__` imposes two rules, neither a depth
  limit.
- **§6.2's reducer names `all` / `any` are ours, not adopted.** Inspect's
  registered set is `collect, at_least, pass_at, pass_k, max, mean, median,
  mode`; `multi_scorer([...], "all")` raises
  `LookupError: all was not found in the registry`.
- **§6.0's reading of Dagster #16569 is backwards.** It is not users conflating
  severity with blocking — it is Dagster **deliberately decoupling** them. And
  `AssetCheckResult(passed=True, severity=WARN)` is constructible then **ignored
  at every consumption site**: a qualified *pass* is structurally unrepresentable
  there. Ours is decided statically, so their stated objection does not bind us.

### Measured here, first-hand

- **`runtime_checkable` Protocol cannot enforce criterion 1.** `issubclass`
  raises `TypeError` outright (non-method members); `isinstance` is
  **presence-only** — `strength=None`, `strength="STRONG"`, and `brief=7` all
  pass. **`inputs="trace"` is the specific trap**: a bare string is iterable, so
  one input kind silently becomes five characters. pydantic catches all three.
  **The admission gate is a pydantic model over the spec record, not an
  `isinstance` over the implementation.**
- **The composite's two reduce axes genuinely disagree.** Criterion 4 returns one
  verdict *per handoff*; criterion 13's reducers reduce *over validators*. Over
  all sixteen 2×2 grids × 3 reducers, **22 of 48 pairs disagree**. Two further
  measured facts: a ragged grid makes `k` incoherent (a `k` meaningful for one
  handoff is unsatisfiable for another), and **`all([])` is `True`** — a handoff
  *no member validator takes* passes. Vacuous truth arriving from the standard
  library is the exact silent pass the system exists to prevent.
- **Inspect's shape is the only one that satisfies both criteria** — reduce each
  key *across* scorers, so the per-item axis survives and a composite stays
  type-substitutable for a leaf. But Inspect **rejects mismatched keys**
  outright, and its keys are epochs of one sample; **ours legitimately differ per
  member**, so that rejection would forbid a composite §4.1 permits. The naive
  third shape — reduce to a scalar and broadcast — **records `False` against a
  handoff every member passed**.
- **The "already validated" skip has no sound cache key today.** Five key schemes
  tested against six changes: **"the validator's code changed" is a stale HIT
  under every one**, because implementation source is in no spec file. And it
  collides with §9.3's "`version` is maintenance metadata, nothing at runtime
  reads it". Either the skip reads `version`, or it digests the implementation,
  or the skip is narrower than criterion 7 reads.
- **Criterion 19 has exactly one free home.** `HandoffStatus` has two verdicts
  and `seal` refuses a third; `check_if_latest_valid` is `status is VALID`, so a
  `WEAK_VALID` would **silently block every consumer**. `HandoffVersion` has no
  field to hang a qualifier on. The qualifier goes in the validation record
  beside the artefact — the `validation.yaml` handoff design §4.1 already places
  outside the digest.
- **Criterion 11's fail-closed direction is inverted** relative to `handoff` §7
  and `env_mgr` §4.3. There *contained* means allow, so unresolvable → deny.
  Here *contained* means **reject**, so unresolvable must be treated as inside.
  **Copying `check_contained` and negating at the call site would accept a
  dangling validator symlink.**
- **Only `realpath` + trailing separator gets criterion 11 right** — five checks
  × six layouts, wrong answers 3 / 2 / 2 / 1 / **0**. §9.1 *sanctions* symlinks,
  so a link in a neutral package pointing **into** the producer's zone is
  reachable by construction; all three lexical checks accept it.
- **Criterion 11 also needs the graph, which the registry does not have.** A zone
  is per-*task*; a validator registers once, and may be outside task A's zone and
  inside task B's. Load-time or dispatch-time is unresolved.
- **Criterion 5's "pool" is ambiguous, and the literal reading breaks.**
  `Scheduler.pools` comprehends over the **whole `TaskStatus` enum**, so adding
  `INPUT_VALIDATING` / `OUTPUT_VALIDATING` **creates two index pools by
  construction** — a test asserting "no validator occupies a pool" over it fails
  on a correct implementation. Read it as *resource* pool; the real assertion is
  `task_graph` criterion 40's one-lease-across-three-phases. (Those two statuses
  are specified in `task_graph` spec §3.2.2 rev. 12 and **absent from
  `models.py:44`**, which still has eight members.)
- **Criterion 5 is assertable because the runner never returns to the scheduler
  between phases.** Exactly three surfaces, all spy-able: `runner.start`,
  `resource:<name>.take`, `policy.select`. Note `select` fires **more often than
  there are dispatches** (4 passes for 2 tasks), so the assertion is "no
  validator id appears in any `select` argument", not a call count.
- **The static half of criterion 5 must walk the AST, not grep the source.**
  `"scheduler" in runner.py` is **True today** — two docstring mentions. An AST
  walk over names, attributes and imports returns 0. `test_authority.py`'s
  existing static check *is* a substring grep, so copying it naively yields a
  test that fails for the wrong reason.
- **Criterion 21 cannot be tested by a directory check.** A fresh `mkdir` closes
  **one** of six leak channels: `/tmp`, `os.environ`, inherited `cwd`,
  `sys.path`, `$HOME` and same-path reuse all still carry the producer's
  leavings. Inherited fds are clean (CPython defaults non-inheritable).
- **Isolation is cheaper than the interpreter it wraps.** A full
  `unshare --user --mount` + bind + remount-ro costs **13.8 ms**; `python3 -c
  pass` costs **61–66 ms**; docker costs 786 ms. **There is no cost argument for
  reusing an environment**, and the roadmap item contemplating a §8.2 relaxation
  should be told this number. All variants ran **unprivileged**, no `bwrap`.
- **The C21 conformance test is ~40 lines and sub-millisecond.** Four strategies:
  reuse-no-clean FAIL, `rmtree`+recreate **FAIL on path identity only**, fresh
  numbered dir PASS, prune-declared-paths-only FAIL. Strategy B is the design
  question — reusing the absolute path keeps any path the producer baked in
  resolvable, which is handoff criterion 17's locality dependence.
- **pytest 9.1.1 renders a qualified pass and still exits 0.** `1 passed, 1
  xfailed, 1 xpassed`, exit **0**, green bar. Reproduced twice, independently.

### From the prior-art surveys

- **Nobody keeps an answer key secret from a co-located party at equal
  privilege.** Three mechanisms exist — physical absence, server-side scoring,
  OS privilege separation — and "told not to look" is not one. BIG-bench's canary
  is convention and is defeated; **HELM has no hidden split at all** and does not
  support the citation it usually gets; **lm-eval-harness's decontamination code
  is orphaned** (zero callers) behind a doc that still describes it.
- **SWE-bench's absence was defeated for two years by git objects.**
  `git remote remove origin` leaves the fix commit in `cat-file
  --batch-all-objects`; reproduced here. **The fix is the lesson**: current
  `main` ends the clone with `[ "$COMMIT_COUNT" -eq 0 ] || exit 1`. **Absence is
  not a property you write; it is a property you assert.**
- **The grading channel leaks independently of the filesystem.** Whitehill
  reached **rank #4 of 848** on a real Kaggle competition from repeated scalar
  feedback alone, no classifier. **Feedback-repetition limiting is a second
  isolation axis, and validator §8 does not address it.**
- **One synchronous `PreToolUse` hook is both the spy and the denier** — it logs
  every attempt before deciding. The async form cannot block, so logging-only is
  unavailable. But **`Bash{'command': 'python3 reader.py'}` returns ALLOW**:
  `env_mgr` §4.1's finding, reproduced against SDK 0.2.144. **The hook is the
  attributable layer; `env_mgr`'s allow-list is the enforcing one.**
- **The SDK has no "producer frame".** No field over any `HookInput` type
  suggests stack, caller, origin or parent; `agent_id` is **optional and absent
  on the main thread**. So each phase must be a subagent or its own session, or
  criterion 10 is not testable as written. `SubagentStart`/`SubagentStop` carry
  `agent_id` as **required** — structurally `test_authority.py`'s `<agent>`
  markers.
- **Go's `internal` is the strongest idea and inverts for us.** It checks
  lexically, then resolves symlinks **only to widen** access — a link can never
  turn an allowed import into a denied one. Ours is a *rejection*, so resolution
  must only ever move a verdict toward *accept*. **A deliberate inversion of Go's
  risk posture, to be stated as such.** ESLint (lexical, `preserveSymlinks: true`
  hardcoded with a TODO) and import-linter are measurably defeated;
  **dependency-cruiser is the only clean precedent** — resolved by default.
- **pandera is the only system that checks the args table against the
  signature, and it does it at registration.** Four lines of `inspect.signature`,
  shipped because of #480 — *"checks that look configured but ignore their
  inputs"*, the most transferable phrase in the survey. dbt's equivalent is off
  by default and broken **both ways** (#11792 always warns without a cached
  manifest; #12574 stops warning after a second `dbt parse`) — **a signature
  check reading from a cache of a previous parse is worse than none.**
- **dbt is the precedent for binding-supplied args**, and Gatekeeper is a third
  answer: the implementation ships an **OpenAPI schema for its own args**. If our
  args live in the binding, `validators_for(kind)` changes type and §8.3 must
  decide whether two kinds may bind one validator with different args. Putting
  them in the validator spec makes the question not arise.
- **`--validation-strict-level` must be structurally unable to reach a verdict.**
  ESLint's `--quiet` erased an **error** and flipped the exit code (#14202); the
  fix was a variable split — *"Errors and warnings from the original unfiltered
  results should determine the exit code."* Then it was **reintroduced one layer
  down** by an RFC that explicitly claimed verdict-neutrality, and #19625 found
  the missed case **22 months later**, closed WONTFIX. **Enumerating the
  interactions is not the lesson; the variable split is.**
- **Four systems, four names for the third outcome, and none of them is
  "pass".** pytest exit **5**, Bazel's `NO_STATUS = 0` *before* `PASSED = 1`,
  GitHub's skipped-workflow checks staying **Pending and blocking merge**, SARIF's
  `notApplicable`. JUnit XML is the counterexample: **pass is the structural
  default**, so a producer that forgets `<skipped/>` emits a pass. **Never let
  PASS arise from an absent field.**
- **SARIF types the pass rather than grading it** — `kind ∈ {pass, open,
  informational, notApplicable, review, fail}`, and if `kind ≠ fail` then `level`
  SHALL be `none` and `rank` SHALL be absent. Its `null`-vs-`[]` rule for
  suppressions, with a run-level uniformity invariant, is exactly our
  history-absent-vs-history-empty distinction. **GitHub code scanning drops all
  of it** — `rank` and `result.kind`, 0 hits.
- **Bazel gives the cache-key contract sentence**: *"if the work to be performed
  by the execution of this action changes, the key must change"*, implemented as
  the tool's **bytes** plus a **hand-bumped GUID per action class**. REAPI does
  **not** name the tool. dbt solves our exact problem by digesting `macro_sql`
  transitively — and the whole #6455/#5202/#8526 family is one bug: **the
  dependency edge was not recorded, because dbt infers edges by parsing.** Ours
  are declared; that is a real advantage to claim deliberately.
- **Bazel reuses a cached PASS and re-runs a cached FAIL** (`--cache_test_results=auto`).
  That asymmetry answers "is a reused verdict folded into the phase result".
- **Nix separates coherence from truth, and says the second is unavailable**:
  *"there is no way to audit a build trace entry except for by performing the
  build again"*. So `--validation-strict-level` is a **trust policy over recorded
  verdicts**, not a correctness mechanism. Nix also specifies its sandbox as
  **what the process can observe, not which mechanism** — a better shape for
  criterion 21 than a mechanism list.
- **Only Bazel and Concourse test freshness, and their test names are our
  criterion**: `test_sandbox_undeclared_deps`,
  `test_sandbox_old_contents_not_reused_in_consecutive_builds`, and
  `It("doesn't mount its file system into the next task")`. Four independent
  "it said fresh and it wasn't" bugs in tox and nox — the sharpest being a nox
  staleness check that returns `True` unless an **undocumented env var** is set.
- **pytest's freshness comes from allocation, never cleanup**, and cleanup is
  explicitly best-effort. **Never make a freshness guarantee depend on a teardown
  succeeding.**
- **Nobody surfaces "this check never ran".** dbt's `result:` selector has no
  `never_ran`; Semgrep's `--list-rules` has been open **four years**; Bazel has
  three graphs and none is "what executed". **Stryker is the exception and the
  model** — `Pending` / `No coverage` / `Ignored` as first-class states, and
  **two denominators published side by side**. Great Expectations shows the
  failure: `success = successful == evaluated` is a tautology, and an **empty
  suite reports `success=True`**.
- **Airflow tombstones rather than deleting** (`REMOVED = "vanished from DAG
  before it ran"`, bidirectional, never while running) — the answer to "a
  validator that ran, then was deleted". Its *asset* orphanage carries the
  warning: **false-positive deadness from an unenumerated reference kind**
  (#58058).
- **Ordering by a declared cost tag has no prior art anywhere surveyed.** Every
  system consuming a cost signal does admission control (Bazel `size` →
  `ResourceSet`, never ordering), measured bin-packing (pytest-split), or a
  human-declared dependency graph (CI `needs`). **If we order by `cost` we are
  ahead of the prior art, not behind it** — and the declared tag's failure mode
  is checked only after the fact, advisorily: Bazel warns (off by default,
  false-positives under variance), pytest-split **silently substitutes the
  population mean and silently discards orphans**.
- **Bazel is moving *away* from rich default error messages** (#25941, #25933,
  #25940) toward one greppable line, because multi-line *"makes grepping
  harder"*. That cuts against module 1's enumerate-the-candidates conclusion —
  the right home may be an opt-in flag.
- **JPMS confirms module 1's two exception classes** (`FindException` vs
  `ResolutionException`) and adds a warning: **a separate consistency pass only
  catches what you put in it** — JPMS omits export-reachability and gets exactly
  the silent-success mode the closure pass exists to avoid.

### Gaps this research opened, none yet decided

- **The composite's reduce axis, `k` on a ragged grid, and a handoff no member
  covers** — three choices the spec does not make and the design must.
- **No sound cache key for the "already validated" skip** without either
  contradicting §9.3 or digesting implementation source. The largest open
  question in the module.
- **Criterion 11: load-time or dispatch-time?** It needs the producing task's
  zone, which `env_mgr` §4.2 builds at task start. **A spec question.**
- **Is path identity part of criterion 21?** `rmtree`+recreate removes every
  leftover yet reuses the absolute path.
- **Does each phase get its own agent session?** Criterion 10's "producer frame"
  is only representable if it does.
- **`TaskStatus` lacks `INPUT_VALIDATING` / `OUTPUT_VALIDATING`**, which
  `task_graph` spec §3.2.2 rev. 12 specifies. Whose change, and when?
- **`claude-agent-sdk` is undeclared and 103 MB**, and pulls `mcp` + `sniffio`.
  Criterion 10's mechanism is its hook API. Module 1 flagged this; criterion 10
  sharpens it.
- **`env_mgr` has no sandbox implementation** — `grep` for
  landlock/bwrap/unshare/seccomp hits only its spec. So which of criterion 21's
  six leak channels its allow-list closes is currently **unmeasurable**.
- **Does "every validator here was weak" deserve distinct treatment** from "some
  weak, some strong"? That is the case carrying no strong evidence at all.
- **Feedback-repetition limiting** is a second isolation axis, unaddressed in §8.

## Module 4 — task_graph. What its research settled

**This is the only module with a shipped implementation.** `task_graph/*.py` is
at spec rev. 7 and green: **358 tests** in `tests/task_graph`, 423 including
`env_mgr`. Design rev. 10 → 11 covers spec rev. 8–12 and criteria 36–54.

Findings in `agent_sys/scratch/design/findings-taskgraph-mine.md` (18 items) and
`findings-taskgraph-reentrancy.md`, with probes in `scratch/design/taskgraph/`
and `probes-taskgraph-*/`. All gitignored.

**`agent_sys/docs/analysis.md` is superseded** — its conclusions are already in
the spec set and the file is gone. The one authoritative open-questions list is
**`task_graph/docs/spec.md` §10**.

### Measured here, first-hand

- **Adding the two phase states touches exactly five live sites** in
  `scheduler.py` — the `stop` guard, the `complete` guard, dispatch's
  `_move(RUNNING)`, and two recovery demotions — and the spec already gives each
  a determinate answer. **Design rev. 10's argument against a `PHASES` constant
  expires**: it says "every other guard tests a single status", and now three
  guards test the same three-member set.
- **`TaskStatus` +2 makes `pools` ten buckets by construction**; three stay
  load-bearing. This is what `validator` criterion 5 is read against.
- **`TaskMgr` has no forward edge and no index** — `by_status` is a *scan*, so
  the in-module precedent is "a collection query is a scan at this scale".
  Leaf-ness (criterion 53) and the downstream index (criterion 49) are **the
  same missing shape** — a forward edge derived from a stored back edge — and get
  one answer. `is_start and is_end` is **not** a sound leaf test; only the
  absence of children decides it.
- **Criteria 42 and 43 are in tension, and there are two structure-free
  escapes.** A policy reading `parent` satisfies 43 and fails 42. Measured:
  **LIFO on `created_at`** reads no structural field and passes criterion 43's
  worked example — but on a measured counter-case it **abandons an in-progress
  subgraph for a newer unrelated task**, which is the opposite of the sentence
  §5.2 states. **LIFO on readiness recency** gets that case right and is also
  structure-free, but `select(eligible, snapshot)` gives the policy no handoff
  access, so it needs a stamped field, a third argument, or a registry lookup.
- **Changing only the default policy to LIFO breaks 1 of 358 tests**, and that
  one is D15's regression test whose *setup* — not whose assertion — depends on
  dispatch order. The suite is almost entirely order-independent.
- **`created_at` is a total order**: 1000 tasks built in a loop get 1000 distinct
  timestamps (~1 µs clock, ~4 µs per construction). Neither key ties in practice
  on this machine.
- **§3.2.4's own wording already picks the re-entrancy discipline.** §3.2.3
  leaves "drained queue or bounded recursion" to the design; §3.2.4 says the
  cascade goes **"level by level"**. Measured on a diamond: a drained queue gives
  `[A,B,C,F,D,E]`, recursion gives `[A,B,D,E,C,F]`. Level-by-level *is* the
  drained queue, so one option contradicts spec text.
- **A cascade needs a visited set, and `cancel()`'s precondition makes it
  sharp.** On the same diamond with no seen-set, `D` and `E` are reached twice —
  and §3.2.3's precondition for `cancel()` is *"a waiting state"*, which an
  already-`CANCELLED` task is not. **A diamond cascade with no dedupe raises.**
- **Naive recursion dies at 250 levels** — 4 frames per level in an optimistic
  modelled shape, against the default limit of 1000. Real frames per level are
  higher (`validate_assignment`, `persist`, the registry lookup).
- **`extra="forbid"` means a rev.12 store record is unreadable by rev.7 code.**
  Old records load fine into new models; there is no rollback across the change.
- **`test_authority.py`'s static half is a substring grep** over
  `inspect.getsource`. Narrow enough for its three tokens; **it does not
  generalise to criteria 45–48**, whose tokens (`status`, `try_dispatch`) appear
  throughout prose and comments.
- **A `Task` can hold a registry reference two ways, and both are measured.**
  `PrivateAttr` and a field with `exclude=True` both stay out of `model_dump`
  and both survive `model_copy(update=...)` — which `update_task` depends on.
  The trade is a type check (`exclude=True`, via `validate_assignment`) against
  a global relaxation (`arbitrary_types_allowed` on the shared `Model` base).
  **Both return `None` after `model_validate`**, so *rehydrate-on-load is
  forced, not chosen*, and `TaskMgr.resume_system` is the only thing positioned
  to do it. `model_copy(deep=True)` and `copy.deepcopy` silently **clone** the
  registry — the worst outcome, since the task then drives a scheduler nobody
  else can see.
- **`import task_graph` resolves to a different worktree** —
  `infera.aiopt.task.graph/`. Every `.py` is byte-identical today, so 423 green
  still means what it says; it stops meaning it the moment implementation starts
  here.

### The criteria audit — how much of 36–54 is actually specified

**13 SPECIFIED · 3 DEPENDS (37, 46, 49) · 2 UNDERSPECIFIED (44, 50).**
Full table with `file:line` in `findings-taskgraph-criteria.md`.

- **36–43 and 51–54 need no §10 row resolved.** The design can write them.
- **49 depends on two §10 rows at once** — the downstream index *and* cascade at
  the edges. **37's cancelled case depends on a third.** That pair is where a
  design stops.
- **46 has two owners.** §10 lists re-entrancy as undecided; §3.2.3 `:698-699`
  says "the design stage picks one, and the choice is not optional." Two
  documents, two answers about whose choice it is.
- **44 is underspecified because the type does not exist** — semantics at
  `:585-608`, no field list, no schema, and §10 does not acknowledge it.
- **50 is underspecified because its home does not exist**, and worse: see below.

### Read first-hand in the spec set

- **Nothing owns the graph-level load check** criteria 50 and 53 require, and
  **`task_graph` misattributes the deferral.** `:755` says the check is one
  "`closure` spec §4.1 defers to 'the system whole task'". Read whole, closure
  §4.1 defers it to *nobody*; the phrase lives at `closure/docs/spec.md:242`
  inside an **open question** — "it has to live somewhere. The likely home is the
  system whole task." Main design §6.3 declines it in terms ("it is still not
  this pass's"). **The two documents that touch it each think the other is
  holding it.** Since the catalogue is static, the check is over *task specs*,
  not runtime `Task`s, and the composition root is the only point where every
  spec is present and nothing has run.
- **The subgraph declaration format is absent everywhere.** Three passages assert
  a subgraph is "declared in the task's spec"; none says how. The closest is one
  clause of a key list, `closure/docs/spec.md:95`, where the word `subgraph` is
  never expanded. `grep -n "subgraph\|is_start\|is_end\|parent" docs/design.md`
  returns **zero hits**, and `find . -name '*.json' -path '*schema*'` finds no
  schema on disk. Unanswered: how subtasks are listed, how `is_start`/`is_end`
  are marked, how edges between subtasks are expressed, and how a subgraph's
  handoffs connect to the parent's `inputs`/`outputs`.
- **`Permissions` is named as a type by §3.2 and defined nowhere.** `agent` §3.2
  and `closure` §1.1 both point back here. **`Task.resources` is the precedent
  for the answer that costs no import edge**: carried by name, interpreted only
  by `env_mgr`. What the consumers actually force is **narrower than it looks**:
  `env_mgr` §4.5's granted read set has four rows and `Permissions` supplies
  **row 3 only** — a named path set beyond the zone and the system default. The
  **write side needs no contribution at all** (§4.5: "may not write outside its
  zones… no exception"). And §5.1 makes the recursive-subtree half **derived from
  the nested layout, not stored**, so criterion 44's "covers its subtasks
  recursively" is a property of the storage layout rather than of the field. The
  one thing it must be able to do is what `closure` load check 6 needs: be
  **queryable against a specific handoff**, distinguishing read from write.
- **Criterion 51 needs the closure catalogue, and main design §7's prohibition is
  scoped to *the scheduler*** — "the scheduler … has no name for a spec registry".
  `replace_with` is a `Task` transition, so resolving `closures` adds no
  scheduler→spec edge and leaves `test_authority.py` untouched. Narrow, real, and
  worth saying out loud.
- **Criterion 41's "reported" is already owned by the validator side** —
  `PhaseOutcome` folds `ran`/`reused`/`skipped`, and `TaskRunner` is the seam.
  Consequence: **`FakeRunner` must grow the three-phase behaviour** or criteria
  39–41 are untestable, even though the real runner is out of scope (§1.2).
- **Four of spec §10's open questions are prerequisites of criteria 45–52**, not
  refinements. Re-entrancy is explicitly the design's to pick; the downstream
  index is mechanism with no spec content to contradict. But **cascade at the
  edges** and **`is_end` under a cancelled subgraph** are semantics — answering
  "stop a `RUNNING` task" makes `cancel()` asynchronous, which §10 itself flags
  as a signature change. Those two are reported, not decided.

### From the nesting survey

- **No surveyed system lets a parent hold a lease across its children.** Our goal
  has precedent in seven systems; our *form* — a blanket load-time rule — does
  not. **Airflow's `SubDagOperator` is the disaster and the record is exact**:
  `class SubDagOperator(BaseSensorOperator)` holds a worker *and* pool slot; its
  own docstring admits "can occupy a pool/concurrency slot… to avoid potential
  deadlock"; #14338 is the user report; **removed** by PR #41390, merged
  2024-08-13, **-3918 lines**.
- **Airflow shipped a load-time check for exactly our case and it is too narrow,
  which is the argument for our blanket rule.** `_validate_pool` filters
  `Pool.slots == 1` — a **2-slot pool, parent holding 1, child needing 2
  deadlocks and parses clean**. Airflow detects *a* deadlock; criterion 53
  forbids *the class*. Copy their message shape: it names both sides and ends
  with a consequence, *"The subdag tasks will never run."*
- **Prefect is our "structural `RUNNING`" almost exactly** — the engine creates a
  parent task run, records `parent_task_run_id`, and never runs it. It **also has
  our deadlock and declined to fix it**: PR #21800, *"It does not change the
  deadlock semantics — it just makes them debuggable."*
- **Dagster is the strongest form**: a `@graph` produces **no `ExecutionStep` at
  all`**. Measured, 4 nodes → 2 steps. **Temporal is a third answer unavailable
  to us** — a worker slot is held for a Workflow Task (milliseconds), not the
  workflow's life; our leaf holds a GPU lease because the work is long.
- **No *workflow* engine separates ordering policy from graph structure** — not
  by a precomputed key, not by priority inheritance, in any of Argo / Dagster /
  Airflow / Prefect / Nextflow / Snakemake. **But kube-scheduler does exactly
  that, and it is a direct precedent for `ready_since`.** `PrioritySort.Less`
  sorts on priority then `GetTimestamp()`, and that timestamp is
  `QueueingParams.Timestamp` — *"The time entity added to the scheduling
  queue"*, stamped by `p.clock.Now()`, **not** the object's creation time. The
  sort plugin never reads the object graph.
  It also answers the stale-derived-field risk in three parts: a **second,
  immutable** `InitialAttemptTimestamp` exists for measurement (*"It shouldn't be
  updated once initialized"*) so the sort key and the first-entry record are never
  conflated; refreshing on requeue is deliberate and commented (*"Refresh the
  timestamp since the pod is re-added"*); and **preserving** it is available per
  path with the reason inline (`preserveTimestamp = hadUnschedulableOrErrorPods`,
  *"so that subsequent scheduling attempt triggers preemption immediately"*).
  **The staleness is not avoided — it is made an explicit per-path decision with
  the reason next to it.** Still NOT REACHED: Slurm's multifactor age factor and
  YARN's `SchedulableEntity.getStartTime()`.
- **Airflow #35689 (open) asks for exactly criterion 43**, and names
  `SubDagOperator` as the only workaround. **The one mechanism that gave Airflow
  depth-first is the one that deadlocked and was deleted.**
- **A trap for the criterion 43 test**: in the agent's first probe two
  structure-blind keys appeared to *pass* only because the tiebreaker was `t.id`
  and the unrelated task's name sorted late. A real test has the same hole.
- **Subgraph containment is static and one-directional everywhere.** Dagster
  errors on an unmapped declared `GraphOut` but **silently discards** an inner
  output nobody mapped; consuming one from outside fails as a *type error about
  `None`*. **Criterion 50 checks the inverse direction, which nowhere else can
  even express** — ours is representable only because handoff ids are global in a
  flat `HandoffMgr` namespace. Scoping ids to the subgraph is the alternative
  *with* precedent, and it is a spec question.
- **Leaf-ness is O(1) by a type test everywhere else.** Deriving it from
  `is_start and is_end` has no precedent, and §3.2.1 states the markers as a
  **consequence** of leafness — using them as the *test* inverts that, sound only
  if no non-leaf can carry both. The spec does not say.
- **Argo #16376**: `NodeID` is FNV-32a of the node name; a collision makes a node
  its own ancestor and the controller stack-overflows. Applies to us if `parent`
  is ever assigned without an ancestor check at the link site.

### From the cascade survey

- **Nobody makes a cascade atomic, and every system names the intermediate
  state** — k8s `deletionTimestamp`/`deletingDependents`, Kotlin `Cancelling`,
  Airflow `RESTARTING`, Dagster `CANCELING`, Temporal `CancelRequested`. So spec
  §10's *"half-cancelled is a state nothing describes"* is true **only of ours**.
- **Airflow answers the `RUNNING` question with a caller-supplied parameter**, and
  it is the counter-example to §10's claim that "stop it" forces an async
  signature: `prevent_running_task` either raises and refuses the whole
  operation, or sets `RESTARTING` — **a request, not a wait** — so `clear` stays
  synchronous. "Skip it" is offered by nobody. Note its origin (#54379) is a
  *concurrent-operator* hazard, not a correctness one.
- **Temporal's cancel/terminate split is the authority model**: cancel is
  cooperative and refusable, terminate is unilateral. Its cascade is a **system
  workflow reached by a signal, after the parent closes — not atomic by
  construction**, one level only, and a partial cascade is directly observable.
  **#604**: a cascade built as N independent one-level steps **fails silently at
  the level where a step is dropped**, and the parent's status says nothing.
- **k8s is the reverse-index model: eager single-writer index to find candidates,
  authoritative re-read before the destructive act.** The source says outright
  that getters "can be inconsistent" and that dependents may change "the moment
  this function returns". An edge naming an unobserved owner becomes a **virtual
  node** plus a scheduled verification — never dropped, never fatal. A **dangling
  ownerRef is treated as absent**.
- **All three systems with a reverse index maintain it eagerly and symmetrically,
  in the same statement** (Airflow `_set_relatives`, Dask `add_dependency`, k8s
  `insertNode`). **Nobody computes it on demand.** So our question is not
  eager-vs-lazy but **which write path is the single edge-creation site, and
  whether `submit` and `update_task` both funnel through it.**
- **Dask's `stimulus_cancel` is our design written the obvious way, and it is
  broken** — recursion over the reverse index, no visited set, no bound.
  Reproduced on a live cluster: it blew inside the scheduler's event loop **while
  the client printed `cancel OK`**. The caller was told the cascade succeeded
  over a silently half-cancelled graph.
- **Airflow already solved it and left the reason in a comment**: an explicit
  loop *"since Python has significant limitation on stack level, and a recursive
  implementation can blow up if a DAG contains very long routes"* — twenty lines
  with **BFS level by level**, **a visited set**, and an optional depth bound.
  That is spec §3.2.4's wording implemented literally.
- **A parent veto over a running cascade is unprecedented.** Veto exists
  everywhere but always as child→parent (k8s `BlockOwnerDeletion`), self (trio
  `shield`, asyncio `uncancel`), or parent-declared-in-advance (Temporal
  `ABANDON`). **No system consults an ancestor at cascade time.**
- **Kotlin gives the upward-reporting answer**: `cancel` takes only a
  `CancellationException`, *"which does not lead to cancellation of its parent"*,
  while a child's **failure** does. **Two causes, two upward paths** — a cancel
  reports nothing upward.
- **Upward reporting has three shapes and the third survives an incomplete
  cascade best**: enumerate-before-acting (Airflow's `dry_run`, structural — the
  same code computes and acts), aggregate-after (measured **lossy** on both
  `asyncio.TaskGroup` and trio: two children raise, the group carries one), or
  **a state plus a reason on each unit** (k8s, Airflow, Argo's per-node reason
  string). Nothing has to aggregate.
- **k8s #77081 sharpens the re-entrancy question**: the danger is not recursion,
  it is **a repeat cascade request producing no change event and being silently
  dropped forever**.
- **Nobody else draws our cancel-vs-invalidate line, and Airflow actively
  conflates them** — one `clear` both terminates a RUNNING task and resets a
  `SUCCESS` task to re-run, separated only by caller booleans **neither set by
  default**. §3.2.4's distinction is a claim to make deliberately, not a gap.
- **Argo does not walk the graph at all** — a `Shutdown` field plus a full scan
  per reconcile. Its sweep body is a catalogue of holes found one at a time.
  **The cost of a scan-based cascade is one sweep rule per state.**
- **Celery's revoke is not durable**: a bounded `LimitedSet` (50 000, 3 h), and
  revoking an unknown id returns `ok('terminate: tasks unknown')` — no error, no
  record that it missed.

### From the re-entrancy survey

- **Every mature system chose queue-and-drain or outright refusal. Bounded
  recursion on a callback cascade has no precedent** outside database trigger
  engines, where the cascade is user-authored SQL rather than the engine's own
  control flow.
- **asyncio's drain snapshots the queue length before the pass** —
  `ntodo = len(self._ready)` — so a callback scheduled by a callback is deferred
  to the next pass *by construction*. The comment is the design statement: *"This
  is the only place where callbacks are actually \*called\*."*
- **Our `_dispatch_again` is a boolean, not a queue.** asyncio's and Akka's are
  *bounded* passes over *real* queues; ours is an *unbounded* pass over an
  implicit one. Whether coalescing can lose work turns on `_dispatch_pass`'s
  idempotence.
- **Redux permits `dispatch` inside a subscriber and documents the infinite loop
  it buys** — our exact `transition → dispatch → completion → transition` shape.
  **Redux and RxJS both snapshot the listener collection before iterating**; a
  cascade mutates the task set while walking it, the same hazard.
- **SQLAlchemy runs two disciplines against one flag**: explicit re-entry is
  refused loudly, implicit re-entry is silently suppressed. **Refuse what a
  caller wrote; suppress what the system triggered.** And #13485 is the flag's
  own failure mode — a `_flushing` set outside `try/finally` bricks the Session
  permanently. **Our `_in_dispatch` is already inside `try/finally`, and that is
  load-bearing enough to deserve a test.**
- **Django is the no-discipline control and was closed `wontfix`.** A `post_save`
  calling `save()` reaches 245 levels then raises **inside
  `django/db/models/lookups.py`** — the crash blames unrelated ORM machinery.
  That is the diagnostic cost of letting a cascade recurse.
- **Akka drains system messages to empty before every user message** —
  *"don't ever execute normal message when system message present!"*. **A cancel
  cascade is structurally a system message** and should not queue behind ordinary
  dispatch work. Its batch bound of 5 exists for *fairness, not termination*.
- **SQL Server is the only clean numeric bound (32 levels) and it fails
  silently** — the trigger just terminates, no error number found. **A bound that
  fails silently hides the bug.** No source surveyed gives a *derivation* for its
  number, so a bound is a tripwire, not a capacity plan.
- **systemd bounds cycles, not depth** — `transaction_verify_order_one` recurses,
  uses `generation` as a visited marker, and on a cycle logs "Found ordering
  cycle" and deletes a job to break it. **Cycle detection is a third option the
  spec's binary framing omits.**
- **Kubernetes controller-runtime never recurses — always requeue**, with
  exponential backoff and a global rate limit. The backoff is the part our
  trampoline has no analogue for.
- **A drained queue needs its ordering rule stated**, and the two canonical
  answers differ: asyncio guarantees global FIFO, Akka only per sender–receiver
  pair, explicitly non-transitive.

### Gaps this research opened, none yet decided

- **Which structure-free key expresses depth-first**, and what it costs the
  `select` signature. Or whether criterion 42 is what should move.
- **Whether the cascade queue is separate from the dispatch trampoline**, which
  Akka's system-message rule argues for and nothing in the spec mentions.
- **Whether a half-cancelled state gets a name.** Every surveyed system has one;
  the spec says ours is "a state nothing describes". Naming it is a spec change.
- **Whether the cascade reports by aggregating or by writing a reason per task.**
  The survey favours the second; nothing in the spec chooses.
- **Whether coalescing in `_dispatch_again` can lose work** — a question about
  `_dispatch_pass`'s idempotence, not yet examined.
- **Where `Permissions` is defined**, given that both import directions are edges
  the design forbids.
- **What a cascade does on reaching a `RUNNING` task**, and whether it is atomic.
  A spec question: the first answer changes `cancel()`'s signature.
- **`is_end` under a cancelled subgraph** — §3.2.1's markers assume completion.

## Carried into module 5 — agent

Not research; things already established that module 5 must not rediscover or
contradict. `agent/docs/spec.md` rev. 4, §8 — 16 criteria.

- ~~**The spec-to-design consistency pass is done for handoff, validator and
  task_graph, and NOT for module 1.**~~ **Done** — commit `698d9d2`; all four
  pairs have now had a section-by-section pass.
- **Permissions are not the agent's.** `agent` §3.2 and `task_graph` §3.2.2 both
  say so; `task_graph` design §3.5 carries the type, `env_mgr` interprets it.
  Criterion 44 is task_graph's, and module 5 inherits rather than restates.
- **`claude-agent-sdk` is undeclared, ~~103 MB, and pulls `mcp` + `sniffio`~~.**
  Flagged by modules 1 and 3. §5 makes it the first backend, so module 5 is
  where hard-dependency-versus-extra actually has to be decided. **Both figures
  were wrong** — measured in module 5 at **376 MB installed and 26 extra
  packages**, plus ~1.3 s to import.
- **The SDK has no "producer frame"** — measured in module 3. `agent_id` is
  optional and absent on the main thread; `SubagentStart`/`SubagentStop` carry it
  as required. So whether each phase is its own session is a module 5 question
  with a module 3 consequence (validator criterion 10).
- **`Bash{'command': 'python3 reader.py'}` returns ALLOW** — the hook is the
  attributable layer, `env_mgr`'s allow-list is the enforcing one. `agent` §5.3
  already says the hook is "a first gate, not the boundary"; the design must not
  quietly upgrade it.
- **`Agent` the runtime record is `task_graph`'s** (spec §3.3, design §3.5) — id,
  spec name, `task_id`, `HandoffRef`s. `agent` §7 is about what the *system
  records*; do not design a second agent object.
- **Logging is o11y's, not the agent's** — `agent` §6 and `ROADMAP.md` §1.

## Module 5 — agent. What its research settled

Findings files: `scratch/design/findings-agent-mine.md` (my own, first-hand
against the installed SDK), `-backends.md` and `-transform.md` (two prior-art
surveys). Probe venv `scratch/design/probes-agent/`; survey workspaces
`scratch/design/survey-backends/` and `survey-transform/`.

**Both surveyors' harnesses forbid subagents writing report files.** Neither
routed around it; both reported as message text and the two files are
transcriptions. Their evidence artefacts are on disk and every line reference
re-resolves.

### A retraction, recorded because the failure mode keeps recurring

I claimed the recorded finding *"LiteLLM #20885, where a missing cell read as
`False`"* was wrong. **It was not; I was.** I read only the first 2500
characters of the issue body, saw Case 1's `get_llm_provider` exception, and
concluded the characterisation was a mischaracterisation. The full body says
plainly (`:85`, `:106`, `:124`): `_supports_factory returns: False (or
default)`, "Functions **incorrectly return `False`**". The duplicate-key
mechanism is the *cause*; the `False` is the *effect*.

**Third miscited upstream reference in this project** (module 2 retracted OPA
#7806; this one twice over — the original citation's dimensions were also
wrong). Every instance is the same: a conclusion drawn from a partial read.

Corrections that stand:

- **`3040 × 35` → `3212 × 37`**, measured on
  `model_prices_and_context_window.json` @ `40423e6`. The stronger figure:
  **90.4% of `supports_*` cells are absent** (11356 filled of 118844), and every
  absent cell resolves to `False`.
- **`utils.py:2568` cites `#20885` in a source comment** — cite the code, not
  only the tracker.
- **The fix's cost is the real argument**: one ambiguous cell was remedied with
  two additional lookup paths plus an `except Exception:` that logs at debug and
  *also* returns `False`.
- **SQLAlchemy did not remove its capability matrix.** ~40 `supports_*` flags
  remain on `DefaultDialect`. What changed, for `supports_statement_cache` only,
  is that `self.__class__.__dict__.get(...)` deliberately bypasses the MRO so
  the flag cannot be inherited, and silence warns. **Rewrite our citation** to
  the narrower rule: *a capability a third party must actively verify may not be
  inherited, and the absence of a declaration must be loud, not defaulted.*
- **PEP 249 `NotSupportedError` is confirmed but is the weaker form** — it sits
  *alongside* capability introspection, it did not replace it. Cite it as
  evidence the raise-channel is normal, not as evidence against matrices.

### Measured here, first-hand

- **Spec §5.1's 13-row capability table is accurate**, verified row by row
  against `claude-agent-sdk` 0.2.144. It is also incomplete: `stop_task`,
  `get_context_usage`, `set_permission_mode`/`set_model`, `max_budget_usd` and
  `task_budget` are all real surface it predates. The SDK has grown its own
  *task* vocabulary (`TaskBudget`, `TaskStartedMessage`,
  `TERMINAL_TASK_STATUSES`) that **collides with our `Task`** by name.
- **Our `PreToolUse` hook must never return `allow`.** `types.py:2118` — "a
  `PreToolUse` hook returning an *allow* decision also skips this callback"
  (`can_use_tool`). `permissionDecision` is `NotRequired` and omitting it flows
  through normal evaluation (`hooks.md:342`). So only `deny` and *omit* are safe;
  the natural phrasing "return allow when the check passes" silently disables
  every downstream check. **Nothing in the spec says this.**
- **Permission evaluation is six steps**, not one gate: hooks → deny rules → ask
  rules → permission mode → allow rules → `can_use_tool` (`permissions.md:21-45`).
  Hooks run first and a hook *deny* applies even under `bypassPermissions`.
  `allowed_tools` does **not** constrain `bypassPermissions`. Subagents inherit
  the parent's mode and cannot override `bypassPermissions`/`acceptEdits`/`auto`
  — **a subagent is not a weaker principal.**
- **The dependency is much larger than recorded.** Wheel 99 MB (recorded 103 MB,
  close), **installed 376 MB** with 328 MB of it `_bundled/claude`, a single
  executable; **26 extra packages** (recorded: "mcp + sniffio") including
  `cryptography`, `uvicorn`, `starlette`, `opentelemetry-api`; and **~1.3 s to
  import** (1.437/1.272/1.333 s). The import time is what decides
  hard-dependency versus extra.
- **The wheel ships the reference docs** — `python.md` (189 KB), `permissions.md`,
  `hooks.md`, `sessions.md`, `cost-tracking.md`, `subagents.md`. An official
  source, offline.
- **Criterion 9 needs no message counting.** `ResultMessage.terminal_reason` is
  `aborted_streaming` / `aborted_tools` for an interrupted turn
  (`types.py:1342`), so the interrupted submission's result is self-identifying.
  It is `None` on a fatal session failure, so the drain still needs a
  terminating condition that does not assume the field.
- **§5.2's drain caveat is now first-hand** — `python.md:617` states it verbatim,
  and the mechanism is visible: `interrupt()` sends a *control request*
  (`_internal/query.py:684`) on a channel separate from the message stream, and
  `_send_control_request` waits for an ack, so a synchronous `stop()` over it is
  implementable.
- **A docstring would have made us add a constraint that does not exist.**
  `interrupt()` says "(only works with streaming mode)" and `python.md:31`
  tabulates interrupts as unsupported in single mode — but **both** entry points
  hard-code `is_streaming_mode=True` (`client.py:191`,
  `_internal/client.py:137`). The adapter is *not* forced to use an
  `AsyncIterable` prompt.
- **`start_async`'s "really started" has a real signal.** `connect()` performs an
  `initialize` control handshake and stores `_initialization_result`, which
  `get_server_info()` returns (`client.py:507-528`). `connect()` returning is the
  boundary, and the init payload is what a monitor wants.
- **The SDK writes the full transcript outside any workspace we grant** —
  `~/.claude/projects/<encoded-cwd>/*.jsonl`, redirectable by
  `CLAUDE_CONFIG_DIR`; subagent transcripts under `<session>/subagents/` are
  found by *scanning that directory* (`_internal/sessions.py:1290`). Levers:
  `CLAUDE_CODE_SKIP_PROMPT_HISTORY` in `env`, or a `SessionStore` adapter.
  Criterion 16 stays true of *our* record; the backend's record holds everything.
- **`ResultMessage` carries content criterion 16 forbids recording** — `result`
  (final text), `structured_output`, `permission_denials`. Only
  `api_error_status` is annotated "Safe to log (no message content)". Persisting
  the whole message violates the criterion; the adapter must project a subset.
- **Subagent output reaches our stream by default.** `forward_subagent_text` is
  `False`, but subagent `tool_use`/`tool_result` blocks are emitted anyway,
  carrying `parent_tool_use_id` (`types.py:2158`).
- **`interrupt()` is connection-wide** — no `session_id` parameter — so "each
  phase its own session on one client" is ruled out. This narrows the module-3
  carry-over rather than answering it.
- **The SDK prefers its bundled CLI**, `shutil.which("claude")` is only a
  fallback (`transport/subprocess_cli.py:247-256`), while
  `env_mgr/installers/claude.py` assumes a `claude` already on PATH and installs
  only *plugins*. **Two CLIs; by default the backend runs the one `env_mgr` never
  touched.** `cli_path` is the lever. Neither spec mentions this.
- **Cost attribution, §9's open question, is answered for the Claude side.** A
  "run" is one `query()` call; **the SDK provides no session-level total**
  (`cost-tracking.md:215`); per-step usage must be deduplicated by message id;
  `/clear` starts a new `session_id`; and a session crash
  (`error_during_execution`) **may zero every cost field** — a settled figure of
  zero rather than an absent one.

### From the backend-selection survey

- **"Available" and "preferred" can be one expression.** keyring's `viable` *is*
  "reading `priority` did not raise" (`backend.py:92-97`); the docstring
  instructs backends to raise `RuntimeError` naming the cause. No separate
  `is_available()` to drift out of sync with the ordering.
- **But that probe is expensive and side-effecting** — live D-Bus calls;
  libsecret's *opens a session* to answer "are you there". `priority` is not
  memoised and is read twice per candidate. Bill: keyring #162, *"Import of
  keyring takes 25s"*. **The fix was deferral, not caching** (NEWS #403, #404):
  *when an explicit override names a backend, do not probe the others at all.*
- **Observing the selection must be separable from making it, from day one.**
  matplotlib's `get_backend()` *resolved and committed* the choice (#12362), so a
  later `use()` became a no-op with a warning; #23298 is worse — resolution
  *destroyed the user's open figures*. Retrofit took 3.0 (2018) → 3.10 plus a
  provisional flag and a deprecation runway (PR #29039). **All three matplotlib
  bugs share one shape: selection is a side-effecting operation disguised as a
  read.**
- **An explicit request should disable the fallback chain, and matplotlib says so
  in a comment**: `# if the user has asked for a given backend, do not helpfully
  fallback` (`__init__.py:1283-1285`). Direct support for §3.3's "a CLI override
  pins the whole run".
- **Only one of five projects enumerates the candidates it tried** — virtualenv,
  and it gets it free from argparse because probing runs *before* the parser is
  built, so `choices=` already holds only available ones. **This corrects our
  expectation**: our "errors enumerate candidates" precedent (pytest, dbt) does
  **not** extend to backend selection. fsspec and matplotlib both have the list
  available and do not print it.
- **Three observed shapes for "none available", and raising at selection is not
  among them**: keyring returns a **null object** raising `NoKeyringError` at
  first *use* (and `fail.Keyring` has `priority = 0`, so the fallback sits inside
  the candidate set it is the fallback for); matplotlib **never fails**, falling
  back to `agg`. Ours would be a third shape.
- **Swallowing probe failures was a deliberate reversal with a known cost.**
  keyring #316: an uncaught exception from inside a probe made `import keyring`
  unusable. Broadening the catch fixed that and lost every rejection reason.
  **You get one or the other unless the probe returns a structured result** —
  which is exactly what virtualenv does with its `defaultdict(list)` keyed by
  rejection reason.
- **`default = next(iter(choices))` in virtualenv is our criterion 3, mechanism
  2, literally** (`creators.py:66-80`).
- **Nowhere in the survey is the ordered preference in a config file.** Our §3.3
  source 3 is a deliberate departure with no prior art. The evidence *for* it is
  keyring's `5.1 if "KDE" in $XDG_CURRENT_DESKTOP else 4.9` — a deployment fact
  smuggled into a library constant because there is no config channel — and the
  need to coordinate priority numbers across packages nobody co-releases.
  matplotlibrc and keyringrc each name **one** backend, not an order.
  **If we put the order in config, the "do these names exist" check is ours to
  write** — LiteLLM's `validate_fallbacks` does not do it.
- **`backend_entry` need not choose between an entry point and a dotted path.**
  fsspec, SQLAlchemy and matplotlib each use both. The dotted string buys one
  thing an entry point cannot: **a per-entry `err` message** — fsspec re-raises
  `ImportError(bit.get("err"))` so the user gets *"Install adlfs to access…"*
  rather than a traceback. Its cost: `_import_class` carries a literal
  `is_s3 = mod == "s3fs"` special case, and the docstring warns it "can import
  arbitrary modules".
- **matplotlib's entry-point validation list is a good checklist for our `key`**:
  may not start with `module://`, may not shadow a built-in, may not be
  duplicated with a different module; identical duplicates tolerated because they
  occur outside the project's control.
- **Mid-run fallback: LiteLLM alone does it, and it cost five machinery pieces** —
  depth bound, an attempted-targets set carried by reference (against fallback
  graphs that loop), a pin predicate, cooldown feedback, and per-failure-class
  chains — plus threading three parameters through a loosely-typed `kwargs` at
  four call sites because "a declared parameter would carry an annotation that no
  call site can actually be checked against". **A cross-cutting concern that
  defeated the type system in the only project that shipped it.** Its pinning
  rule is also a *correctness* constraint emerging from run state, not only a
  user preference. Every other surveyed project lets a mid-run failure be a
  failure.

### From the format-transform survey

- **Criterion 13 as written is not testable at the output level.** "What both
  support" is not a fact the converter knows; everyone hand-maintains a
  per-target table, and both reference implementations' tables fail:
  - **pandoc drops silently and `--fail-if-warnings` does not catch it** —
    `BlockNotRendered` is **INFO**, so the gate exits 0 with the content gone.
    Attribute-level loss (a link title) is not logged *at any level*: empty log,
    exit 0. Twenty years of use and still no severity meaning "this conversion
    lost data".
  - **kompose's 25-entry unsupported-key table has no production caller** — its
    only caller is its own unit test, and the exported sibling is invoked with an
    empty map. Eight declared-unsupported keys converted clean, exit 0. No
    `--strict` flag exists.
  - **A feature matrix not on the conversion path is decoration.** It must be
    what the converter *dispatches through*.
- **The only executable form of the criterion found anywhere** is rulesync's
  per-(target, feature) e2e fixtures: a native file **imports to the expected
  canonical event**. Nobody attempts a universal round-trip property.
- **Separate "unknown" from "unsupported"** — GHA Importer does; *unsupported* is
  "fundamentally not supported by the target", *unknown* is "not automatically
  converted". **Criterion 13's "what both support" is only the first; conflating
  them makes it untestable.**
- **Seven residue strategies observed**, each with a project: silent drop
  (pandoc); drop logged below the gate (pandoc); textual marker substitution
  (`[STRIKEOUT:…]`, `RST.hs:777`); warn-and-ignore (kompose); refuse by default
  plus in-band `x-s2o-warning` on opt-in (swagger2openapi); a four-state
  conversion report that **keeps the original beside the output** (GHA Importer);
  and prompt-layer emulation (rulesync `--simulate-*`). Plus **namespaced opaque
  passthrough**, which is rulesync's and is the one that answers the hub
  bottleneck.
- **§4.5's premise has a live counter-example.** rulesync — ~30 harnesses, 299
  releases — deliberately chose a **neutral hub**, not any vendor's format, with
  per-tool override keys as the escape hatch. It also shows why: **"Claude Code's
  format" is ambiguous.** What everyone copied is the *declarative*
  `.claude/settings.json`; the SDK's `ClaudeAgentOptions(hooks={...})` callbacks
  are a different model, and **nobody converts programmatic callbacks at all.**
  §4.5 must name which surface it means.
- **The hub bottleneck is real and measurable**: pandoc's `Underline` — a feature
  **both LaTeX and docx support** and the hub could not express — cost **~18
  months** and an AST breaking change (`pandoc-types#68`); the `<mark>`
  equivalent has been **open four years** (`pandoc#8220`).
- **Hooks are translated by generating code**, not by transforming data:
  OpenCode/Kilo get a generated `.js` plugin, Amp a `.ts` plugin, Pi a `.ts`
  extension, with generated guard code where the target's event is broader.
  Blocking semantics are the hard part — a non-zero exit becomes
  `{block: true, reason}` with the reason scavenged from stderr→stdout→exit code,
  sanitised of control/bidi/zero-width characters, capped at 2000 chars.
- **Hook mistranslation fails silently in both directions**, with two shipped
  defects in rulesync's own comments: a dropped matcher made a Grok hook "fire on
  everything instead"; a wrong event mapping meant a hook "never fired on shell
  commands". **Both produce valid output files.** This is the part of criterion
  13 with no known testable formulation.
- **A → hub → A is not identity, and nobody claims it is.** pandoc md→md is not
  identity but **is idempotent from generation two** — a formatter's contract.
  rulesync makes `--from X --to X` a **hard error**: "likely a mistake and may
  cause lossy round-trips". Its remedy for field loss is an explicit
  `importPassthrough()` allow-list. **The achievable contract is idempotence
  after one pass plus enumerated passthrough**, not identity.
- **The converter is independent in every project examined**, and the versioning
  cost is the dominant one: rulesync has **299 releases, seven majors in two
  weeks**. **The converter's cadence is the union of the spokes' cadences** —
  which is the argument for §4.4's "independent module": it cannot share a
  version with the agent spec.

### Gaps this research opened, none yet decided

- **A task with no agent is required by `closure` §2.2 and `demo` criterion 7,
  and is inexpressible in `task_graph`** — `Task.agent_spec: str` is not
  optional, dispatch instantiates unconditionally, and `TaskRunner.start`
  requires an `Agent`. Either the loader supplies a `kind: program` spec (costs
  nothing, but then criterion 15's "program executor" means a spec, not
  *no* agent) or `task_graph` changes (costs a frozen, 423-green component). **A
  spec finding spanning three modules.**
- **Hard dependency or extra**, given 376 MB installed and 1.3 s to import.
- **Which `claude` CLI the backend runs**, and therefore whether `env_mgr`'s
  plugin installs are visible at all.
- **Whether "Claude Code's format" in §4.5 means the declarative settings file or
  the SDK's callback registration.** The survey says the two are different models
  and only the first has any prior art.
- **What criterion 13's "what both support" is defined *by*.** Without a named
  artefact and a test that keeps it honest, it is unfalsifiable.
- **Whether each phase gets its own session** — narrowed (not one client with
  several `session_id`s, because `interrupt()` is connection-wide) but not
  answered.
- **How `AgentSpecRegistry` and `AgentMgr`'s spec table relate.** Two things hold
  "the agent spec table" — principle §1, one invariant one writer.
- **Whether the selection returns a structured result** — `(backend, source,
  rejected-with-reasons)` — or only the backend. Both surveys converge on the
  first; nothing in the spec asks for it.

## Carried into module 6 — closure

Not research; things module 6 must not rediscover or contradict.
`closure/docs/spec.md` rev. 7, §5 — 11 criteria.

- ~~**`closure` §2.2's "no agent spec at all" is inexpressible in `task_graph`,
  and module 5 assumed the loader supplies a `kind: program` spec instead**~~
  **Settled 2026-08-27, after module 8** — see "The no-agent question, closed"
  below. The spec changed: every task has an agent, and `kind` is what varies.
- **The closure check is a separate pass after all four registries are loaded**
  — module 1's finding, dbt's `ManifestLoader.load()` shape. `check_graph` was
  added to `build_registry` by `task_graph` rev. 11, and main design §7 now
  states it is *not* the closure pass. Two passes, sequenced.
- **The composition root is a shared surface.** `task_graph` rev. 11 was the
  first module design to put something *back* into `build_registry` (main design
  D6). `closure` and `env_mgr` are the likely next.
- **Placeholders pass every check a real value passes** (main design O9) — the
  Hugging Face `[More Information Needed]` measurement, 636 321 repositories.
  Whether a template may carry placeholder defaults is a spec question and
  `closure` is where templates are bound.
- **`ValidatorId` is named by main spec §4.6 and defined nowhere** (main design
  D5); every design keys validators by name.
- **A closure carries no version; each of its four members carries its own**
  (`closure` spec §1.2, rev. 3). `agent` design §3.1 gives `AgentSpec.version` a
  field and no reader for the same reason.

## Module 6 — closure. What its research settled

`closure/docs/spec.md` rev. 7, §5 — 11 criteria. Findings live in
`scratch/design/findings-closure-mine.md`; probes in
`scratch/design/probes-closure/`.

The shape of this module's research is different from every module before it.
`closure` is the last module whose subject is *already described by other
designs*: main design §6 gives the closure pass a home and a report format,
main design §2.3 and D2 put the task spec registry inside `closure/`, and
`task_graph` design §3.5 and §8.5 name two of its collaborators. So most of
what there was to find was **found in this repository, by reading the four
designs against each other** — and what that reading turned up is six
collisions, four of them consequential.

### Measured here, first-hand

- **`Permissions.covers()` cannot serve closure check 6.** `task_graph` design
  §3.5 says the type exists for this check and gives it
  `covers(hid: HandoffId, access) -> bool`, with `Grant.handoff: HandoffId |
  None`. `HandoffId` is a `uuid.UUID` subclass; check 6 runs at load, where a
  closure names handoff **kinds by name**. Measured: `HandoffId('trace')` →
  `ValueError: badly formed hexadecimal UUID string`, and the pydantic
  `_coerce` path raises identically, so a declared grant naming a kind cannot
  be loaded into the declared type at all. Three candidate shapes are on
  record; all three are `task_graph` design changes, so this is reported.
- **`Task.agent_spec` is a required `str`.** `Task()` → `missing`;
  `Task(agent_spec=None)` → `string_type`; `Task(agent_spec='')` → admitted and
  meaningless. So `closure` §2.2's "names no agent spec" had no spelling in the
  runtime model. **This measurement is what eventually decided the question** —
  in favour of the runtime model, not against it. See "The no-agent question,
  closed" below.
- **Criterion 11 costs nothing.** `additionalProperties: false` already yields
  `Additional properties are not allowed ('version' was unexpected)`, which
  names the key and is actionable. Adding an explicit `not: {required:
  [version]}` produces a *second*, worse message alongside it. So §1.2 is
  expressed as a schema difference — the closure schema omits `version`, the
  four member schemas declare it — and not as a check.
- **The closure pass catches placeholders in every referential field.** Main
  design O9 (a placeholder passes every check a real value passes) is carried
  here because "closure is where templates are bound". It is largely
  unrepresentable *here*: `agent`, `handoffs`, `validators`, `task.inputs` and
  `task.outputs` are all names, and `"TODO"` does not resolve. The residue is
  `name`, `description` and `task.goal`. O9 stays open, but it is worth much
  less in this module than elsewhere.

### Collisions between designs, which is what this module actually found

- **The closure pass is called from a module forbidden to import it, and it
  runs too early.** Main design §3.6 puts `check_closures(...)` inside
  `spec_loader/package.py::load_package`, while §2.3 says *"`spec_loader`
  imports nothing from this repository. It is the leaf, and it must stay one"*.
  Worse: §7's composition root calls `load_package` **once per package**, so
  with two packages the pass runs before the second package's specs exist —
  exactly what §6.1 forbids (*"Resolve-during-load cannot see the far side of a
  binding"*), and main spec §4.3 makes cross-package references a supported
  case. The natural fix moves the call beside `check_graph` at the composition
  root, which also resolves the import. A main-design change; reported.
- **Nothing owns "turn a closure into the root `Task`", and the only sentence
  that assigns it names the one component forbidden to do it.** Main design §7:
  *"the scheduler is what assembles it"*; closure criterion 8: *"The scheduler
  never reads a closure"*. The subgraph half **is** owned — `Task.unfold`
  (`task_graph` design §8.5) reads `closures` from a `Task` transition, which
  main design §7 explicitly permits. The root half is unowned. The only named
  non-test callers in the whole spec set are `demo`'s `show` verb and
  `--dry-run`.
- **A closure's phase validators are a reference kind nobody's index counts.**
  `ValidatorSpecRegistry.users_of(name)` is *"which handoff kinds bind this
  validator"* only (`validator` design §10.5). A validator a closure names as a
  **phase** validator — which `closure` §2.4 says the handoff specs *cannot*
  carry — is invisible to it, so `users_of` reports a validator two closures run
  as used by nothing. That is Airflow #58058's failure mode, which `validator`
  design §10.5 records as the thing to avoid, and `validator` O7 already asks
  who owns the enumeration of reference kinds. **Module 6 adds the third edge
  kind**, so it answers O7 or ships a known false negative.
- **`closure` criterion 10 cites main spec criterion 7**, which is *"The
  producing agent cannot reach the validation's context"*. It means criterion
  **11**, *"The six-step reference workflow is expressible"*. A cross-reference
  error in a frozen spec; reported, not edited. Related and not an error: the
  six closures have **no artefact in this repository** to be checked against —
  `demo` §1.3 puts the reference workflow out of scope and its own graph is
  three tasks — so how criterion 10 is tested is a real question for the design.
- **Two different types are both called `LoadReport`** —
  `spec_loader`'s `(admitted, problems)` (main design §3.6) and `handoff`'s
  `(admitted, without_validator)` (handoff design §8.5). Closure check 3 /
  criterion 6 needs the second. Both are in scope in this module's `check.py`.
- **`TaskSpec` and `Registries` are named and defined nowhere.** `task_graph`
  design §8.7 types `check_graph(specs: Mapping[str, TaskSpec])`, while main
  design §4.1 says *"a spec is a plain `dict` throughout"*. `Registries` appears
  three times in main design. `TaskSpec` is this module's to define, because the
  task spec registry lives in `closure/`. Same shape as `ValidatorId` (main
  design D5): a name one document uses and none defines.

### From the permission-coverage survey

`scratch/design/findings-closure-perms.md`. Dagster, Kubernetes RBAC, Android
lint read in depth; systemd shallow and negative. Three citations spot-checked
first-hand and exact.

- **The two-level naming trap has no prior art, in either direction.** No
  surveyed permission model references a runtime instance ID, and none carries
  one type with both meanings. Kubernetes' `RoleRef` has `{APIGroup, Kind,
  Name}` and **no UID field anywhere in the RBAC model**; Dagster's requirement
  carries `asset_key: str`, the declaration-time key stringified; lint compares
  bare permission-name strings. **The permission model speaks only declared
  names, and a resolution step maps name → instance at dispatch.** That decides
  M1 toward its third option and against the first two.
- **Criterion 5 is not an outlier, and the reason is precisely that we declare
  our artefacts.** Dagster's `ensure_satisfied` raises
  (`resource_requirement.py:64`, verified) and is called unconditionally at
  repository build. Where the check is absent, it is because no requirement set
  exists to compare against: systemd declares `ReadWritePaths=` and a unit never
  declares what it will touch, so the question is unaskable, not deferred.
  Android lint has to *infer* the requirement and therefore fails open at four
  separate points. **Every implementation that actually performs the check is
  blocking.** The corollary is sharp: any feature letting a task consume an
  artefact it did not declare converts us from Dagster's position into lint's.
- **Anchor the check at the close of composition, and expose the predicate
  twice.** Dagster's per-job check is conditional and the mandatory one runs at
  repository build, with the reason in the source: `# Late validate all jobs'
  resource requirements are satisfied, since they may not be applied until now`
  (verified at `repository_data_builder.py:460`). And `is_satisfied` sits
  directly beside `ensure_satisfied` in the same class — one query, one
  assertion, one code path. Independent confirmation of M2's conclusion, from a
  system that hit the ordering problem in production.
- **A covering relation must own a closed name grammar.** Kubernetes' check
  *wrongly rejects* a legal delegation (#122154) because `resourceNames` has no
  glob support while the CSR admission plugin had given `example.com/*` a
  meaning. `liggitt`: *"example.com/\* covering example.com/specific is a
  semantic specific to the CSR admission plugin, it is not part of the
  authorization API or RBAC"* — then `/remove-kind bug`. **They shipped the
  wrongness rather than open the grammar.** Our grant is `(path, read|write,
  ...)` and paths invite exactly this. Fail-closed is asserted by a named test,
  verified verbatim: `TestCoversEnumerationNotCoveringVerbStar` — an exhaustive
  enumeration on the grant side does **not** cover `*` on the requirement side.
  Where a requirement cannot be enumerated at all, Kubernetes demands the
  maximal grant rather than skipping the check (`aggregationRule` requires
  `*` on `*.*`).
- **Existence and coverage are separable, and Kubernetes separates them.** RBAC
  checks rule coverage statically and deliberately does *not* check that a
  `RoleBinding`'s target exists — dangling references are legal and resolved at
  evaluation. Relevant to whether closure checks 2/4/5 (existence) and check 6
  (coverage) should share a failure mode.
- **The actionable message names three things, not two.** Everyone clears "name
  both sides"; the differentiator is a computed repair drawn from what is in
  scope — Dagster appends *"or change the required key to one of the following
  keys which points to an IOManagerDefinition: [...]"*. Notable negative: a
  tracker search for the Kubernetes message found three issues and **none**
  complaining it was unreadable; all three dispute the verdict. Message
  legibility is not where the pain is in these systems.

### From the reverse-index survey

`scratch/design/findings-closure-rindex.md`. dbt-core (`1.10.latest`), Sphinx
and Cargo read in source; Bazel from documentation only, which the agent flags.

- **Eager-and-frozen is the cheap position; a mutable index is the expensive
  one.** Nobody surveyed maintains a reverse map incrementally. dbt rebuilds the
  whole O(V+E) map defensively at **six** call sites, each immediately before a
  consumer, with `# parent and child maps will be rebuilt by write_manifest` as
  the stated policy. Cargo inverts the graph destructively per invocation and
  throws it away. Sphinx is the one system whose index outlives the load, and
  the price is a `clear_doc` + `merge_domaindata` obligation on **every** owner
  including third-party extensions, with `StandardDomain.clear_doc` degenerating
  into five full scans and an unresolved `# XXX duplicates?` in the merge path.
  Our closed world removes the need. The constraint: **make post-load mutation
  impossible, not merely discouraged.**
- **"Not found" and "found, used by nothing" must be separated at the API
  boundary.** dbt gets it right in the data — pre-populated `[]` — and loses it
  at all six call sites, which guard with `if unique_id in child_map`, and in
  the user-facing message, which must hedge across three causes: *"The selection
  criterion '{spec_raw}' does not match any enabled nodes"* covers typo, unused,
  and disabled alike. Cargo keeps them apart by resolving the name against the
  catalogue **before** touching the graph, so the two cases never share a code
  path. That is the cheaper pattern and it is what `closures_using` should do.
- **A hand-maintained membership list for the index is a live bug source.**
  `build_parent_and_child_maps` chains seven collections by hand (verified) and
  `build_node_edges` silently drops any edge whose target is outside that set.
  dbt issue 14436 is that bug reaching users: `depends_on` correct, `child_map`
  incomplete, no error. This is the same failure `validator` design §10.5
  recorded from Airflow #58058, now with a second independent instance —
  which strengthens the case that **module 6 adding a third edge kind (M5) must
  derive membership rather than restate it.**
- **The reverse index answers "who", and nobody extends it to "is this safe".**
  dbt ships both halves — `check_for_model_deprecations` walks `child_map` and
  warns each consumer, and `same_contract` is a real structural compatibility
  diff with named breakage categories — and **never joins them**: warn-vs-error
  keys on whether the model declares a version, and `same_contract` does not
  take the manifest as a parameter, so it structurally cannot see consumers.
  `closure` spec §6's "change propagation" gap is therefore confirmed as a real
  gap, not something we are failing to copy. Two transferable details if it is
  ever closed: gate the expensive diff behind a cheap checksum, and return named
  breakage categories rather than a bool.
- **Scope the answer to the loaded world and say so.** Bazel's `rdeps` makes the
  caller name the universe, and the documentation's own worked example shows the
  failure being a confident **empty** answer rather than an error. We get the
  universe free — five registries, all loaded — but the docstring has to say
  that is the universe.
- **Do not let the reverse query grow into a graph API.** When Bazel's node
  identity gained a configuration dimension, `cquery` **removed** `allrdeps`
  rather than generalising it. If a closure could ever reference one handoff
  kind under more than one context, `closures_using(kind)` stops having a single
  well-defined answer, and the precedent says restrict the query rather than
  widen it. (The agent flags the causal link as doc-inferred, not verified.)

### Delivery note on the two surveys

Both agents went idle repeatedly without emitting anything and produced their
findings only on the fifth request. Nothing was lost, and the material is good —
three citations were spot-checked first-hand and matched exactly — but the
pattern is now twice in a row (module 5's two agents did the same), and it is
worth assuming rather than rediscovering: **a survey subagent's findings do not
arrive on their own.**

### What this module owns that the spec does not say out loud

Two registries, not one: `TaskSpecRegistry` **and** `ClosureRegistry`, both
registered by the composition root (main design §7). The task spec is nested
inside the closure document as the `task` key and carries no `name` of its own
in `closure` §2's table, so what it is registered *under* is an open design
question — and if it is the closure's name, the two registries share a key space
by accident rather than by decision.

## Carried into module 7 — env_mgr

Not research; things module 7 must not rediscover or contradict.
`env_mgr/docs/spec.md`, and `closure` design rev. 1 §6 and §13.

- **A permission grant names a handoff *kind*, never a `HandoffId`**
  (`closure` design D2). `task_graph` design §3.5 typed `Grant.handoff` as a
  runtime uuid and `HandoffId('trace')` raises, so the type cannot serve the
  load-time check it was built for. The three-system survey found **no**
  permission model anywhere that references a runtime instance id. **The
  resolution step is `env_mgr`'s**, at the moment it builds the zone — which is
  the same place it already resolves paths. This is the single biggest thing
  module 7 inherits.
- **`env_mgr` may not interpret a path in a way the covering function does
  not** (`closure` design §6.3). The alpha's covering grammar is **exact string
  equality, no wildcards**, because that is the only *total* grammar available.
  kubernetes#122154 is what a second interpreter costs: the static check
  wrongly rejects a legal case, the maintainers removed `kind/bug`, and it has
  been wrong ever since. If `env_mgr` gives `foo/*` or a trailing slash or a
  symlink a meaning, the load check becomes provably wrong in both directions.
- **The composition root now carries three passes**, in order: `load_package`
  per package, then `check_closures`, then `check_graph` (`closure` design
  §7.1, correcting main design §3.6). Main design D6 predicted `closure` and
  `env_mgr` would be the next modules to put something back into
  `build_registry`; the first half happened.
- ~~**"A task with no agent" is still unresolved after three modules**~~
  **Settled 2026-08-27** — see "The no-agent question, closed" below. Every task
  has an agent; `kind` is what varies; `Task.agent_spec: str` was right. What
  module 7 correctly inherited from this note is unchanged: `env_mgr` prepares an
  environment for a `kind: program` executor exactly as for any other.
- **Nothing owns turning a closure into the root `Task`** (`closure` D5). The
  only attribution in the design set names the scheduler, which criterion 8
  forbids. `demo`'s `show` and `--dry-run` are the two named callers.
- **`Permissions` is carried by `task_graph` and interpreted by `env_mgr`
  alone.** `task_graph` design §3.5 is explicit: no method there resolves a
  path, compares a prefix, or decides containment. `closure` reads the field for
  one check and stores none.

## Module 7 — env_mgr. What its research settled

The safety-critical module, and the first whose spec is written "against measured
behaviour, not intuition" (its §4). So the research was to **re-measure the
mechanisms the spec turns on** rather than survey opinions about them — and the
measurements found that several of the spec's own sections cannot hold together.

Evidence: `scratch/design/findings-envmgr-mine.md` (M1–M22),
`findings-envmgr-selftests.md` (S1–S5), probes in `scratch/design/probes-envmgr/`.
`probes-envmgr/landlock.py` is a ~110-line ctypes sandboxer written as a
**measuring instrument**, not a proposed implementation.

Machine: Linux 6.5.0-45-generic, CPython 3.13.13, **`bwrap` absent**,
**Landlock ABI 3**, LSMs `lockdown,capability,landlock,yama,apparmor`.

### Measured here, first-hand

- **The kernel already defeats all three `startswith` attacks, for free.** Sibling
  `zone-EVIL`, a symlink inside the zone pointing out, and `zone/../outside` are
  each denied with EACCES and **no userspace path check involved**. So §4.3's
  canonical containment is *not* the enforcement mechanism — it is policy
  construction, the hook's first gate, and diagnostics. Criterion 3 can be
  satisfied by unit-testing a comparison function that proves nothing about
  confinement, and the design must say which layer each test targets. M2.
- **Confinement is inherited and cannot be widened.** A `bash -c` child and a
  scripted bypass are both denied; a second ruleset granting `/` does not widen
  the first, because layers intersect. M2, and the basis for M20.
- **§4.5.1's default granted set cannot start a Python on this machine.** The
  interpreter is under `$HOME` (conda), which §4.5.1 deliberately excludes, so
  `subprocess` raises `PermissionError` on the *interpreter path* before the child
  exists. Every ordinary Python install — conda, pyenv, uv, venv — is under
  `$HOME`. M3.
- **`/dev` is absent from §4.5.1 entirely, and `/dev/null` must be *writable*.**
  Without it git dies immediately: `fatal: could not open '/dev/null' for reading
  and writing`. And **a Landlock rule whose target is a file EINVALs if given
  directory-only rights** — the mask must depend on what the target is. M5.
- **Landlock also hooks ptrace, and domain membership decides.** With `/proc`
  granted, a process started *before* confinement is denied
  (`/proc/<pid>/environ`, `cwd`, `root/…`), while a child spawned *after* is
  readable. Verified against the unconfined case so the denial is not misattributed.
  The supervisor's environment — which here holds API keys — is protected **by
  ordering**, not by the filesystem grant. M10.
- **A grant never widens access beyond DAC** (`/etc` granted, `/etc/shadow` still
  denied), which bounds what a generous default set can cost. M11.
- **Landlock stacks at most 16 layers**; the 16th `restrict` returns `E2BIG`,
  matching the kernel's own `LANDLOCK_MAX_NUM_LAYERS 16`. M20, S5.
- **`rsync` cannot express "make both sides identical", and cannot detect a
  conflict.** `--delete` destroys destination-only content, so "identical" has an
  unstated direction; with both sides edited, `-a` and `--checksum` silently
  discard one and `--update` guesses by mtime. **No flag reports the case.** M21.
- **Sub-commands are compatible with criterion 22.** All six shipped CLI call
  shapes parse identically under `add_subparsers`, and the invalid-stage path
  still exits 2. 65 tests stay green. M12.

### Six collisions inside the spec, which is what this module actually found

1. **§6.1 (worktree) and §4.5 (no writes outside the zone) cannot both hold.**
   Measured three ways: main repository ungranted → `fatal: not a git repository`;
   read-only → `git add` fails on `<main>/.git/worktrees/<name>/index.lock`;
   read-write → works. **The worktree's index lives in the main repository**, so
   staging is a write outside the zone. M6.
2. **And the working configuration reintroduces the CVE criterion 11 names.**
   With the main repository read-write, the agent writes
   `<main>/.git/hooks/pre-commit` and **it runs** (`git commit` rc=0, artefact
   present); it also rewrites another task's branch ref and deletes objects from
   the shared store. The zone boundary itself holds — every control passed — so
   the entire leak is through the shared repository that §6.1 requires. §4.4 cites
   CVE-2026-26268 (CVSS 9.9) as the reason criterion 11 exists. M17.
3. **A clone with alternates resolves both, and both of its costs have measured
   answers.** `git clone --shared` under a **read-only** main repository: status,
   log, `cat-file` from the shared store, `add`, `commit`, `checkout -b` all
   rc=0, while `<main>/.git/hooks` and `<main>/.git/config` are denied. Objects
   are not copied (176 KiB vs 468 KiB). M18.
   - *The hazard is real and total*: main deletes a branch and runs
     `gc --prune=now` — routine housekeeping — and the borrower is left at
     `fatal: bad object HEAD`, with `fsck` reporting invalid pointers and a
     missing blob. Not degraded; unreadable.
   - *Mitigation, measured*: `extensions.preciousObjects = true` in main makes
     git refuse — `fatal: cannot prune in a precious-objects repo` — and the
     borrower stays healthy. Escape hatch: `git repack -a` plus dropping
     `objects/info/alternates` makes the clone self-contained and `fsck` clean.
   - *And the lost property is recoverable*: the agent cannot push to a
     read-only main (`remote unpack failed`), but **main can fetch from the
     agent's clone**. The write happens on the main side, performed by the
     supervisor. §4.5 holds unmodified and §6.1's "visible to the operator"
     becomes one supervisor-side `git fetch`. M23.
4. **Canonicalisation and `closure`'s covering grammar are two interpreters, and
   they disagree on every case tried.** Trailing slash, `.` segment, symlink vs
   target, `..` traversal: exact equality says "different", realpath says "same" —
   four for four. `closure` design §6.3 forbids exactly this direction. Proposed
   fix, two levels: a **syntactic** canonicality requirement checkable at load with
   no filesystem, and a **realpath-equality** check at zone build, failing closed.
   M9.
5. **§5.1's layout has nowhere to put a validation's materials**, yet criterion 13
   says the property is "resolved entirely by §5.1's containment". A validation is
   not a subtask (`validator` design settles that), and the only place the layout
   has room is *inside* the producing task's subtree — which
   `task_graph` §3.2.2 makes reachable. The placement must be named. M22.
6. **The zone is a property of the attempt, not the task.** Grants resolve to
   `<root>/<hid>/v<N>/` and the versions live on `Execution`, not `Task`, so a
   retry rebuilds the granted set — while §4.5 reads as though the sandbox is
   built once and never again. M14.

### The kind→instance resolution, inherited from `closure` D2

`Handoff.type` already carries the kind name on the runtime object
(`task_graph/models.py:107`), so resolution is
`task.inputs → Handoff → .type == grant.kind → <store_root>/<hid>/v<N>/`, with no
manifest read and no store access at zone-build time. **The hole: `type` defaults
to `""` and nothing requires it to be a registered kind**, so an unset type
matches no grant and the agent gets an empty granted set rather than an error.
M13.

### From the Landlock kernel selftests, read first-hand

Not a subagent report — `base_test.c`, `fs_test.c`, `landlock_common.h`, read
directly.

- **The kernel's own suite fails rather than skips when the mechanism is absent**,
  and hard-asserts the exact ABI: `ASSERT_EQ(11, landlock_create_ruleset(NULL, 0,
  LANDLOCK_CREATE_RULESET_VERSION))`. Direct support for spec §10's "it does not
  skip". S1.
- **But it does skip — 15 times, all for filesystem support** (`overlayfs is not
  supported`), never for Landlock. That is the sharper rule worth adopting:
  **skip for environmental variation orthogonal to the property; never skip the
  property.** S2.
- **A denial is asserted by errno against a named path**, never by exit status:
  `ASSERT_EQ(EACCES, test_open(file1_s1d1, O_RDONLY))`, with the helper returning
  errno. This is exactly what M4 arrived at after my own instrument produced a
  false PASS from a `returncode != 0` check. S3.
- **`TEST_F_FORK()` is deprecated** — "should not be used for new tests", now an
  alias for `TEST_F`, because the harness forks per test. Isolation is the
  runner's job, not each test's; cross-process cases use separate helper binaries
  synchronised by a pipe. pytest does not fork, so this is ours to arrange. S4.

### From the sandbox-testing survey

`findings-envmgr-sbxtest.md`, F1–F16. Kernel selftests, `rust-landlock`,
`bubblewrap`, `openai/codex`. The kernel citations I re-read first-hand and they
matched exactly.

- **Nobody restricts *reads*.** Codex's entire filesystem policy is `/`
  **read-only**, `/dev/null` read-write, plus the caller's writable roots
  (`linux-sandbox/src/landlock.rs:137-163`) — so the conda/pyenv/uv problem M3
  found simply dissolves for them. And when full read access is unacceptable they
  **refuse rather than approximate**: *"Restricted read-only access is not
  supported by the legacy Linux Landlock filesystem backend."* Our criteria 12 and
  13 both require read restriction, and P2/P3 measured that it works — so this is
  a cost we take on deliberately, not a gap. M7's absent-versus-denied problem is
  the price. F12.
- **`// Linux would return EINVAL.`** — `rust-landlock/src/fs.rs:316`, in the
  `fstat`-and-mask code that is exactly M5's workaround. The canonical
  implementation confirms the measurement and the fix. F13.
- **The ecosystem's default helper is fail-open on a missing path.**
  `path_beneath_rules` does `Err(_) => None`: no rule, no error, so a typo in an
  allow-list silently evaporates. Codex uses it for `/`, `/dev/null`, and every
  writable root; bubblewrap instead exposes the choice as two flags, `--ro-bind`
  versus `--ro-bind-try`. Against our principle 3 this has to be a deliberate
  decision, not an inherited default. F14.
- **"Fail closed" is two rules at two tiers, everywhere.** Production:
  best-effort construction, hard error only if the ruleset ended up
  `NotEnforced`. Test: `HardRequirement`, assert `FullyEnforced`. rust-landlock
  says it under a heading called "Test strategy" and adds *"applications should
  only check that no error is returned"*. Our §4.2 states one rule for both tiers,
  which is stricter than anything surveyed. F15.
- **Nobody achieves "fail, never skip" by probing at run time — they pin the
  environment.** The kernel ships `config` with `CONFIG_SECURITY_LANDLOCK=y`;
  rust-landlock pins `LANDLOCK_CRATE_TEST_ABI` per CI runner and asserts the
  kernel matches, and boots UML kernels to cover the old-ABI cases. **The ABI
  belongs in the CI job definition, not in a runtime probe** — which is a better
  answer to M16 than an injectable probe alone. F7, F8.
- **The project closest to ours took the opposite decision, in the weakest form.**
  Codex skips with a bare early `return`, so the test reports green
  (`tests/suite/landlock.rs:202-234`), and its `expect_denied` helper is
  `assert_ne!(output.exit_code, 0)` — M4's anti-pattern, shipped. The pattern
  across all four projects: the two whose *job* is the mechanism use exact errno;
  the one that merely consumes it uses `!= 0`. F10, F11.
- **bubblewrap's env-var gate is the cleanest answer to the developer-versus-CI
  tension, and it leaks.** `[ -z "${BWRAP_MUST_WORK-}" ] && ! $RUN true` → skip;
  CI sets `BWRAP_MUST_WORK=1`. But the variable appears in exactly one file, and
  the Python half of their own suite skips unconditionally — passing green in the
  CI job that sets it. **If we adopt this shape the gate must live in one place
  every test path traverses.** F9.
- **Three harness shapes for the self-poisoning problem**, and the kernel deleted
  its per-test opt-in: `TEST_F_FORK` is now an alias for `TEST_F` because the
  harness forks everything. bubblewrap's is the cheapest analogue for pytest —
  a class decorator that re-runs the same test method inside a sandboxed child
  over a socketpair, marshalling the assertion text back. F1, F2, F6.
- **A live hazard on this kernel:** `landlock_restrict_self()` restricts only the
  *calling thread*; `all_threads()` arrives at ABI 8. At ABI 3 best-effort
  **silently leaves sibling threads unrestricted while the status still reports
  enforced**. F15.
- **Codex has migrated away from Landlock to a bundled bubblewrap**, which
  supports our chain order — and warns that `PR_SET_NO_NEW_PRIVS` (required for
  seccomp) breaks the setuid bwrap deployments many hosts use. F16.

### A method note, third module running

**A survey subagent's findings do not arrive on their own.** Two agents were
dispatched with explicit "your final message text is the deliverable, do not write
a file" instructions and a stated budget. Neither answered the first seven
messages over about an hour. `sbxtest` then delivered in full, twice, unprompted
by anything new — sixteen findings with first-hand citations, and good ones.
`wksync` produced three git experiments on disk and sent only idle
notifications.

Two lessons, both cheap: **keep pinging, the work usually exists**; and **read
what they leave on disk** — the kernel selftests and the alternates experiments
were both sitting in their scratch directories, and reading them myself answered
four questions without waiting. Same non-delivery pattern as modules 5 and 6.

### Gaps this research opened, none yet decided

- **Criterion 9 has no machine on which all three branches run.** Rung 1 is
  untestable where `bwrap` is absent; rung 3 is untestable where Landlock works.
  Mechanism selection has to be injectable for the branches to be testable at all.
  M16.
- **Criterion 17 is not testable as written** — "nothing depends on its contents
  having survived" is a property of all future code, not an observable of a run.
- **Criterion 21 has no artefact.** It needs a knowledge handoff carrying cluster
  conventions, and spec §11 concedes the system-level tasks that would produce one
  are unspecified.
- **Rung 1 and rung 2 are not the same kind of confinement.** bubblewrap isolates
  network and PID; Landlock at ABI ≤ 3 isolates neither, and cannot touch the
  network before ABI 4. The spec presents the chain as an ordered degradation of
  preference, not of properties. M1.
- **An allow-list makes an ungranted file look *broken*, not absent.** Same tool,
  same option: a nonexistent config → rc=0; an existing-but-ungranted one → rc=128
  fatal. Every path a tool merely *probes* becomes a new hard failure, fixed
  per-tool rather than in the sandbox. M7.
- **Whether task executors nest as processes decides whether task depth is capped
  at 16.** If the supervisor spawns each executor, the limit never binds. Nothing
  says which. M20.

## Carried into module 8 — demo

Not research; things the last module must not rediscover or contradict.
`cli/docs/spec.md` rev. 5, §6 — 16 criteria. The demo composes all seven other
modules, so almost everything they deferred lands here.

- **Nothing owns turning a closure into the root `Task`** (`closure` D5, and
  carried unchanged into module 7). The only attribution in the design set names
  the scheduler, which `closure` criterion 8 forbids. **`demo`'s `show` and
  `--dry-run` are the two named callers**, so this module either builds it or
  says who does.
- **"A task with no agent" is unresolved after four modules** — `agent` D1,
  `closure` D1, carried into 7. `demo` criterion 7 is the requirement that
  needed it, so the chain closes here or nowhere.
- **The workspace is a `git clone --shared`, not a worktree** (`env_mgr` D1),
  and `extensions.preciousObjects` on the main repository is a precondition
  `prepare()` enforces. The demo runs against *this* repository, so it is the
  first caller that must satisfy that precondition on a reviewer's checkout.
- **The zone is per attempt, not per task** (`env_mgr` D6), and a validation's
  materials are a **sibling** of the producing task's zone (D5). Criterion 13's
  "running twice without hand-editing" is therefore a statement about zone
  naming and cleanup, not about the graph.
- **[STALE — `_jsonnet` is no longer a dependency at all]** **Three runtime dependencies are undeclared** — `_jsonnet`, `jsonschema`,
  `jsonpath-ng` (main design O1). The demo is the first artefact whose user is a
  reviewer with a fresh checkout.
- **The machine-readable output is an interface, because criterion 14 makes it
  the assertion surface.** Nothing before this module has had to own one.
- **Isolation criteria 2–14 are already CI-enforced in `tests/env_mgr`**
  (`demo` spec §5). The demo adds *showing* the block to a person; it does not
  own the property.

## Module 8 — demo. What its research settled

The last module, and the only one that composes all seven others. Every part had
been measured by the module that owns it and **no two had been measured
together**, so the research was composition measurement — starting with the
pairing nobody had run at all: a real `claude-agent-sdk` backend *inside* a
Landlock domain.

Evidence: `scratch/design/findings-demo-mine.md` (M1–M15),
`findings-demo-survey.md` (S1–S6), probes in `scratch/design/probes-demo/`.

Machine: Linux 6.5.0-45-generic, CPython 3.13.13, **`bwrap` absent**,
**Landlock ABI 3**, `claude` 2.1.246.

### Measured here, first-hand

- **The confined agent works, and the demo's headline scene is real.** In one
  Landlock domain with cwd = the zone: a real model call answers in 8.4 s; a
  write inside the zone succeeds; a **scripted** write outside it is denied,
  nothing is written, and the agent reports the denial accurately and explains
  that the path is outside its working directory. Criterion 8's three clauses
  need no invention — two of them are already true. M4.
- **The backend is a Bun binary and aborts in 3 ms without `/dev/urandom`**,
  claiming *"This indicates a bug in Bun, not your code"* and printing a
  crash-report URL for the wrong project. Granting the whole of `$HOME`
  read-write does not help; one character device does. M1.
- **`/etc/resolv.conf` is a symlink into `/run`, and Landlock grants resolved
  paths**, so granting `/etc` does not give you DNS. Without
  `/run/systemd/resolve/stub-resolv.conf` every model call hangs ~184 s and
  returns `Request timed out`; with it, 5.5 s and the right answer. Under the
  same domain `getent` fails in 0.0 s with rc=2. M2.
- **`CLAUDE_CONFIG_DIR` inside the zone removes the `$HOME` grant entirely.**
  With `~/.claude` granted, the demo agent read the operator's personal
  `CLAUDE.md` and answered in Chinese — a demo whose transcript changes with the
  reviewer's dotfiles. M5.
- **Criterion 6 is already satisfied by the backend**: no credentials → rc=1,
  `Not logged in · Please run /login`, 0.6 s, identical confined and
  unconfined — **on stdout**. A demo that prints only stderr loses the message
  the criterion is about. M6.
- **A fresh process resumes from all twelve interruption points**, no exception
  and no unreadable record; the interrupted attempt is re-run as attempt 1 with
  attempt 0 recorded `SUSPENDED`. Stronger than `test_recovery.py`, which
  restarts with fresh managers over the *same live store object*. M8, M10.
- **Criterion 1's budget is comfortable and three quarters of it is `git
  clone`**: 2.97 s of 60 s, of which 2.25 s is four workspaces at ~0.56 s and
  12 MB each. M14.

### What the demo cannot do as written

- ~~**Criterion 7's "no agent at all" is not expressible.**~~ Reported to the
  user on 2026-08-27 and **settled the same day** — see "The no-agent question,
  closed" below. The measurement stands: `Task.agent_spec: str` is required,
  `Execution.agent_id: AgentId` is required, and `scheduler._dispatch_pass`
  calls `instantiate` unconditionally before `push_execution`. M12.
- **`depends_on` must be filled by whoever builds the `Task`.** It is
  `list[TaskId]`, runtime ids, so no spec can carry it, and omitting it makes
  `scheduler._warn_depends_on` log on every run. The reference example of the
  system would ship a warning. M11.
- **A task record is written before the handoff it names** (write 1 vs write 2),
  so a kill between them leaves a dangling reference. Nothing crashed in twelve
  tries, but only because the consumer is written *after* the handoff — an
  accident of this graph's write order, not a property. M9.
- **`examples/` is not installed, and a wheel drops the specs.** A console
  script pointing into an unpackaged directory installs fine and dies with
  `ModuleNotFoundError` when run; with `examples*` packaged, the editable
  install works and the wheel still ships `.py` only. Criterion 1 says
  `pip install -e`, so the editable row is the supported one. M13.

### From the prior art, read first-hand with `gh`

- **Airflow tests every shipped example, and every test is load-time.**
  `test_should_be_importable`, `test_should_not_do_database_queries`,
  `test_should_not_run_hook_connections` — parsing only, never execution. That
  dissolves the tension between `demo` §1 ("the first thing to break when one of
  them drifts") and §5 ("CI does not run it"): **CI loads the example on every
  commit; a human runs it.** Our load-time half already has a name — `--dry-run`,
  which needs no credentials, no sandbox and no model. S1.
- **The price is visible in the same file**: two hand-maintained exemption
  tuples and a per-file timeout table. With one example the list is empty; the
  moment there are two, the list is the thing to watch. S2.
- **The counter-example is dbt's `jaffle_shop`** — the most-copied example in
  its ecosystem, 544 stars, **archived, and `.github/workflows` is a 404**. S3.
- **Terraform owns its machine-readable stream as an interface**: a closed
  enumeration of message types, operations as
  start/progress/complete/errored quadruples, and `JSON_UI_VERSION = "1.3"` with
  a comment obliging a bump on any change. Criterion 14 makes ours an interface
  too. S4.
- **One event stream rendered twice, not two writers.** Terraform's JSON view is
  a selectable `View` and each call carries the human sentence *and* the typed
  fields. §4.2's "alongside" should be read that way; a demo whose narration and
  whose JSON disagree fails at the one job it has. S5.
- **pytest's `xfail` is the model for a failure that is the expected outcome.**
  Verified first-hand: `xfailed` is green, `xpassed` is its own category, and
  `xfail(strict=True)` that passes is `FAILED`. The failing validator and the
  blocked write are *strict* expected failures — a demo that reports "all good"
  when its sandbox stopped working is the worst outcome available to it. S6.

### A method note, and the end of the survey pattern

**No survey subagent this module.** The harness forbids the Agent tool unless
the user asks for it, and modules 5, 6 and 7 each spent roughly an hour on
agents that answered none of the first seven messages. Reading the four
upstream sources directly with `gh` took under twenty minutes and every citation
above re-resolves. Recorded because three modules of evidence say the delegation
was not paying for itself.

### Gaps this research opened, none yet decided

- **The demo needs `extensions.preciousObjects` set on the reviewer's own
  checkout** (`env_mgr` §7.2). That is a mutation of the repository a reviewer
  ran `git clone` on, before anything has been demonstrated to them.
- **Nothing enumerates what a *second* backend would need granted.** M1 and M2
  were found by running one binary; a different agent harness has a different
  set, discovered the same way — by it breaking.
- **`JsonFileStoreMgr` has no cross-record transaction** and the store's own
  docstring says sqlite is the upgrade path. M9's dangling reference is
  unreachable today by luck.
- **Criterion 12 costs a model call.** Resume re-runs the interrupted attempt,
  so where the demo interrupts is a demo-design choice with a price. M10.

## The no-agent question, closed

**2026-08-27, by the user, after module 8 reported it.** The one question four
modules deferred, and the only spec change this design stage produced.

> A task must have an agent. What varies is that the agent need not be an AI —
> it may be a human or a program.

`docs/spec.md` §4.8 rev. 9, `closure/docs/spec.md` §2.2 rev. 8,
`cli/docs/spec.md` criterion 7 rev. 6. Three designs follow — `agent` rev. 3,
`closure` rev. 2, `demo` rev. 2 — and three deviation entries retire: `agent`
D1, `closure` D1, `demo` D1.

**Nothing is implemented differently because of it.** `agent` design §9's
`ProgramExecutor` was already the thing a task without an AI runs; `closure`
still synthesises nothing; the demo's program node was always going to declare a
`kind: program` spec. What changed is that it stopped being a workaround.

Worth recording as a method note, because the shape recurs:

- **The measurement decided it, and it decided it the other way round.**
  `Task(agent_spec=None)` raising `string_type` was recorded in module 6 as
  evidence that the *runtime model* was missing a spelling. It turned out to be
  evidence that the runtime model was right and the wording was wrong.
- **Four modules deferring the same question was the signal.** Each one was
  correct to defer — none of them could change three specs — but "reported, not
  resolved" appearing in `agent` D1, `closure` D1, `env_mgr`'s carried notes and
  `demo` D1 is what a question that needs a decision, not more analysis, looks
  like from inside a design stage.
- **The rule "a design document does not amend a spec" held, and paid.** Had any
  of the four modules quietly made the spec fit its own module, the collision
  would have surfaced as a bug in the demo instead of as a decision.

## Stage three — the consistency pass, and the interface contract

**2026-08-27.** The eight designs are written. This stage read all seventeen
documents *in seam order rather than document order*, fixed what disagreed, and
produced the one artefact no single module could produce: a contract for what
crosses a module boundary, so eight people can implement eight packages and
integrate afterwards.

Findings in `scratch/design/findings-consistency.md`. Deliverable:
[`interfaces.md`](interfaces.md), plus six importable `protocols.py` / `.pyi`
pairs and 22 tests over them.

### What the pass found

**15 contradictions · 6 unowned jobs · 7 cross-module types nobody defined ·
4 dead cross-references · 1 measurement.**

The largest, and the reason the file exists: **the composition root was written
in four documents and no version was complete.** `agent` design §7.1 resolved
`env_mgr` and the validator's phase runner by name; **nothing registered
either**. Five of eight modules could not be wired as written.

Two were checkable against shipped code and both were wrong:

- **`task.cancel_reason = reason`** in `task_graph` §8.6's cascade. There is no
  such field, `Model` is `extra="forbid"` with `validate_assignment=True`, and
  the assignment raises. **The cascade died on its first entry.**
- **`resume_system` rebuilt the pools as plain `set()`** while §8.1.1 makes them
  `OrderedIdSet`. Promotion order — the thing `DepthFirstPolicy` is built on —
  would be destroyed after every restart, silently, on the one path no test
  covers.

And one collision that would have type-checked because neither side was
annotated: **`env_mgr.Access` and `task_graph.Access`**, mixed in one `Policy` by
`prepare()`, with `READ_WRITE` a member of only one of them. Renamed `Mode`,
because the two are genuinely different — what an author *declared* versus what
the *kernel* gets.

### The dependency measurement, which corrected three designs

Nine runtime dependencies are installed and **none is declared**. The suite is
green by accident. Three records were wrong in the same direction — a design
recorded the state of the world and the world had moved:

- **`python-jsonpath` is the one that is NOT installed**, and it is the library
  `handoff` §8.4 chose after measuring six. Both libraries it *rejected* are
  present, so a test written today would pass using a rejected one.
- **`rfc8785` IS installed**; `handoff` O2 said it was not.
- **[STALE — installed, and imported by nothing; `tests/interfaces/test_import_rules.py` now forbids it]** **`rjsonnet` IS installed**; main design O2 flagged `_jsonnet`'s missing
  aarch64 wheel without knowing the fallback was already there.

No type checker is installed either — which now matters, because the six
`protocols.py` are checkable and nothing checks them.

### Two spec changes, both decided by the user

**jsonpath → RFC 6901 JSON Pointer**, `handoff` spec §5.1 rev. 5 and `validator`
spec §4.1 rev. 7. `handoff` design D1 had reported it and correctly declined to
edit; the argument that carried it is a property of the standard, not a
preference — RFC 9535 §2.5.1.2 forbids a valid JSONPath query from erroring, so
**no** implementation can distinguish a wrong path from an absent value. `handoff`
D1 retires.

### Two questions reported and not answered, and what was done anyway

Both were put to the user and both are still open. Neither blocks
implementation, because in each case the *design*-level half was closed:

- **Who fills `Handoff.type`.** `Grant.kind` is a kind name and `env_mgr`
  resolves it by matching `type`, which nothing sets. Three routes, one of them a
  spec change. **What is done:** `env_mgr.resolve` now **raises
  `UnresolvedGrant`** instead of returning an empty granted set, so whichever
  route is taken, forgetting it is loud. That is `env_mgr` M7 — *an ungranted
  file looks broken, not absent* — caught one level up, where we control it.
- **`ValidatorId`.** Main spec §4.6 asks for a fourth typed id; three designs key
  validators by name, consistently and with a reason. Nothing is blocked: if the
  id arrives it is an addition, because a name is what the verdict record carries
  either way.

### The method note worth keeping

**Almost nothing needed research.** One measurement and one code check; the other
twenty-odd findings were two documents that each stated their reasoning and
disagreed. That is the design stage working — the disagreements were *visible*
because every module wrote its reasons down, and every one of them would have
cost a single message to resolve at the time.

Two shapes recurred and both are now enforced by a test rather than by care:

- **A seam has two sides and only one is in front of you.** `binds_to` versus
  `inputs`, two `LoadReport`s, two `Access`es, two `Verdict`s. Each was one
  module naming the other module's field.
- **A document that describes a shared surface will describe its own half.** The
  composition root, four times. The fix is not more care; it is one normative
  listing that the four defer to.

### Revisions this stage produced

| Document | Rev. |
|---|---|
| `handoff/docs/spec.md` | 4 → **5** |
| `validator/docs/spec.md` | 6 → **7** |
| `docs/design.md` | 1 → **2** |
| `handoff/docs/design.md` | 1 → **2** |
| `validator/docs/design.md` | 1 → **2** |
| `task_graph/docs/design.md` | 11 → **12** |
| `agent/docs/design.md` | 3 → **4** |
| `closure/docs/design.md` | 2 → **3** |
| `env_mgr/docs/design.md` | 1 → **2** |

Deviations retired: `handoff` D1, `closure` D2, `closure` D3 — each adopted by
the document it was reported against. Two added to main design, D7 and D8.

`pytest agent_sys` is **445 green** — 423 unchanged, 22 new over the contract.

## The user-interface brief, and four decisions it forced

**2026-08-27.** `image.how.to.usedb.yuser.md` (repo root) sets out how a user is
meant to reach this system, layer by layer. Checked against all seventeen
documents. **The direction, the module split and the principles hold** — a task
is `<handoffs, agent>`, subgraph-or-leaf, handoff is README + define + the real
files, the agent's five verbs, sync wrapping async, a validator has no subtasks.

**One part already exists**: the env template layer (§1.2 of the brief). `env_mgr`
ships `apt`, `uv`, `bin`, `embed`, `oneline` and `claude` installers, and
`installers/claude.py` already handles the `superpowers` plugin.

**Four things the spec set does not have.** `grep` over every `.md` and `.py`:
`mainloop`, `entry.sh`, `monitor_spec`, `set_task`, `make_async` — **all zero
hits**. None of them is in `aiopt.whole.system.md` either, so the spec stage did
not drop them; this brief is one layer more concrete than the original.

Decided by the user, recorded here before propagation:

### D-A. A task has a `body`, and it is not `goal`

| | |
|---|---|
| `goal` | One sentence, **≤100 characters** (the limit is globally configurable). What this task is *for*, for a human |
| `body` | **What the task actually is** — what to execute and how |

**`readme.md` is always required.** A programmatic task additionally carries an
**`entry.sh`**, which is its exact execution entry point.

The user's reason is the whole argument: *without a body, how does the user tell
the system what the task's semantics are, what to run, and how to run it?*

This subsumes the `goal`-has-no-runtime-home finding two sections above, and
enlarges it: the gap was never "`goal` was dropped in translation". **There is no
`body` concept at all.** Two independent routes reached the same place — a
spec-key-to-runtime trace, and this brief — which is what makes it the real one.

### D-B. A validator is structurally a task. The Python callable is out

Its checking logic is an **`entry.sh` or a `readme.md`, plus its own
`materials`** — the same body mechanism D-A gives a task. `validator` spec §3
already said *"A validator is a special kind of task"*; the design did not follow
it there.

The user's reason, and it is the one that decides it: **a validator an agent is
responsible for would need a wrapper around a callable anyway.** For code-shaped
checks the system ships a pytest harness, and the validator puts its test code,
its assembly command and its run command in `entry.sh`.

**Cost, stated plainly.** `validator` design §3.2's `logic: LogicRef`, §10.2's
two-registry join, and §10.6's `inspect.signature` argument check are all built on
the callable model and all have to be redone. What survives is everything about
*verdicts* — strengths, dimensions, the fold, the empty-phase rule, the
separation check.

### D-C. Two mainloops, with different jobs

**They are two things, not one.**

| | Job |
|---|---|
| **The agent's** | An agent is a live, stateful thing. *Without its own mainloop, how does anyone interact with it?* This is what drives `start()` |
| **The monitor's** | Handling the **task's** exceptions — an agent that stops behaving, a graph node that breaks |

So the monitor is **alpha scope**, not `ROADMAP.md` §2. `monitor_spec` joins the
bottom-level task define, the default monitor loop is resolved by name from the
component registry, and a monitor has `set_task`.

**One ROADMAP entry to add**: an agent may later attach its mainloop to a global
thread that round-robins over every attached agent.

### Propagated, 2026-08-27

All four are in the documents. Specs first, then designs — the order the process
requires, since a design follows a spec and not the reverse.

| Document | Rev. | What moved |
|---|---|---|
| `closure/docs/spec.md` | 8 → **9** | §2.5, §2.6 — `body`, `goal` ≤100, `materials`, `repos`, `monitor` |
| `task_graph/docs/spec.md` | 12 → **13** | §3.2.5–§3.2.6 — `closure`, `kinds`, `monitor_spec`; §3.5 — the monitor |
| `validator/docs/spec.md` | 7 → **8** | §6.1 — the body; the callable withdrawn |
| `agent/docs/spec.md` | 4 → **5** | §4.3 — `mainloop()`; every sync verb is sugar |
| `docs/ROADMAP.md` | — | §2 keeps the analysing dispatcher; §7.1 is new |
| `validator/docs/design.md` | 2 → **3** | §3.8, §10.2, §10.6 rewritten; D6 records the cost |
| `agent/docs/design.md` | 4 → **5** | §5.1, §5.1.1; the program executor gets a loop |
| `task_graph/docs/design.md` | 12 → **13** | §3.6–§3.8, §8.9; D23 retires |
| `closure/docs/design.md` | 3 → **4** | §3.6 — three accessors and check 7 |
| `env_mgr/docs/design.md` | 2 → **3** | §7.1.1 — per-task dependency repositories |
| `cli/docs/design.md` | 2 → **3** | §3, §4.2.1 — three bodies, and what they show |
| `docs/interfaces.md` | 1 → **2** | `monitor:<name>`; `mainloop`; `Body`; §5.1 closes |

**Two assumptions were stated rather than asked**, because the user's answer had
not arrived and both are cheap to reverse:

- **A non-leaf needs a `readme.md` too.** The exclusion is `entry.sh`-versus-
  subgraph, not body-versus-subgraph. The user's own argument decides it — without
  a body there is no channel for the semantics — and a non-leaf has semantics.
- **The monitor runs on its own thread.** Not a guess: `task_graph` design §9's
  table already contemplates a second thread calling in, and ROADMAP §2 already
  requires every monitor action to be a transition it *calls*. Transitions go
  through `_move`, under the scheduler's `RLock`.

**`Handoff.type` closed as a side effect.** It had been reported and unanswered;
`Task.kinds` is what the body decision made obvious, because once a task spec is
the thing the runtime resolves, the kinds are already there to carry. The
`UnresolvedGrant` raise stays anyway — being loud costs one raise.

**One thing is strictly weaker than before, and D6 says so rather than burying
it**: withdrawing the Python callable withdraws pandera's `inspect.signature`
argument check with it. A shell script has no signature. Three lesser things
stand in its place and they do not add up to it.

### D-D. `main repo` is global; the dependency repo list is per task

`main_repo` stays in `env_mgr.Context`, one per run. **Each task declares which
dependency repositories it needs** — `sglang`, `mooncake`, `aiter`. Nothing
carries that today.

## Module 9 — monitor. The task, and where its material already is

**Started 2026-08-27.** The ninth module, and the only one whose **spec does not
exist yet** — the other eight were specified in stage one. So the order is
different: **write the spec first**, then the standard four steps — deep research
→ update this file → plan and get it agreed → write the design, then stop and
report.

### Why it is a module and not a protocol

The monitor was in `ROADMAP.md` §2 until 2026-08-27, when the user's interface
brief moved it into the alpha. The propagation added `Task.monitor_spec`,
`task_graph` spec §3.5, and a `monitor:<name>` row in the composition root — and
then the spec-key-to-runtime trace found the consequence:

```
interfaces.md §2 registers monitor:<name>, typed Monitor
grep -rn '\bMonitor\b' *.py *.pyi   ->  zero hits
```

**`Monitor` exists nowhere**, and four designs mention a monitor while none
designs one. That is the same shape as the composition root's two missing
registrations, created by this stage rather than found by it. It is not closable
by adding a protocol: a monitor has a mainloop, two forms, a `set_task`, and an
exception-reporting story.

### What is already decided, and must not be rediscovered

| Source | What it fixes |
|---|---|
| `image.how.to.usedb.yuser.md` §2.4 | The user's own statement: two kinds (with an agent, without), both with a mainloop; `set_task`; exception reporting; the mainloop polls agent status |
| `docs/ROADMAP.md` §2 | Every task has a monitor; per-task **or** global with round-robin, holding only that task's permission scope while it works; the alpha is a **simple pusher** — a status check plus *"continue, do it until finished"*; the analysing dispatcher's 11-action set stays roadmap |
| `task_graph` spec §3.5 | Its job is the **task's** exceptions, not the task's work. Own mainloop, because a monitor that could only be called cannot notice a *stall* |
| `task_graph` design §8.9 | The boundary from the other side: `task_graph` owns the **verbs** a monitor calls and the lock that makes them safe from another thread, and nothing else |
| `task_graph` spec §3.2 rev. 13 | `Task.monitor_spec` — a name resolved from the component registry, `None` takes the default |
| `agent` spec §4.3 rev. 5 | **The agent's mainloop is a different loop.** Conflating them gives the watched and the watcher one heartbeat |

**The authority rule is the load-bearing one and it is settled**: every monitor
action is a task transition it *calls*, never a status it *assigns*. Transitions
go through `_move`, under the scheduler's `RLock`. `task_graph` spec §2 principle
4 stands unamended because of it, and that is what makes a second loop cheap
rather than dangerous.

**A standing assumption, stated not asked**: the monitor runs on its own thread.
`task_graph` design §9's table already contemplates a second thread calling in
and blocking on the lock.

### What the spec and the research settled

**`monitor/docs/spec.md` is at rev. 3.** The five questions below were the targets;
all five are answered, four of them by measurement. The research ran as four
independent threads, all filed in `scratch/design/findings-monitor-*.md` with
probes kept.

| Question | Answer |
|---|---|
| Where §3.5 lives | The mechanism moved to `monitor/docs/spec.md`. `task_graph` keeps the boundary and `Task.monitor_spec` — **not yet trimmed**, see the propagation list |
| What "stalled" means | **The user redefined it and made it cheap: non-delivery, not unresponsiveness.** The agent returned a terminal result and a declared output handoff is not there. The **runner** reports it — no clock, no threshold, no heartbeat |
| The permission scope | **The alpha needs no environment reach at all.** Every action is an in-process call on an object the supervisor already holds. A monitor must not enter a sandbox: it cannot (a multithreaded process cannot join a user namespace, and the supervisor is multithreaded by construction), and if it could, entry arrives with full `CapEff` and none of the target's Landlock domain — an escalation, not a scoping |
| Where the exception is recorded | **The carrier was never open** — a persisted value through `StoreMgr`, not a log line, a rule this repo had already stated three times. Only the vocabulary was open: OTel names + OTP report structure + Sentry fingerprints, at **zero new dependencies** |
| Whether an AI monitor is a task | **No** — the user decided it. The consequence is stated rather than hidden: nothing monitors the monitor |

**Four findings that changed something outside this module**, each reported and
not edited, because the spec set is agreed:

1. **`agent` spec §5.1's `instruct` row maps to the one form that cannot be
   pushed.** `query(AsyncIterable[dict])` closes stdin at the first turn
   boundary; measured `CLIConnectionError`. Correct form is `query(str)` on a
   client connected with a string or `None`.
2. **`agent` design §7 is missing a constraint**: the backend client's lifetime
   must be the *agent's*, not the *turn's*. The natural `async with` spelling
   destroys the subprocess before the runner checks handoffs, degrading every
   push to a lossy resume.
3. **`env_mgr`**: man `landlock_restrict_self(2)` states a layer cap of 64 where
   design §8.4 measured 16 on this kernel. `env_mgr`'s to reconcile; non-binding
   for the monitor either way.
4. **The tree-shaped subgraph view and `env_mgr` criterion 14 are one
   measurement** — a hierarchy grant on the parent zone root reaches every
   descendant, including subtrees created after the ruleset was built, while
   denying the sibling.

**Three things this module's own documents got wrong and had to correct**, worth
keeping because they are the same shape each time:

- Rev. 1's §4.1 described two detectable states; `put` refuses malformed content
  before creating anything, so **only one of them exists** and `exists(hid)` is
  the whole check.
- Rev. 1's §8 claimed the recording shape was free to choose. **The carrier was
  already fixed**; only the vocabulary was open.
- **§9's requirements table — the section whose entire purpose is to catch
  "X consumes Y and X cannot receive Y" — was itself missing the store row.**
  §8 required a record; §9 listed eight routes and no place to put one. Found by
  the research, not by the table. A checklist does not check itself.

### The questions as originally recorded

Kept for the record — this is what the targets looked like before the research:

- **Where does §3.5 live in the end?** It is currently inside `task_graph`'s
  spec, written before this module existed. Most likely it moves into
  `monitor/docs/spec.md` and `task_graph` keeps a pointer plus §8.9's boundary.
- **What does "stalled" mean, mechanically?** The pusher's whole job is detecting
  it, and nothing yet defines it. An agent's `status` does not change while it is
  stuck, which is precisely the case.
- **How does a global monitor hold "only that task's permission scope"** when it
  is one loop over many tasks? `env_mgr`'s zones are per attempt.
- **Where is the exception recorded?** The user asked for an industry-standard
  shape. Nothing in the system records an exception today.
- **Does a monitor with an AI in it need an agent spec, a zone and a lease?** If
  it does, it is a task; if it is a task, who monitors it.

### The layering survey — 2026-08-28, and it reopened the module

**Asked after the design was written**, when the user looked at the result and
asked whether Scheduler + one global Runner + a per-agent `mainloop` + a Monitor
is a redundant four-layer stack. Four threads: workflow engines
(`findings-arch-workflow.md`), agent frameworks (`-agentfw.md`), supervision
systems (`-super.md`), and an audit of our own documents against our own code
(`-ours.md`).

**The layer count was not the problem.**

| | Runner-equivalent layer | Watcher separate from the loop that runs the work |
|---|---|---|
| Workflow engines | **6 of 6** | **6 of 6** |
| Supervision systems | — | **5 of 6.** The exception, Pekko Typed, cannot see a hung actor and dropped `Escalate` |
| Agent frameworks | rare | **1 of 9.** Retries and turn limits are counters *inside* the loop that is misbehaving |

**The agent-framework population disagrees and the survey says why**: they mostly
run one graph in one process for one request, where a wedged loop is the caller's
problem and Ctrl-C is the recovery. **Two of the nine are moving toward a split;
none is moving away from one.**

**What was actually wrong was the Runner's cardinality.** Runner-equivalents split
into shared and per-unit-of-work, and every system in the per-unit column made it
so **because that object holds the per-task state a supervisory loop reads**;
every system in the shared column keeps that state somewhere else (Temporal in
server state, Dagster in a database). **Ours was shared and kept it nowhere** —
which is exactly what the monitor design had recorded as O1 a day earlier, mistaking
the bill for the choice for a missing accessor.

**Four defects the self-audit confirmed, none of them about redundancy:**

| | |
|---|---|
| Who spawns and joins a thread | **No document says, for either loop.** Zero `Thread(` or `.join()` in 42 component files |
| `task_graph` design §8.5 | Breaks spec §3.2.1 **twice** — skips `OUTPUT_VALIDATING`, and acts on `is_end` where the spec says the scheduler must not |
| What re-enters a non-leaf after its subgraph | **Absent everywhere**, and two documents disagree on whether the runner is even called |
| Threads today | **1.** Documents implied K+2 to 3K+N+1, with four of six rows unspecified |

**And one correction to this stage's own output**: `monitor` design §9 said
`agent.Runner` *"must already hold"* a task→executor map. The inference was sound,
"already" was false, and the map a runner is forced to hold is
`(Task, Agent, OnDone)` — **no executor in it**. The proposed `executor_of`
assumed more than existed.

### What the redesign settled — spec rev. 14, 2026-08-28

The user's framing, and it was the right cut: **one question is "which thread
pushes the phases forward", the other is "who decides the next phase".** They are
not alternatives — an event mechanism with no thread underneath it still needs
one.

| | |
|---|---|
| **The monitor becomes the task's event loop** | Two channels through one `report()`. Planned advances handled by **code, always**; unplanned outcomes decided. The queues differ in their collapse rule, because deduplicating an advance skips a phase |
| **No AI on the ordinary path, permanently** | The queue split is what guarantees it. `_advance` sits on `BaseMonitor` where no subclass can replace it, so the guarantee is structural rather than a convention |
| **The task owns the thread; the agent borrows it** | `TaskAttempt`, one per dispatch. Threads converge to the executing leaves |
| **A non-leaf holds no thread during its subgraph** | Its thread ends at `unfold`; the re-entry takes a new one and is the **same `Execution`** |
| **The scheduler is not in the re-entry** | Reported, argued, and rejected — my proposal. Routing it there makes one task's progress depend on the scheduler *observing another task's status*, against §2 principles 2 and 4 and §3.2.1's own rule on `is_end`. The chain is subtask-monitor → parent-monitor → `enter_phase` → a thread from the runner |
| **No thread pool** | Measured: 71 μs fresh against 21 μs pooled. The 50 μs is not worth a **second admission control the scheduler cannot see** — the leases are already one |
| **"Nothing monitors the monitor" becomes two mechanisms** | Because ordinary progress now depends on this loop. Measured: a thread's uncaught exception prints a traceback and dies with the exit code unchanged and producers none the wiser, so a `threading.excepthook` is required; plus a heartbeat checked against N stale periods by a pure function over one float — the "make the top trivial" answer, against systemd's hardware watchdog and Ray's delegation outward |

**Three things I got wrong in the argument, recorded because two of them were
mine and the user caught both.** I claimed OTP's "a supervisor that works cannot
supervise" ruled out the monitor handling normal events — it does not, since
dispatching a phase is not the blocking start/stop that quote is about. I claimed
separating the queues was insufficient — it is sufficient, and it is what makes
the dedup objection vanish. And I anchored on the scheduler through two rounds
after the spec had already forbidden what I was proposing.

**Criteria 19–26.** Propagated to `task_graph` spec rev. 14 and design rev. 14,
`agent` spec rev. 6 and design rev. 7, `interfaces.md` rev. 4, main spec §10 and
ROADMAP §2 / §7.1.

## Stage four — implementation, and what executing the contract found

Eight packages built in parallel, 2026-08-28. The findings below are the ones
that **only appear once there is code**, recorded here because they correct a
design document's premise and the module that found them may not edit the
document that holds it.

### `monitor` design §14's O6 — its premise is measured false for a non-leaf

O6 reads *"the cost of an advance is a thread handoff rather than a poll period.
That is derived from the design, not measured."*

**It has now been measured, and the bolded clause is false for a non-leaf.**
`enter_phase(RUNNING)` unfolds the subgraph and submits it *inside the
transition*, and `submit` dispatches. `task_graph`'s probe
(`scratch/impl-2026-08/task_graph/probe_unfold_cost.py`):

| shape | n | passes | `_ready` calls | ms |
|---|---|---|---|---|
| chain | 50 | 50 | 1277 | 6.29 |
| fan-out | 50 | 50 | 150 | 6.52 |

Quadratic on a chain, linear on a fan-out, **~6 ms either way at n=50** — and
that 6 ms is `FakeRunner`, where `runner.start` is a dict write. The real cost is

```
monitor's lock hold  =  ~6 ms of task_graph bookkeeping  +  n × Runner.start
```

because `_dispatch_pass` calls `runner.start(...)` **while holding the scheduler
lock**.

**Since first recorded, all three terms have been measured, and the entry above
named the wrong one as unmeasurable.** `Runner.start` **never touches a
backend** — it builds an attempt and starts a thread, while `env_mgr.prepare` and
`select_backend` run on the attempt's thread, *outside* the lock. So what is
multiplied by *n* is thread creation, not an executor launch.

| term | owner | n=50 | probe |
|---|---|---|---|
| `task_graph` bookkeeping | `task_graph` | ~6.3 ms | `probe_unfold_cost.py` |
| **`Recorder.open`, file store** | **`monitor`** | **~5.1 ms** | `p8_recorder_open_cost.py` |
| `Runner.start` thread creation | `agent` | ~3.5 ms | `p5_thread_cost.py` |

**3.5 ms against 6.3 ms is the same order, not a multiple, so O6's premise is
closer to fine than the correction first implied.**

**The term nobody had looked at is `monitor`'s own, and it is a third of the
total.** `Recorder.open` runs inside `TaskAttempt.__init__` → `Runner.start` →
`_dispatch_pass`, so it is **a filesystem write under the one global lock**, once
per subtask, serially:

```
store                first open     repeat   n=50 first
MemoryStoreMgr           4.7 us     0.7 us      0.23 ms
JsonFileStoreMgr       102.4 us     9.6 us      5.12 ms
```

Latent rather than live — `build_registry` defaults to `MemoryStoreMgr`. It
appears with `JsonFileStoreMgr`, the store for a real run. **The other two terms
degrade predictably; a write under a lock degrades badly on a slow or networked
store, and nothing says the store is local.**

The fix is one line and is `agent`'s to take or refuse: move `Recorder.open` from
`TaskAttempt.__init__` to the top of the attempt's thread, which holds no lock.
Same write, off the lock path. `monitor` checked they could not fix it from their
own side first — the bulk is the write, not the `exists` probe, and the marker
cannot be made lazy without reopening the `recorder` decision.

**The half that is `monitor`'s own, and is not in `task_graph`'s account:** the
monitor's loop has **one consumer**. While it is inside that transition it drains
no other task's planned advance, so for a global monitor watching *N* tasks, one
non-leaf's advance stalls every other task's phase progress for the whole
duration.

Three things were deliberately *not* done, and each is the right restraint:

| | |
|---|---|
| The single consumer was not changed | Spec §5.2 chose one on purpose. Adding one is small — and making the move on an unmeasured number is the guess the measurement exists to avoid |
| The submits were not batched | It is the biggest lever on the lock hold, and `task_graph` refused it **on correctness**: per-submit dispatch is what puts `is_start` first, and batching puts it last under `DepthFirstPolicy` on a fan-out. *A latency win that costs a correctness assertion is not a win* |
| No number of `monitor`'s own was produced | Under `FakeRunner` it would restate the loop's structure rather than measure anything. Saying the term is unmeasurable is the honest deliverable |

**The measurement that closes it**: wall time of `Runner.start` entry-to-return
against a real backend, first-call and steady-state separately. A fan-out at
n=10 answers it by subtracting `task_graph`'s measured 1.35 ms.

### `validator` criterion 10's attribution leg was weaker than reported

Found by `agent` asking one question about a declared shape. The phase agent id
was

```python
agent_id = f"{task.current.agent_id}:{kind.value}"
```

— **the producing task's agent with the phase appended. Distinct per phase, and
not a distinct agent.** Criterion 10 wants a checking context the producer cannot
reach, and a string suffix does not buy that. The hook and separation legs are
unaffected; the attribution leg was being satisfied by a derived string.

Named in the code so nobody reads the string as satisfying §8.1, and closed by
`agent`'s `validator_executor` — a fresh executor per phase, which `validator`
confirmed satisfies it because **the asserted `agent_id` is ours and not the
SDK's**, so it needs no SDK feature and works for a `kind: program` body too.

**Both defects at this seam were found the same way**: the other side read the
declared signature against its own import rule and asked one question. Neither
was catchable by either package's own tests, and each cost one message.

## Core principles

Unchanged from the spec stage, and they still decide questions daily:

1. **Validator-first.** Not "the output looks right" but "the output passes its
   validator".
2. **Producer and validator contexts are separated, by hook and by OS**, never
   by convention.
3. **Anything that can be code is code.** An agent fills blanks in a fixed
   procedure; it does not invent the procedure. A rule that lives only in a
   prompt is not a rule.
4. **The scheduler decides when, never what** — mechanically enforced by
   `tests/task_graph/test_authority.py`. Nothing in this stage weakens it.
5. **Record and replay is the v1 scope.** The catalogue is static; the instance
   count need not be.
6. **Everything is decoupled, or stays easy to decouple.**

## Method

**Suspend, don't conclude.** Without first-hand evidence, describe the
observation and stop. A design document that asserts a mechanism nobody measured
is how the spec set would start being wrong.

**Cross-reference, never restate.** Where a spec or another design already says
something, link to it.

**Every non-obvious choice carries its reason inline.** That is what makes these
documents reviewable.

**Scratch stays in `agent_sys/scratch/`** (gitignored) and nowhere else. Probe
scripts are kept, not deleted — they are the evidence.

## Key references

| | |
|---|---|
| Task brief | `aiopt.whole.system.md` (repo root, gitignored) |
| Design house style | `agent_sys/task_graph/docs/design.md` rev. 10 |
| Spec house style | `agent_sys/task_graph/docs/spec.md` |
| Module 1's findings | `agent_sys/scratch/design/findings-main.md` |
| Module 2's findings | `agent_sys/scratch/design/findings-handoff.md` |
| Module 3's findings | `findings-validator.md`, `-isolation.md`, `-registry.md`, same directory |
| Module 4's findings | `findings-taskgraph-mine.md`, `-reentrancy.md`, `-criteria.md`, `-nesting.md`, `-cascade.md`, same directory |
| Modules 5-8's findings | `findings-agent-*.md`, `findings-closure-*.md`, `findings-envmgr-*.md`, `findings-demo-*.md`, same directory |
| Module 9's findings | `findings-monitor-{loop,record,sandbox,push}.md`, and the layering survey `findings-arch-{workflow,agentfw,super,ours}.md`, same directory |
| Scenario, chapters 1–5 + appendix A–G | `/home/yihou/dev/git.16-19/wsl_sync/AIHIP/materials/infera-ai-optimization-kickoff.html` |
| `task_graph` prior art (RCPSP) | same directory, `agent-task-graph-prior-art.html` |
| claude-agent-sdk reference | `https://code.claude.com/docs/en/agent-sdk/{overview,python,sessions,hooks,permissions,custom-tools,subagents}.md` |

The kickoff report earns its place twice: **chapter 3's six-step main loop** is
the reference graph the design must express, and **appendix A's Hyperloom
teardown** — "the candidate writes the exam, the answer key, and grades it" — is
the failure the producer/validator separation exists to prevent.
