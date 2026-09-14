# handoff

The content and lifecycle layer. A handoff is a module's input or output — the
only thing that crosses between tasks, which makes it the only place where
quality can be enforced system-wide.

The runtime **slot** — `Handoff`, `HandoffVersion`, `open_next()`, `seal()` — is
`task_graph`'s. The two layers meet at one point: `Handoff.type` names a
**kind**, and a kind is what this package specifies.

## Documents

| | |
|---|---|
| [`docs/spec.md`](docs/spec.md) | What a unit of transfer carries. Rev. 5, 17 acceptance criteria |
| [`docs/design.md`](docs/design.md) | Files, interfaces, the measurements behind each choice, and the test plan |
| [`../docs/interfaces.md`](../docs/interfaces.md) §4.2 | Normative for what leaves this package |
| [`protocols.py`](protocols.py) | The frozen seam, importable. **Every type is declared there and re-exported here** — a second class of one name is the failure this package would be least able to see |

## Layout

```
handoff/
├── protocols.py     the frozen seam (+ .pyi)
├── errors.py        five names imported from protocols.py, plus BindingConflict
├── digest.py        the tree walk (§4.2) and the canonical encoder (§4.6)
├── readme.py        CommonMark section extraction and the three checks (§9)
├── pointer.py       RFC 6901 resolution with three-way failure (§8.4)
├── containment.py   check_contained (§7)
├── locality.py      the anchored allow-list plus oracles (§10)
├── content.py       the four content types, the file/data item split
├── kind.py          HandoffKind and the kind's own load-time checks
├── registry.py      HandoffSpecRegistry, the reverse index, the agreement check
├── verdict.py       validation.yaml — a sibling of content/, outside the digest
└── store.py         HandoffStore and FilesystemStore
```

`digest`, `readme`, `pointer`, `containment` and `locality` import nothing from
this package except `errors`. That is what makes them testable without a store
and reusable by `validator` without an import cycle.

## Libraries adopted, and why

Mission rule 5. Five decisions, three of them dependencies this package adds.

| Concern | Chosen | Rejected, and why | Where |
|---|---|---|---|
| **JSON canonicalisation** | **`rfc8785`** 0.1.4, wrapped by a rejecting encoder | `json.dumps(sort_keys=True)` is *not* a canonicalisation — RFC 8785 sorts by UTF-16 code units and Python's `sorted()` does not for non-BMP keys. `canonicaljson` is a different specification and [does not implement it](https://github.com/matrix-org/python-canonicaljson/issues/22) (open since 2019). **`jcs` silently rounds `12345678901234567168`**, which yields a correct-looking digest for the wrong value | `digest.py` |
| **RFC 6901 pointer** | **`python-jsonpath`** 2.2.1 (`JSONPointer`) | The only one of six measured that separates *malformed*, *missing* and *null*. `jsonpath-ng` leaks 19 stdlib exceptions from validly-parsed queries (issue #203), so `except JSONPathError` does not hold; `jsonpointer` 3.1.1 raises **one class for both** failures; `referencing` leaks bare `ValueError` and percent-decodes. And no *JSONPath* library can work: RFC 9535 §2.5.1.2 forbids a valid query from erroring | `pointer.py` |
| **Markdown headings** | **`markdown-it-py`** 4.0.0 | A `^#{1,6}\s+` regex is wrong in **both** directions, measured: it misses setext and ≤3-space headings and finds headings inside code fences — and the false positives are the security-relevant half. `markdownlint`'s MD043 is the only off-the-shelf required-headings rule and provably cannot express per-kind requirements (#32); `remark-lint` has 85 rules and no such rule | `readme.py` |
| **JSON Schema** | **`jsonschema`** (already a design-set dependency) | `check_schema` as a **named step**, never `$ref`-ing the metaschema: measured, `$ref` turns `{"type": "nonsense"}` into 8 identical errors | `content.py`, `kind.py` |
| **YAML** | **PyYAML** `safe_dump`/`safe_load` | Already declared. The manifest and the verdict file are ours, never hand-written, so YAML 1.1's `NO`-is-`False` trap cannot reach them | `verdict.py`, `store.py` |

Three written here instead, each with the reason:

| Written | Instead of | Why |
|---|---|---|
| `tree_digest`, ~40 lines | `checksumdir`, `dirhash`, `git hash-object`, tar + sha256 | `checksumdir` hashes *"only file contents and not filenames"* by its own docstring — two trees with contents swapped hash identically. `dirhash` has no option for the executable bit at all and an open pickling bug (#34). tar is rejected outright: a bare `touch` changes the digest, and five flags are needed to suppress the variation. **Git's own tree format cannot represent an empty directory**, and we record them |
| `HandoffStore`, 8 methods | `fsspec`, MLflow's `ArtifactRepository`, a CAS | `fsspec` is 68 public callables where ~6 must be written, and its own author called the API *"incomplete"* on apache/arrow#4225. A CAS fails v1: REAPI cannot list by prefix, cannot name, and blob lifetime is only a SHOULD |
| `locality.check` | diffoscope, a shape regex alone, Nix's reference scan | diffoscope is differential and needs two inputs. A shape regex alone is **96% noise**, measured on this repository: 650 matches, 23 genuinely local |

## Status

**2172 lines of package, 2139 of tests. 141 tests here, plus 8 in
`tests/interfaces/test_composition.py` and 6 in
`tests/interfaces/test_handoff_layout.py`.**

**All 17 acceptance criteria map to the test [`docs/design.md`](docs/design.md)
§12.1 names for each**, and all nine of §12.2's measured-fact tests exist, and
`StoreConformance` — verified by name against the tree, not from memory.

```bash
pip install -e agent_sys
pytest agent_sys/tests/handoff
pytest agent_sys/tests/interfaces/test_composition.py    # the assembled system
```

Two tests assert a claim that is *structural* rather than behavioural, because
in both cases the design argues the structure **is** the guarantee:
`test_copy_out_refuses_to_return_store_path` (that `dst` has no default) and
`test_staging_is_a_sibling`.

### Where the tests are, and why some are not here

| | |
|---|---|
| `tests/handoff/` | everything this package can prove alone |
| `tests/interfaces/test_composition.py` | **the assembled system** — real `build_registry`, real `declare(types=...)`, real `load_package`, then `put` → `copy_out` → manifest. It exists because two defects were catchable nowhere else: a store built with no `KindSource` published a handoff missing four of five required README sections with `kind: ""`, and a `load_report`/`report` name mismatch behind a `getattr` default silently disabled `closure`'s escape-hatch check. Both had every package suite green |
| `tests/interfaces/test_handoff_layout.py` | the `version_dir` drift test — see below |

## The path shape is declared twice, and that is paid for

`handoff.store.version_dir` and `env_mgr.fs.layout.handoff_version_dir` both
compute `<root>/<hid>/v<N>/`. Design §6.2 says *"exactly one function computes a
path and the on-disk shape is private"*, so this is a violation — **by
construction, not by carelessness**: `docs/interfaces.md` §4.6 permits `env_mgr`
to import `task_graph` and nothing else of ours, so it cannot call
`version_dir` even if it wanted to, and `grants.py` and `meta.py` need the shape
to grant access to it.

The alternatives were a package edge for one function, or a shared constant with
no honest home — a path layout belongs to neither `task_graph` nor
`spec_loader`. So the shape stays declared twice **and something checks it**:
`tests/interfaces/test_handoff_layout.py`, six tests, the same bargain
`test_pushable.py` strikes one directory over. *The test is not a nicety
attached to the decision; it is the decision's price.*

**If you change the on-disk layout, that test is what tells `env_mgr`.**

## A version is allocated before it is written — `interfaces.md` §4.14

`put` copies a finished `content/` in and publishes it. That works for anything
publishing on its own behalf, and it could not work for an **agent's** output:
the version number did not exist until the attempt closed, so `env_mgr`'s
kind-named grant had no `<store>/<hid>/v<N>/` to resolve to and raised
`UnresolvedGrant` for the whole of the attempt that was supposed to fill it.

So the write splits into the two moments the ruling needs:

| | |
|---|---|
| `allocate(hid) -> int` | at **dispatch**. Creates `<root>/<hid>/v<N>/` and nothing else. This directory *is* the agent's grant |
| `seal(hid, version, *, producer)` | at **close**. The content is already at `v<N>/content/`; nothing is copied, the digest is taken over the bytes as they lie |

**`MANIFEST_FILE` is what makes a version published**, and it is written last.
`list_versions`, `latest` and `exists` all read that one fact, so an allocated
directory is invisible until it is sealed.

**A failed attempt leaves a hole, and a hole is skipped rather than compacted.**
v3 is allocated, the attempt fails, v3 stays absent forever and v4 is allocated
next. Renumbering would move an artefact a digest already names.

### What the hole did before it was filtered, measured

`interfaces.md` §4.14 recorded *"whether a pre-allocated empty `v<N>` pollutes
any other reader of `latest`"* as **not measured**. It does, and the failure
mode is an uncaught exception rather than a wrong answer
(`scratch/impl-2026-08/handoff/probe_hole.py`):

| reader | with an unsealed `v<N>` on top | now |
|---|---|---|
| `agent/gate.py:90` — `list_versions[-1]`, then `get_manifest` | bare `FileNotFoundError`, out of `run_gate` | the hole is not in the list; the gate reports `OUTPUT_ABSENT`, which is the truth about the attempt |
| `cli/main.py:761` — `read_verdicts` per version | `Malformed` | skipped |
| `exists(hid, N)` | `True` | `False` |
| `_next_guess` | correct already — it counts holes, and **must**, or allocation would hand out a number in use | unchanged |

**Nothing reaps the directory, and that is the decision, not an omission.**
Reaping would race an agent that is still writing, and it would destroy the one
on-disk trace of a failed attempt. Since a hole is invisible to every published
read, there is nothing to reap for correctness — only for disk, which no
criterion covers today. It is cheap to add later and expensive to undo.

## The one pending item with a clock on it

**`done_by_self_check` is still not on `Manifest`, and the reason has changed.**

The reason recorded here was that `docs/interfaces.md` §5.14 resolved
publication to a supervisor-side pull, under which the completeness gate ran
before any manifest existed. **That premise is dead**: §4.14 dissolves §5.14,
and §4.16 measured `agent/gate.py:90` calling `store.get_manifest(hid, version)`
— the gate runs against the store, so a manifest is exactly what it has. A
manifest field could carry it.

**What blocks it now is that no producer has a way to claim it.** The field is a
statement by the *agent* — `monitor` §4.1.2, *"set by the producing agent when it
believes the package is complete"* — and there is no channel from an agent's body
to `seal`. Landing the field without one forces a default, and every default is
wrong:

| default | what happens |
|---|---|
| `False` | every existing producer trips `SELF_CHECK_UNSET`; the gate blocks the demo |
| `True` | the check is inert, and now lies about being satisfied |
| `None` | behaviour identical to absence — but `agent/gate.py:101`'s tolerance clause (*"absent means `handoff` has not landed the field"*) becomes permanently untrue, so the guard can never be tightened |

**So it lands with a channel or not at all.** The shape that works is
`seal(..., done_by_self_check: bool)` as a **required** keyword — `seal` is only
called on the agent-produced path, so nothing else has to answer, and no default
is needed. That needs `agent` to carry the claim out of the body first.

`tests/handoff/test_verdict.py::test_the_manifest_does_not_carry_done_by_self_check`
guards it, and pins `Manifest`'s whole field set so a change has to be
deliberate. It asserts the **absence of the mechanism**, not the presence of a
consequence — a marker written the other way survives its own fix and goes on
reporting a gap that has closed.

`tests/handoff/test_verdict.py::test_the_manifest_does_not_carry_done_by_self_check`
guards it, and pins `Manifest`'s whole field set so a change has to be
deliberate. It asserts the **absence of the mechanism**, not the presence of a
consequence — a marker written the other way survives its own fix and goes on
reporting a gap that has closed.

## Three things a reader should know before changing anything here

**The digest algorithm is registered as `agent_sys.handoff.tree.v1`, and the
reference vectors in `tests/handoff/test_digest.py` are pinned.** A change to
the walk is a `v2`, never an edit. Matrix's `canonicaljson` is the cost of not
having done this: fixing a non-conforming digest required a whole new room
version, because the wrong digests were already persisted.

**`delete_version` is deliberately absent from the Protocol.** Garbage
collection between an artefact and its verdict is unsolved everywhere — OCI
distribution-spec#378 open since 2023, REAPI#138 open ~7 years — and the design
declines to invent an answer. Decide before anything is deleted, not after.

**The locality check is sound on oracle hits and best-effort otherwise, and it
says which fired.** Three false negatives are known and none is closed by more
regex: compression, runtime concatenation, and Nix's own caveat that a clean
scan asserts nothing was *found*. Every project that made a path check mandatory
acquired false positives within a release or two.
