# Handoff — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 2 — 2026-08-27. **The stage-three consistency pass.** D1 is retired: spec rev. 5 adopts JSON Pointer, so the design no longer deviates (§8.4, §14). The two-way agreement check reads `inputs`, not the `binds_to` that exists in no model (§8.3). `Verdict` is named as this module's type and `validator`'s `VerdictRecord` as its payload (§6.1). O2's dependency picture is corrected against measurement (§15). (rev. 1: 2026-08-26. Initial) |
| Implements | [`spec.md`](spec.md) rev. 5, criteria 1–17 |
| Language | Python ≥ 3.10. PyYAML, `python-jsonpath`, `markdown-it-py` |
| Scope | Content, the digest, storage, the validator binding, and the two checks that gate admission |
| Part of | [`../../docs/design.md`](../../docs/design.md) — the system design |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces.
**It adds no requirements.** Where it makes a choice the spec left open, the
choice is stated with its evidence; where implementing the spec exposed a
contradiction, §14 says so rather than papering over it.

The spec's 17 acceptance criteria are the definition of done. §12 maps every one
to a named test.

**This document specifies interfaces, not bodies.** A body appears only where
the ordering of steps *is* the design decision — which here is the tree walk
(§4.2), publishing a version (§6.3), and the two-way agreement check (§8.3).

### 1.1 What this document covers

The content and lifecycle layer, as spec §1 defines it:

- **Content** — the README plus the typed dictionary. §3.
- **The digest** — what it covers, how it is computed, and why the verdict file
  cannot move it. §4.
- **The consumption protocol** — copy out, work on the copy. §5.
- **Storage** — the interface, and the two instances. §6.
- **Permission by containment**, and the assumptions it rests on. §7.
- **The validator binding**, including the two-way agreement check. §8.
- **The two checks that gate admission** — README structure (§9) and locality
  independence (§10).

### 1.2 What it defers

| Deferred | To |
|---|---|
| The runtime slot: `Handoff`, `HandoffVersion`, `open_next()`, `seal()` | `task_graph` — **already built**, spec §3.1 |
| Rendering, schema validation, and admission to the registry | [`../../docs/design.md`](../../docs/design.md) §3, §5 |
| What a validator *is*, and how a verdict is decided | `validator` |
| Enforcing the permission model at the OS level | `env_mgr` |
| Which agent skills produce each content type | `../../docs/TODO.md` |

The split with `task_graph` is the one worth restating, because two modules own
one word. **`Handoff` is the slot; `HandoffKind` is the kind.** `task_graph`'s
`HandoffMgr` already holds slots keyed by `HandoffId` and persists them;
`Handoff.type` is a string, and this module is what that string names. Nothing
here reaches into `task_graph`'s models, and `task_graph` does not import this
package.

---

## 2. Layout and import graph

The `handoff/` package that [`../../docs/design.md`](../../docs/design.md) §2
reserved. `docs/` already exists.

```
agent_sys/handoff/
├── __init__.py
├── docs/
│   ├── spec.md
│   └── design.md              this document
├── kind.py                    HandoffKind: the spec's shape and its load-time checks
├── registry.py                HandoffSpecRegistry(SpecRegistry) — kinds, and the reverse index
├── content.py                 Content: the README + items pair, and the four content types
├── digest.py                  the tree walk. §4
├── readme.py                  CommonMark section extraction and the three checks. §9
├── pointer.py                 RFC 6901 resolution with three-way failure. §8.4
├── locality.py                the anchored allow-list. §10
├── store.py                   HandoffStore: the interface, and FilesystemStore
├── verdict.py                 the validation record beside the artefact. §4.1
└── errors.py                  Malformed, DigestMismatch, NotContained, BindingConflict
```

The schema is **not** here. `handoff.schema.json` lives in
`spec_loader/schemas/` with the other four, for the packaging reason
[`../../docs/design.md`](../../docs/design.md) §2.2 measures.

### 2.1 The import graph

```
             spec_loader ──────────────┐
                  │                    │
                  ▼                    ▼
handoff.errors ◄── handoff.kind ── handoff.registry
      ▲                 │
      │                 ▼
      ├── digest ◄── content ──► readme
      ├── pointer                  │
      ├── locality ◄───────────────┘
      └── verdict ◄── store
```

Acyclic, and one rule keeps it so: **`digest`, `readme`, `pointer` and
`locality` import nothing from this package except `errors`.** They are pure
functions over bytes, a tree, or a document. That is what makes them testable
without a store and reusable by `validator` without an import cycle.

`store` imports `verdict` and not the reverse: a verdict is data that a store
persists, and it does not know where it is persisted.

---

## 3. Content — a README and a typed dictionary

### 3.1 The shape on disk

Spec §3.1 fixes the content as a README plus `items`. On disk that is:

```
<version-dir>/
├── content/                 ← everything under here is digested. §4
│   ├── README.md
│   └── items/
│       ├── <key>            a file, or a directory of files
│       └── items.json       the typed values that are not files
├── validation.yaml          ← the verdict record. NOT digested. §4.1
└── manifest.yaml            ← digest, kind, producer, timestamps. NOT digested
```

**`content/` is a subtree, and the sibling placement is the whole of criterion
7.** §4.1 gives the measurement and the prior art.

`items` is split in two because its values are of two kinds. A `reproducible`
handoff's `logs` is a file — possibly a large one — and forcing it through JSON
would be a mistake nobody could undo later. A `structured_text` handoff's
`schema` is a document. So: **a value that is a file is a file; a value that is
data lives in `items.json`.** The kind's `items_schema` declares which is which,
and `content.py` is what enforces the split.

```python
@dataclass(frozen=True)
class Content:
    root: Path                     # the content/ directory
    readme: Path                   # root / "README.md"
    items: Mapping[str, Item]      # key -> Item

@dataclass(frozen=True)
class Item:
    key: str
    kind: Literal["file", "tree", "data"]
    path: Path | None              # for file and tree
    value: Any | None              # for data
```

`Item.kind` is three-valued and not two, because a `code` handoff's `codes` is a
directory and a `reproducible` handoff's `script` is one file, and the digest
walks them differently (§4.2).

### 3.2 The four content types

Spec §3.2 fixes what each type carries. The design's contribution is to make it
a **table in one module**, so the four sets are visible together and a fifth
type is a table row rather than a code change:

```python
CONTENT_TYPES: Mapping[str, ContentType] = {...}

@dataclass(frozen=True)
class ContentType:
    name: str
    required_items: frozenset[str]
    optional_items: frozenset[str]
    readme_sections: tuple[str, ...]      # ordered for the template; checked as a set. §9
    extra_checks: tuple[Check, ...]
```

| Type | `required_items` | `extra_checks` |
|---|---|---|
| `reproducible` | one of `script` \| `command`; `result`; `env` | **`env` required when `script` or `command` is present** — spec §3.2, criterion 4 |
| `code` | `codes` | — |
| `structured_text` | one of `text.json` \| `text.yaml` \| `text.xml` | `schema` validates the text when present |
| `text` | `content` | — |

`extra_checks` exists because criterion 4's rule is not expressible in the
`items_schema` a package author writes — it is a rule *about* that schema, and
it is checked at kind-admission time, not at content time. §3.5.

**`readme_sections` is a tuple here and a set at the check.** Ordered so a
template can be generated from it; checked as membership because
markdownlint #394 is a required-headings matcher that accepted every document —
it failed **open**. §9.2.

### 3.3 Runtime-generated keys

Spec §3.1 permits `items` keys the kind did not declare, "when the number or
names of items are not knowable in advance". The mechanism is the one JSON
Schema already has: the kind's `items_schema` carries
`additionalProperties: {...}` where runtime keys are permitted and
`additionalProperties: false` where they are not. Criterion 5 is then a property
of the package author's schema, checked by the same `jsonschema` pass every
other field gets — **no second mechanism, and no `allow_runtime_keys` flag.**

The one thing the design must add: **a runtime key is still a key, and it can
contain any character.** That is why §8.4 chooses JSON Pointer, whose `~0`/`~1`
escaping addresses a key containing `/` or `.`; and it is why §7.3's name
allow-list applies to the *storage* name and not to the item key.

### 3.4 Scope is one tag with four values, and it decides three things

Spec §4.1's table. The tag is a field on the kind, and it is the only input to
three otherwise-unrelated decisions:

```python
class Scope(str, Enum):
    FIXED_REQUIRED   = "fixed.required"
    FIXED_OPTIONAL   = "fixed.optional"
    ADDONS_TEMP      = "addons.temp"
    ADDONS_KNOWLEDGE = "addons.knowledge"
```

| Scope | Storage | Permission | Retention |
|---|---|---|---|
| `fixed.required` | the handoff store | the consuming task's, `task_graph` §3.2.2 | life of the graph run |
| `fixed.optional` | same | same | same |
| `addons.temp` | the **playground**, `env_mgr`'s | injector and receiver only | discardable at run end |
| `addons.knowledge` | the **knowledge store** — a separate instance, §6.5 | broadly readable, narrowly writable | outlives every run |

**None of those three columns is implemented by reading the tag at use time.**
Storage is decided once, at publish, by which store the kind resolves to;
permission is `env_mgr`'s and is a property of the zone the artefact lands in;
retention is a property of the store's root. So the tag is consumed **once**,
and everything downstream sees only where the artefact is.

That is why criterion 14 is asserted *"by where the artefact lands, not by
reading the tag back"* — a test that reads the tag would be testing that a
string round-trips. `test_scope_tags_land_where_declared` resolves the store for
each of the four and asserts the path.

**`addons` cannot satisfy a `fixed.required` input**, spec §4.2 and criterion
15. This is a check on the *binding*, not on the artefact: at closure time, an
input declared `fixed.required` whose bound kind is `addons.*` is a load error.
The reason spec §4.2 gives is the one that matters — *"if it could, the declared
interface would be advisory"* — and it belongs at load because a graph whose
interface is satisfiable by injection is misdeclared, not misrun.

### 3.5 The kind's load-time checks

Spec §8 lists five. They run after the schema pass, as a `SpecRegistry`
subclass's own checks
([`../../docs/design.md`](../../docs/design.md) §5.1):

```python
class HandoffSpecRegistry(SpecRegistry):
    kind = "handoff"

    def check(self, spec: Mapping, *, origin: str) -> list[Problem]: ...
```

| # | Check | Note |
|---|---|---|
| 1 | name unique | The base class's, not this one's |
| 2 | `items_schema` is itself a valid schema | **`check_schema` as a named step, never `$ref`** — [`../../docs/design.md`](../../docs/design.md) §3.5 measured that `$ref`-ing the metaschema turns one fault into 8 identical errors |
| 3 | every named validator resolves | **Deferred to the closure pass** — the validator registry may not be loaded yet. §8.2 |
| 4 | at least one validator, or the flag | The flag's report is a return value, not a log line. §8.5 |
| 5 | `reproducible` + `script`/`command` ⇒ `env` | Reads the *author's* `items_schema` and asks whether it *permits* a document with `script` and no `env` |

Check 5 deserves a sentence because it is subtle. It is not "does this content
have `env`" — no content exists at kind-admission time. It is **"could a
document satisfying this `items_schema` have `script` without `env`"**, which is
answered by testing the author's schema against a synthetic document. That is
cheap and exact for the shape criterion 4 describes, and it is stated here
because the naive reading of criterion 4 puts the check in the wrong phase.

---

## 4. The digest

Spec §3.3 fixes sha256 and fixes the *scope* — content in, metadata and
validation status out. It defers canonicalisation to this stage and says why:
"getting it wrong makes the field useless and it should not be decided by
accident".

### 4.1 Digest the `content/` subtree; the verdict is its sibling

Measured (`scratch/design/probes-handoff/probe_yaml_sidecar.py`): the content
digest is unchanged by creating and then rewriting a sibling `validation.yaml`,
and survives a `copytree` of the whole version directory.

**The alternative — hash the version directory and exclude `validation.yaml` by
name — is not stable.** It re-hashes on every rewrite unless the exclusion is
applied at every level of the walk, and it invites the bytes-versus-`str`
comparison bug §4.2 has to avoid anyway.

**Criterion 7 is a structural fact, not a convenience.** PyPA's binary
distribution format: *"every file except `RECORD`, which cannot contain a hash
of itself, must include its hash"*, with `RECORD.jws`/`.p7s` excluded one level
up. Debian's `md5sums` breaks the recursion identically, and its `Release` file
names 772 SHA256 entries and never itself while `Date`/`Valid-Until` are
re-issued weekly without moving one package digest.

The general rule, from REAPI's Action/ActionResult split: **does omitting this
let a wrong answer masquerade as a right one?** Bazel puts `timeout` in the
Action key for exactly that reason and `execution_metadata` — worker, timings —
in the value. Our producer, timestamps and verdict all fail the test, so all
three are correctly outside.

The cost of getting it wrong is documented in two places. Git puts committer
identity and dates in the commit hash and pays with `git patch-id`, a second
metadata-free identity scheme existing solely to undo the pollution. OCI's image
config digest includes `created`, and buildkit has been paying since #4057,
#4231, #5960, #6704, #3180.

### 4.2 The tree walk, specified exactly

**Nothing in the spec set names the algorithm**, and in-toto registers its
`dirHash` with a shell equivalent precisely because leaving it implicit makes
two implementations disagree silently. So this is a specification, not a
description:

```
file   := sha256(contents)
link   := sha256(os.readlink(path))            # the target text, never followed
dir    := sha256(b"tree " + str(len(body)).encode() + b"\0" + body)
           where body = concat over sorted entries of:
               mode + b" " + name_bytes + b"\0" + digest32

mode      ∈ {b"100644", b"100755", b"120000", b"040000"}
name_bytes = os.fsencode(name)
sort key   = name_bytes                         # plain byte order. §4.3
digest of the handoff := dir(content/)
```

The one place ordering *is* the design:

```python
def tree_digest(root: bytes) -> bytes:
    entries = []
    for e in os.scandir(root):                      # bytes in, bytes names out
        st = e.stat(follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode):
            mode, d = b"120000", sha256(os.readlink(e.path)).digest()
        elif stat.S_ISDIR(st.st_mode):
            mode, d = b"040000", tree_digest(e.path)
        elif stat.S_ISREG(st.st_mode):
            mode = b"100755" if st.st_mode & stat.S_IXUSR else b"100644"
            d = _file_digest(e.path)
        else:
            raise Malformed(f"{os.fsdecode(e.path)}: not a file, directory or symlink")
        entries.append((e.name, mode, d))           # e.name is bytes
    entries.sort()                                  # byte order, by construction
    body = b"".join(m + b" " + n + b"\0" + d for n, m, d in entries)
    return sha256(b"tree " + str(len(body)).encode() + b"\0" + body).digest()
```

Three details that are not incidental:

**`os.scandir` is given `bytes` and yields `bytes` names.** `pathlib.Path`
**rejects bytes paths outright**, and mixing the two is how the excluded-name
comparison silently becomes `False`. The whole walk is `os`-level for that
reason.

**A device node, FIFO or socket raises rather than being skipped.** Skipping
would make two different trees hash the same, which is the one failure a digest
must not have.

**The length prefix is git's, and it is load-bearing.** Without it, two
different entry lists can concatenate to the same body.

### 4.3 Sort order: plain byte order, and it is enforced on read

Two internally consistent orders exist and there is no middle ground.

**Git sorts as if a directory name had a trailing `/`** —
`tree.c::base_name_compare`, the two lines `if (!c1 && S_ISDIR(mode1)) c1 = '/';`
Measured: for entries `a` (a directory), `a-b`, `a.txt`, git orders
`['a-b', 'a.txt', 'a', …]` where plain `sorted()` gives `['a', 'a-b', 'a.txt']`.

**NAR uses plain name order**, and that is what this design takes. The reason is
not aesthetic: plain byte order over `os.fsencode(name)` depends on **no
platform, no locale, and no directory-vs-file distinction**, so a second
implementation in another language reproduces it from one sentence. Git's rule
would buy interoperability with git's tree objects, and we cannot have that
anyway — **we record empty directories and git cannot represent them** (§4.5).

Two hazards the choice must survive, both measured:

**Sort the encoded bytes, never the `str`.** A file named `b"\x80name"`
surrogate-escapes to U+DC80 — a *high* codepoint — so `sorted(os.listdir(...))`
places it **after** `éname` while byte order places it before. They agree for all
valid UTF-8, which is why the bug would survive every ordinary test.

**Locale collation is a trap for the shell, not for Python.** `LC_ALL=C ls`
gives `A Z a a-b` where `en_US.UTF-8` gives `a A a-b`; Python's `sorted()` is
codepoint-based and unaffected by `setlocale`. Only `locale.strxfrm`, `sort(1)`
and shell globs follow the locale — so no tooling around this module may compute
an order with a shell pipeline.

**The order is enforced on read, not merely produced on write.** Both prior
implementations do exactly this — git's `fsck.c::verify_ordered` and Nix's
`archive.cc`, which raises `badArchive("NAR directory is not sorted")`. Ours is
cheaper: the walk re-derives the order from the filesystem every time, so a
verifier that computed a different order gets a different digest and says so.
What is *stated* is the rule; what is *tested* is that a second implementation
agrees (§12, criterion 6).

### 4.4 What the digest ignores, and why each

| Ignored | Justification |
|---|---|
| **mtime, atime, ctime** | **Copy-method-dependent, measured**: `cp -r` resets mtime, `cp -a` preserves it. Including it would make criterion 6 depend on which flag an agent used. Nix's manual gives the same reason for not using tar: it *"store[s] more information than we have in our notion of FSOs, such as time stamps"* |
| **owner, group** | Never survives a cross-machine handoff, which is the whole point of §7 of the spec |
| **every permission bit except x** | **Measured**: under `umask 077`, `cp -r` turns 0644 into **0600** and 0755 into **0700** — but the **executable bit survives all six copy × umask combinations**. git and Nix converged on the same whitelist independently; git's `fsck.c` carries the comment *"early on when we honored the full set of mode bits"* |
| **symlink targets are hashed, never followed** | NAR treats a symlink as a first-class node type; Bazel's `declare_symlink` documents *"Bazel will never dereference this symlink"*. Following would also make a dangling link an error at digest time, which is the wrong phase |
| **line endings** | **git is the sole outlier** and pays with `.gitattributes`, `merge.renormalize`, and a binary-detection heuristic (`convert.c::convert_is_binary`) that misclassified `ff fe 0d 0a ab` as text and deleted a byte from it in testing. Our content includes logs and binaries |
| **empty directories are kept**, unlike git | §4.5 |
| **the version directory's siblings** | §4.1 |

**Do not route the digest through tar.** Measured: a bare `touch` changes the
tar digest, and five flags are needed to suppress the variation. OCI's own spec
concedes that layers *"SHOULD be packed and unpacked reproducibly… for example
by using tar-split"*.

### 4.5 Empty directories are recorded

Measured: `cp -r`, `cp -a`, `shutil.copytree`, tar and zip **all** preserve an
empty directory; **only git cannot represent one**. A handoff whose `items`
declares a `logs` directory that a run legitimately left empty is a different
artefact from one with no `logs` at all, and the storage layer keeps that
distinction faithfully. Adopting git's tree format wholesale would silently
discard it.

This is the concrete reason §4.3 does not owe git compatibility.

### 4.6 Canonicalising `items.json`

The values in `items.json` come from two places — a jsonnet-rendered kind
default, and an agent writing at runtime — and they must digest identically
either way.

**Serialise the parsed value, never rendered text.** Measured: `_jsonnet` 0.22.0
emits `0.10000000000000001` where `rjsonnet` 0.5.6 emits `0.1`; `1e-7` becomes
`9.9999999999999995e-08` versus `0.0000001`; indentation is 3 versus 4 spaces
and rjsonnet omits the trailing newline. **Digesting rendered text would bind
the digest to the backend** — and [`../../docs/design.md`](../../docs/design.md)
§12 O2 records that we may need to switch backends for aarch64. They agree after
`json.loads`, so the parse step is exactly what makes them interchangeable.

**Refuse, do not coerce.** This is the one clearly directional finding. On a
single value our own toolchain produces — `12345678901234567168` — three
libraries do three different things: `rfc8785` raises `IntegerDomainError`, `jcs`
**silently rounds to `…567000`**, and `canonicaljson` passes it through. Silent
rounding yields a correct-looking digest for the wrong value, which is worse
than any error.

So `digest.py` canonicalises with an explicit encoder that **rejects** rather
than adjusts:

| Input | Result |
|---|---|
| `int` outside `[-(2^53)+1, (2^53)-1]` | `Malformed` — RFC 8785 Appendix D: *"numbers that do not have a natural place in the current JSON ecosystem MUST be wrapped using the JSON string type"*, on RFC 7493 §2.2's range |
| `float` that is NaN or ±Inf | `Malformed` — RFC 8785 §3.2.2.3 MUSTs an error |
| `-0.0` | `Malformed`. Measured: jsonnet renders it as `-0`, `json.loads` returns `-0.0`, and `-0.0 == 0.0` is true while the serialisations differ |
| everything else | RFC 8785 (JCS) via `rfc8785` |

**`json.dumps(sort_keys=True)` is not a canonicalisation** and the design must
not use it. Measured: NFC `"\u00e9"` and NFD `"e\u0301"` are distinct keys, and
their relative order **changes with `ensure_ascii`** — `{"é": 4, "é": 3}` under
UTF-8 output, the reverse under ASCII escaping. RFC 8785 sorts by **UTF-16 code
units** (§3.2.3), which Python's `sorted()` does not do for non-BMP keys; the
`weird.json` test vector exists to catch exactly this.

**JCS explicitly refuses to normalise Unicode** — §3.1: *"all components
involved in a scheme depending on JCS MUST preserve Unicode string data 'as
is'"*. So NFC and NFD remain two distinct keys, and that is the specified
behaviour rather than an oversight. §15 O4 records the consequence.

Two facts inherited from the pipeline, worth stating because they remove traps
rather than create them: jsonnet **rejects duplicate keys statically**
(`STATIC ERROR: duplicate field: a`) where `json.loads` silently keeps the last;
and jsonnet **cannot produce NaN or Inf**, so that row of the table is
unreachable from a kind default and reachable only from an agent.

### 4.7 The digest is a map, and it is namespaced

```yaml
# manifest.yaml
digest:
  sha256: "e3b0c44298fc1c149afbf4c8996fb924…"
algorithm: "agent_sys.handoff.tree.v1"
```

**A map, not a string** — in-toto: *"multiple entries MAY be used for algorithm
agility"*, matching if *any* field matches. It is the only cheap migration path
and it costs one nesting level today.

**Namespaced from day one.** W&B prefixes its manifest digest
(`b"wandb-artifact-manifest-v1\n"`); DVC did not, and its md5 migration cost a
permanent second cache, a per-file `hash:` discriminator, and a
`dvc cache migrate` command (#4658 → PRs #9517/#9538, regression #9580). The
name `agent_sys.handoff.tree.v1` is what §4.2's rule is registered as, and a
change to that rule is a `v2`, never a silent redefinition.

The operational rule behind both: **if a digest is recorded durably, the rule
and its enforcement land in the same change.** Matrix's `canonicaljson` does not
implement its own specification — it emits `1.0` and `-0.0`, a defect its own
maintainers call a footgun (matrix-org/python-canonicaljson#22, open since 2019)
— and fixing it required a **new room version** (MSC2540, synapse#7381) because
non-conforming digests were already persisted and enforcing the rule would have
caused a *"split brain room"*.

### 4.8 What the digest is for, and what it is not

Spec §3.3 already says it: it detects accidental corruption and casual
tampering, and answers "is this the same artefact I validated?". It is **not a
security boundary** — an agent that can write a handoff can write its digest.

Two consequences the design must honour rather than quietly improve on:

**No signature, and no signing hook.** DSSE removed canonicalisation from
TUF/in-toto as *"an unnecessarily large attack surface"*, and JWS avoided it
entirely (RFC 7515 §5.1) — but both are reasoning about **signature
verification over adversarial input**, which is not our threat model. Importing
their conclusion would over-fit. If a verdict is ever signed, DSSE's
pre-authentication-encoding shape is the model, and that is a future document's
problem.

**And the counter-lesson: an unenforced integrity sidecar is worse than none.**
PEP 815 is removing `RECORD.jws`, and its Motivation is the general form —
*"neither pip nor uv validate the hashes in `RECORD`… potentially resulting in
user confusion"*. So the digest is checked on every consumption (§5.2), or it
should not exist. §4.9 makes that affordable.

### 4.9 Recomputation is cheap enough not to cache

Measured (`probe_digest_cost.py`): **1000 files / 4 MB in 17 ms**; 210 MB at
roughly 1000 MB/s, which is I/O-bound. Stable across
`copytree(symlinks=True)`.

So `verify()` runs on every `copy_out` (§5.2) and there is no digest cache to
invalidate — which removes a whole class of staleness bug for a cost that does
not appear in a profile.

---

## 5. The consumption protocol

### 5.1 `copy_out(dst)`, with `dst` mandatory

Spec §6.3: an agent copies a handoff into its playground and works on the copy;
it never edits the stored artefact and never depends on the storage path.

```python
class HandoffStore(Protocol):
    def copy_out(self, hid: HandoffId, version: int, dst: Path) -> Content:
        """Copy this version's content/ into `dst` and return a Content over
           the copy. Verifies the digest before returning. `dst` must not
           exist; it is created."""
```

**`dst` is mandatory and there is no `get_local_path`.** MLflow's
`LocalArtifactRepository.download_artifacts(dst_path=None)` returns **the
store's own path, not a copy** (`local_artifact_repo.py:230-247`) — an agent
handed that return value is editing the store in place, and nothing in the type
signature says so. Making the parameter mandatory is a one-word decision that
makes the failure unrepresentable.

### 5.2 The copy mode is named, not left to the caller

Measured, over a tree with a 0755 script, an empty directory, and both relative
and absolute symlinks:

| Method | Tree identical? |
|---|---|
| `shutil.copytree(symlinks=True)` | yes |
| **`shutil.copytree(symlinks=False)` — the default** | **no** — symlinks dereferenced into regular files, mode 0644/0664 |
| `cp -r` | yes |
| `cp -a` | yes |

The default dereferences. Since §4.2 hashes a symlink over its target text, an
agent that copied with `shutil.copytree`'s defaults would produce a tree with a
**different digest** and no obvious cause. So `copy_out` uses
`copytree(symlinks=True)` internally, and the protocol does not expose a copy
mode for a caller to get wrong.

`copy_out` then **verifies before returning**: recompute §4.2 over the copy,
compare to `manifest.yaml`, raise `DigestMismatch` naming both digests and the
version directory. This is criterion 6, executed on every consumption rather
than in a test.

### 5.3 Writing a new version

The producing agent works in its playground and hands the store a directory:

```python
def put(self, hid: HandoffId, content_dir: Path, *, producer: TaskId) -> int:
    """Publish content_dir as the next version. Returns the version number.
       Runs the README check (§9) and the locality check (§10) first; a
       failure raises Malformed and publishes nothing."""
```

**The two admission checks run before publication, not after.** A malformed
handoff that reached storage would then need retracting, and §15 O3 records that
nobody has solved retraction. Refusing at the door is the cheap half of a
problem whose expensive half is unsolved everywhere.

---

## 6. Storage

### 6.1 Ten operations, and the seven leaks to design against

Spec §6.1 fixes v1 as a filesystem and asks for a clear interface. The risk is
not that a filesystem is wrong; it is that **the filesystem becomes the
interface** and a second backend is then impossible. Prior art names exactly
where that happens:

```python
class HandoffStore(Protocol):
    def list_versions(self, hid: HandoffId) -> list[int]: ...
    def get_manifest(self, hid: HandoffId, version: int) -> Manifest: ...
    def open_item(self, hid, version, key: str) -> BinaryIO: ...
    def copy_out(self, hid, version, dst: Path) -> Content: ...        # §5.1
    def put(self, hid, content_dir: Path, *, producer: TaskId) -> int: # §5.3
    def exists(self, hid: HandoffId, version: int | None = None) -> bool: ...
    def record_verdict(self, hid, version, verdict: Verdict) -> None: ...  # §4.1
    def read_verdicts(self, hid, version) -> list[Verdict]: ...
```

| Leak | Where prior art hit it | What this interface does |
|---|---|---|
| **Atomic directory rename** | Arrow's S3 refuses outright: *"we don't implement moving directories as it would be too expensive"* (`s3fs.cc:3399`). Hadoop S3A: *"callers cannot safely rely on atomic renames as part of a commit algorithm"* | **`put` is the commit token, not `rename`.** §6.3. If rename *were* the interface, an object store would have nothing to implement |
| **Directory listing vs prefix listing** | MLflow's base `_is_directory` is `len(list_artifacts(path)) > 0` — "a prefix with children" — and `LocalArtifactRepository` must **override** it because an empty POSIX directory lists empty. The leak runs *toward* the object store | No `list_items`; the manifest enumerates. A store never has to answer "is this a directory" |
| **Empty directories** | S3 fakes them with zero-byte markers (`s3fs.cc:2368`); Arrow carries a `have_implicit_directories` test predicate | The manifest records them (§4.5), so a backend recreates rather than discovers them |
| **Append** | *"It is not possible to append efficiently to S3 objects"* (`s3fs.cc:3469`) | No append. A version is written once |
| **`stat`, mtime, permissions** | Arrow's `FileInfo` is 4 fields with no symlink; S3A reports directory permissions as 777 and files as 666 | Not in the interface. §4.4 already excludes all of it from the digest |
| **Locking** | Nothing surveyed has one. MLflow *has* `ExclusiveFileLock` and does not use it where the race is — and it raises on Windows | §6.3's allocator needs no lock |
| **Delete** | Six MLflow backends refuse it in **two different exception types** — `NotImplementedError` in four files, `MlflowException("Not implemented yet")` in `dbfs_artifact_repo.py:185` | **Not in the Protocol at all.** §15 O3 |

**`Verdict` is this module's type, and it is the one that crosses the seam.**
Both this design and `validator` design §2 list a type by that name, which the
stage-three consistency pass found and which would have been two records of one
fact. The split:

| | Owner | What it is |
|---|---|---|
| `Verdict` | **`handoff/verdict.py`** | The persisted record: validator name, result, strength, dimension, task, agent, environment, timestamp. What `record_verdict` writes and `read_verdicts` returns |
| `VerdictRecord` | `validator/history.py` | `validator` design §7.1's *view* of one `Verdict` — what `top()` returns and `may_skip()` reads |

`validator` does not persist and this module does not decide; the storage layer
owns the shape because it is the layer that has to keep it readable across
versions. `validator/protocol.py` re-exports the name rather than declaring a
second one.

**Capability goes in mixins and a conformance suite, never a flags dict.**
MLflow pushed backend-specific operations into four ABCs
(`MultipartUploadMixin`, `MultipartDownloadMixin`, …, `artifact_repo.py:629+`);
Arrow put 13 deviation predicates in its **test** header
(`test_util.h:164-194` — `allow_move_dir`, `have_implicit_directories`) and has
**no `supports_*` field at runtime**. This is
[`../../docs/design.md`](../../docs/design.md)'s "backends raise; no capability
matrix" arriving from a second direction, and it means `tests/handoff/` ships a
`StoreConformance` suite that any backend runs against itself.

**`fsspec` is not the interface to copy**, despite being the obvious candidate:
68 public callables where roughly 6 must be written. Its own author, on Arrow's
founding PR (apache/arrow#4225), called the API *"incomplete … and contains a
lot that is unnecessary"*; Arrow built 15 pure virtuals instead.

### 6.2 The layout, and one symbolic accessor

```
<root>/<hid>/v<N>/{content/,validation.yaml,manifest.yaml}
```

`env_mgr` spec §4 owns the prefix/substring/suffix conventions that make a
permission decision a string match. This design's contribution is one rule:

**Exactly one function computes a path, and the on-disk shape is private.**

```python
def version_dir(root: Path, hid: HandoffId, version: int) -> Path: ...
```

Bazel #23576 is the lesson: a path-shape change survived only because consumers
use `file.path` and `$(location)` rather than composing strings. Every other
module asks for a path; none builds one.

### 6.3 Publishing a version atomically

The second place where the ordering is the design:

```python
def put(self, hid, content_dir, *, producer):
    readme.check(content_dir, kind)            # §9  — raises before anything is created
    locality.check(content_dir, kind)          # §10 — likewise
    d = digest.tree_digest(os.fsencode(content_dir / "content"))

    base = root / str(hid)
    for n in itertools.count(self._next_guess(base)):
        stage = base / f".staging-v{n}"        # a SIBLING of the destination
        try:
            os.mkdir(stage)                    # atomic allocator: FileExistsError
        except FileExistsError:
            continue
        try:
            shutil.copytree(content_dir, stage / "content", symlinks=True)
            write_manifest(stage / "manifest.yaml", digest=d, producer=producer)
            (stage / "validation.yaml").write_text("verdicts: []\n")   # §6.4
            os.rename(stage, base / f"v{n}")   # ENOTEMPTY if v{n} exists
            return n
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
```

Four measured facts hold this up:

**The staging directory is a sibling of the destination.** Measured: `/tmp` and
the working tree are **different filesystems on this machine** (`st_dev`
differs), so a `/tmp` stage cannot be renamed into the store — the rename would
fall back to a copy or fail. This is not hypothetical: `fsspec` calls its own
transaction *"semi-atomic"* (`fsspec/transaction.py:5-10`) because
`LocalFileOpener._open` calls `mkstemp()` with **no `dir=`**; a 200 MB write with
target on tmpfs and temp on ext4 showed **185 of 248 polls seeing a partial
size**, and 0 on one filesystem. Atomicity decided by mount layout, silently.

**`os.mkdir` is a free atomic allocator.** MLflow's FileStore is read-max+1
(`file_store.py:681-684`) whose guard calls `write_yaml` with `overwrite=False`
— and `write_yaml` (`file_utils.py:854-875`) **ignores its own `overwrite`
argument**, so two writers both allocate v3 and the second wins. `FileExistsError`
costs nothing and cannot be ignored.

**`rename` onto a non-empty directory fails `ENOTEMPTY`.** Measured. Since
versions are never overwritten this never fires — and if it ever did, it fires
loudly rather than clobbering.

**MLflow gets the single-file case right and says why** —
`local_artifact_repo.py:125-151` stages with `mkstemp(dir=artifact_dir)` then
`os.replace`, commented *"so readers never see a partially-written artifact"* —
**and does not do it for directories**: `log_artifacts` (line 226) is a plain
copytree into the live directory. We do for directories what they do for files.

`.staging-` is a reserved prefix and the lister filters it, as MLflow's does
(`local_artifact_repo.py:262`).

### 6.4 Versions are integers; the digest is the identity

Monotonic integers as the public name, the content digest as the manifest's
identity — W&B's shape. Integers give free ordering and a directory name a human
can read; the digest gives replay comparison, which is what "is this the same
artefact I validated?" needs.

`validation.yaml` is created empty at publication rather than on first verdict,
because **absent must be stated, not omitted**. SVR requires `verifier.policies`
*"even when no policy can be referenced; in that case, the value MUST be an
empty array"*, and Nix's source carries the same rule in a comment: *"Mandatory
check: absent whitelist, and present but empty whitelist mean very different
things."* An empty `verdicts:` list says "nothing has checked this yet"; a
missing file says "something is wrong with this version".

### 6.5 Two instances, one implementation

Spec §6.2 requires a separate storage instance for knowledge handoffs. The
design's contribution is to say what "separate" means: **two `FilesystemStore`
objects with different roots, registered under different names**, and nothing
else different.

```python
r.register("handoff_store",   FilesystemStore(cfg.handoff_root))
r.register("knowledge_store", FilesystemStore(cfg.knowledge_root))
```

The differences spec §6.2 lists — outlives every run, not scoped to a task's
subtree, broadly readable — are all **lifecycle and permission**, and permission
is `env_mgr`'s (§7). None of them is a behaviour of the store. So there is no
`KnowledgeStore` subclass, and the day one is needed is the day the interface
was wrong.

The knowledge instance's *validators* are restricted (spec §4.1: schema
conformance and internal consistency only). That restriction lives in the
**kind**, not the store — a knowledge kind declares validators of those
dimensions, and §8 binds them like any others.

---

## 7. Permission is containment

Spec §6.1 makes reach a matter of containment, and `env_mgr` spec §4 owns the
naming scheme and the OS enforcement. What this module owns is **the check that
runs in-process**, and the honest statement of what it is worth.

### 7.1 Three different properties, routinely conflated

| | Cost | What it proves |
|---|---|---|
| **Lexical containment** | no I/O | The path *string* is under the zone |
| **Filesystem containment at an instant** | `realpath` | It was under the zone when measured. Immediately stale |
| **Enforced containment** | a kernel feature | It stays under the zone against a concurrent or out-of-process actor |

A validated path *string* can only ever deliver the second. The design says so
rather than implying the first is the third.

### 7.2 The naive checks are both wrong, measured

Zone `…/a/b`, candidate `…/a/bc/x`:

| Check | Verdict |
|---|---|
| `str.startswith(zone)` | **True — wrong.** The `/a/b`-matches-`/a/bc` bug |
| `str.startswith(zone + os.sep)` | False — correct |
| `Path.is_relative_to(zone)` | False — correct |
| `realpath` + separator | False — correct |

But for `…/a/b/../bc/x`, `is_relative_to` says **True** — it is purely lexical
and does not collapse `..`; and for an **in-zone symlink pointing out**,
`is_relative_to` says True while `realpath` says False.

The first row is live: three CVEs in 2026 — CVE-2026-5422 (jupyter-server,
`startswith(root)` with no trailing separator), CVE-2026-40256 (Weblate, fix
`e30dbcb`), CVE-2026-48544 (Taipy), whose fix commit `129fd40` is titled *"is
relative to is better than starts with"* and is one line. And the second row is
why copying that one-line fix is not enough: the Taipy fix works only because
`.resolve()` ran the line above.

So:

```python
def check_contained(candidate: Path, zone: Path) -> None:
    """Raise NotContained unless `candidate` resolves inside `zone`.
       Rejects '..' by policy before resolving, so a rejected path is
       reported as written rather than as resolved."""
```

`os.path.commonpath` is **worse** than `is_relative_to` here: it raises
`ValueError` on mixed absolute and relative inputs and still needs a
caller-side comparison that can itself be wrong.

`os.RESOLVE_BENEATH` is **not exposed by CPython 3.13** (measured), so
`openat2` from Python means `ctypes`. That is `env_mgr`'s decision, not this
module's.

### 7.3 The five assumptions the string check rests on

Stated because the design depends on **all five**, and a reader who cannot see
them will reasonably conclude the check is security theatre.

**(A) The threat is mistakes, not attackers.** Kustomize says so verbatim — its
containment check protects *"the person inclined to download kustomization
directories from the web and use them without inspection"* — and ships
`--load-restrictor LoadRestrictionsNone` to turn it off. That is good precedent
for "a guardrail with a stated threat model", not for "a string check is
secure".

**(B) A kernel layer is the actual boundary.** The runc/Kubernetes shape:
`subpath_linux.go::doSafeOpen` runs the string check `mount.PathWithinBase`
*inside* a loop whose per-component `openat(O_NOFOLLOW|O_PATH)` file descriptors
are the real guarantee. `filepath-securejoin` states why the string form cannot
be fixed, verbatim in `join.go`: *"There is no way to solve this problem with
SecureJoinVFS because the API is fundamentally wrong (you cannot return a 'safe'
path string and guarantee it won't be modified afterwards)."* Hence
`OpenInRoot` returns a file descriptor.

The string check still buys three things the kernel layer cannot: an early
**attributable** error naming the offending task; a check on paths that **do not
exist yet**, which `openat2` cannot do; and coverage before any syscall.

**No prior art was found pairing a string precheck with Landlock specifically.**
The pattern generalises from openat2 and file descriptors; the Landlock pairing
is ours. Recorded so nobody later assumes it was copied.

**(C) `..` is rejected by policy**, because §7.2 measured that
`is_relative_to` does not collapse it.

**(D) The name space is a mint-time allow-list that cannot express an escape.**
Nix's `checkName` permits only `[0-9a-zA-Z+\-._?=]` and separately rejects `.-`
and `..-` **because `-` is its field separator**. Validate when a name is
created, not when it is used. This is the load-bearing assumption for
`env_mgr`'s structured-prefix scheme, and it belongs here because this module
mints the names.

Two facts that constrain it: `/` is the **only** byte POSIX forbids in a
filename — newline, tab, space, `:`, `~`, `-` and a leading `-` are all legal —
so a separator is excluded by allow-list or not at all. And `NAME_MAX` is **255
bytes, not characters**: 128 `é` characters is `ENAMETOOLONG`. Nix's 211-byte
limit is a *derived budget*, `255 - 32 - 1 - 4 - 7`, pre-reserving room for
every decoration it might later append — which is the discipline, not the
number.

**(E) Symlink creation inside the tree is prevented, not detected** — by
withholding `LANDLOCK_ACCESS_FS_MAKE_SYM`, which is `env_mgr`'s to grant. Helm
is the counter-case: CVE-2025-53547's advisory reads *"Helm warns of the
symlinked file but did not stop execution."* A warning an agent reads and
proceeds past is not a rule.

Arrow states the same limit about its own `SubTreeFileSystem`: *"This makes no
security guarantee. For example, symlinks may allow to 'escape' the subtree"*
(`filesystem.h:418`).

---

## 8. The validator binding

### 8.1 Many-to-many, recorded on both sides

Spec §5.1. The kind names its validators; a validator names its kinds. The
registry answers both directions, so `HandoffSpecRegistry` carries a reverse
index built at admission:

```python
class HandoffSpecRegistry(SpecRegistry):
    def validators_for(self, kind: str) -> list[str]: ...
    def kinds_for(self, validator: str) -> list[str]:      # the reverse index
        """Which kinds name this validator. Built once at admission, not
           searched on each call."""
```

The reverse index is built rather than searched because spec §8 requires it
("through a reverse index — which kinds a given validator covers") and because
§8.3's check needs it once per validator, not once per query.

### 8.2 The agreement check runs in the closure pass

Not at admission. `HandoffSpecRegistry.check` cannot verify that
`check_trace_shape` binds back to `trace`, because the validator registry may
not be loaded yet. So spec §8's load-time checks 2, 3 and 4 are **contributed
to the closure pass** described in
[`../../docs/design.md`](../../docs/design.md) §6, and the layering gate applies:
a kind that failed its own schema is skipped, because "your validator does not
resolve" on top of "your schema is broken" is noise.

### 8.3 What a mismatch looks like

The third place where ordering is the design — the check itself is four lines,
and the report is the work:

```python
for kind in handoffs.names():
    for vname in handoffs.validators_for(kind):
        if vname not in validators:
            raise SpecNotFound(...)                      # with candidates
        if kind not in validators.get(vname)["inputs"]:
            raise BindingConflict(kind, vname, ...)      # both origins
```

**The field is `inputs`, and rev. 1 called it `binds_to`.** There is no
`binds_to` anywhere: `ValidatorSpec` (`validator` design §3.2) declares
`inputs: tuple[str, ...]`, and that model is `extra="forbid"`, so a spec carrying
`binds_to` is rejected at admission. Rev. 1 of this document and `validator`
design §10.3 check 4 had **two names for one field**, and the two-way agreement
check — the thing criterion 10 is about — read the one that cannot exist. Found
in the stage-three consistency pass; corrected on both sides.

The error message below still says `binds_to:` in its example because that is the
*label a reader sees*, and `inputs:` would be ambiguous next to the handoff's own
input list. The label is prose; the key is `inputs`.

Seven fields, from OPA PR #7808's before-and-after, Django E304, rustc E0119 and
pluggy:

```
BindingConflict: handoff kind 'trace' and validator 'check_trace_shape' disagree
  handoff/trace.jsonnet:12               validators: [check_trace_shape, check_trace_cov]
  validators/check_trace_shape.jsonnet:8 binds_to:   [trace_v2]
  differing: 'check_trace_shape' binds to 'trace_v2', and 'trace' is not in that list
  fix: either add 'trace' to check_trace_shape's binds_to, or remove
       'check_trace_shape' from trace's validators. Both are valid repairs.
  hint: 'trace_v2' also exists — one of the two was renamed and the other not.
```

Both sides named · **both file paths, with line numbers** · the **specific
differing element**, not "they differ" · a fix naming both as repair points ·
deterministic ordering (OPA sorts twice — the bundle set *and* the paths within
each pair) · enumerated candidates when the fault is a bad name · a hint where
one is plausible.

The bad version is worth recording because it is what symmetric blame alone
produces — OPA's pre-fix message, symmetric and still useless:
`detected overlapping roots in bundle manifest with: [bundle1.tar.gz bundle2.tar.gz]`.

### 8.4 Addressing into content: JSON Pointer

Spec §5.1 says "a jsonpath into the content". **This design uses RFC 6901 JSON
Pointer instead**, and §14 D1 records the deviation. The reason is a
specification, not a preference:

**RFC 9535 §2.5.1.2 forbids a valid JSONPath query from erroring**: *"A
syntactically valid segment MUST NOT produce errors when executing the query.
This means that some operations that might be considered erroneous, such as
using an index lying outside the range of an array, simply result in fewer nodes
being selected."* And §2.1.1: *"An empty nodelist is a valid query result."*

So **no JSONPath implementation can distinguish "the path is wrong" from "the
value is absent"** — the caller receives `[]` either way. That is precisely the
silent pass this system exists to prevent, and it is a property of the standard
rather than of any library.

Measured, over the four candidate libraries:

| | bad syntax | matched nothing | value is `null` |
|---|---|---|---|
| `jsonpath-ng` | `JsonPathParserError` | `[]` | `[None]` — distinguishable only by list length |
| `jsonpath-ng`, **validly parsed** | — | **19 leaked stdlib exceptions** — `IndexError`, `KeyError`, `TypeError`, `ValueError` | — |
| `python-jsonpath` | `JSONPathSyntaxError`, with a caret rendering | `findall`→`[]`, `match`→`None` | a match object |
| **`python-jsonpath`'s `JSONPointer`** | `JSONPointerError` at **construction** | `JSONPointerKeyError` / `JSONPointerIndexError` at **resolve** | `UNDEFINED` sentinel |
| `jsonpointer` 3.1.1 (installed) | `JsonPointerException` | `JsonPointerException` — **same class, conflated** | — |
| `referencing` (installed) | `InvalidAnchor` | `PointerToNowhere` | leaks bare `ValueError` on `/lst/x`; **percent-decodes** |

`jsonpath-ng`'s leak is disqualifying on its own: an `except JSONPathError`
handler does not hold, and issue #203 reproduces it — `$[-2]` on `["text"]`
raises a bare `IndexError`. Only `python-jsonpath`'s pointer separates all three
outcomes.

```python
def resolve(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer. Raises PointerInvalid for a malformed
       pointer and PointerMiss for one that addresses nothing. A JSON null
       is returned as None and is not a miss."""
```

**Two failures, two exception types**, for the same reason
[`../../docs/design.md`](../../docs/design.md) §6.2 separates `SpecNotFound`
from `SpecInconsistent`: a malformed pointer is an author's typo, while a miss
means the content is not what the binding expected — and a validator must be
able to treat them differently.

What we lose: wildcards, recursive descent and filters. If a validator ever
needs "every element of `items.operators`", Pointer cannot express it. Choosing
`python-jsonpath` — one zero-runtime-dependency wheel carrying Pointer,
JSONPath **and** JSON Patch — makes that upgrade an import rather than a
dependency negotiation. Its maintenance record is the other half of the choice:
1 open issue against `jsonpath-ng`'s 86, and it runs the official **JSONPath
Compliance Test Suite as a git submodule in CI**. cburgmer's comparison
quantifies the gap: divergent queries **142 (51 failing consensus)** for
`jsonpath-ng` against **92 (5 failing)** for `python-jsonpath`.

Pointer is also unambiguous where JSONPath is not — RFC 9535 §2.5.1.1 is
explicit that *"`$.foo.bar` is shorthand for `$['foo']['bar']` (**but not for
`$['foo.bar']`**)"* — and its `~0`/`~1` escaping addresses **any** key,
including one containing `/` or `.`. §3.3 makes that concrete: our `items` keys
can be agent-generated.

### 8.5 The escape hatch is a return value, not a log line

Spec §5.3: a kind with no validator is rejected unless a flag permits it, and
every kind the flag lets through is **reported** by name at startup and in the
run record.

```python
@dataclass(frozen=True)
class LoadReport:
    admitted: list[str]
    without_validator: list[str]      # sorted. Empty is the normal case
```

A list on the report object rather than a `logging.warning`, because criterion
12 asserts the name appears in the startup report **and** the run record, and an
assertion over a log capture is a test of the logging configuration. The flag
does not disable existing validators; it only permits absent ones.

---

## 9. The README check

Criterion 2: a handoff version with no README, or missing a section its content
type requires, is malformed.

### 9.1 Existence, non-empty, and no placeholder — all three

**A presence-only check on an agent-authored artefact is theatre by
construction.** An LLM told "your README must have sections X, Y, Z" emits
exactly X, Y and Z with placeholder bodies. That is not speculation; it is
measured at scale elsewhere:

- Hugging Face's live `POST /api/validate-yaml` returns **HTTP 200 `OK`** for a
  card whose entire prose is `[More Information Needed]` — a string its own
  template emits **39 times**, and which HF full-text search finds in
  **636,321** repositories.
- arXiv 2402.05160, over 32K model cards: only **44.2%** of repositories have a
  card at all; Environmental Impact **2.0%**, Citation 14.4%, Evaluation 15.4%,
  Limitations 17.4% — and **84.8% of the Environmental Impact sections were
  machine-generated**.
- Kubernetes' KEP linter (`hack/verify-toc-vs-template.sh`, PR #5267) does
  exactly our check with a real AST, then ends
  `# TODO(soltysh): for now this should not fail` / `exit 0`.

So three layers, and the check is `Malformed` at any of them:

```python
def check(content_dir: Path, kind: HandoffKind) -> None:
    """Raise Malformed unless every required section exists at document root,
       has non-empty inline text, and is not a placeholder."""
```

| Layer | Rule |
|---|---|
| **exists** | The section heading is present **at document root** |
| **non-empty** | The extracted **inline text** between this heading and the next is non-blank |
| **not a placeholder** | The text is not in the reject list, which is **generated from our own templates** |

The reject list is generated rather than hand-maintained because a hand-written
one drifts away from the templates that produce the strings it is meant to
catch.

### 9.2 What "the section exists" means, exactly

Measured against a naive `^#{1,6}\s+(.+)$` regex versus a CommonMark parse:

| Case | naive regex | CommonMark |
|---|---|---|
| `## Results` | found | found |
| `## Results ##` | found | found |
| `Results` + `-------` (setext) | **missed** | found |
| ≤3 spaces + `## Results` | **missed** | found — ≤3 spaces is still a heading |
| 4 spaces + `## Results` | missed | not a heading — correct, it is a code block |
| `## Results` inside a fence | **false positive** | correctly not a heading |
| `##Results` | missed | not a heading — correct |

**The regex is wrong in both directions**, and the false positives are the
security-relevant half: a producer satisfies the anti-blob check with headings
that are **invisible in the rendered README**.

So: parse to a CommonMark AST with `markdown-it-py`, walk `heading_open` tokens,
slice section bodies between heading boundaries. Four details:

**Strip YAML front matter first** — `---` is a setext underline, and mdtoc
carries a `stripFrontMatter` for exactly this.

**Require the heading's parent to be the document root.** markdownlint's MD043
filter is flat, so a heading inside a blockquote or a list item satisfies it.
No surveyed tool implements this refinement.

**Test the extracted inline text, not the token count.** An HTML-comment-only
body yields one token and empty text, and `&nbsp;` survives as literal text —
so entity-decode before the whitespace test.

**Set membership, not sequence matching.** markdownlint #394 is a wildcard
required-headings matcher that **silently accepted every document** — it failed
*open*. And MD043's global ordered-list model provably cannot express per-kind
requirements (#32), which is our four-content-types-plus-`readme_sections`
shape exactly. §3.2 stores the sections ordered for templating and checks them
as a set.

### 9.3 No autofix, ever

Ruff #23562 is the precedent: line-by-line section matching misfired inside a
`.. code-block:: yaml` and **`--fix` rewrote the user's docstring**. This module
raises and names the section; it never edits a README.

---

## 10. Locality independence — what criterion 17 can be

Criterion 17: a handoff whose content declares an absolute local path fails its
locality-independence check. This is the section where the honest answer is
narrower than the criterion sounds.

### 10.1 Nobody detects locality dependence by a path's shape

Every working system either **matches a prefix supplied by an oracle** —
lintian's `quotemeta($buildinfo->Build-Path)`, rpmlint's
`rpm.expandMacro('%{?buildroot}')`, conda-build's `PREFIX_PLACEHOLDER`, Nix's
fixed `referenceablePaths` set — or **prevents the path existing at all**
(Bazel).

**The shape-refinement fix was proposed and refused on the record.** Debian
#1002451: lintian's `^build/` pattern matched the English word "build" in a C++
comment. The reporter proposed a tighter regex; Mattia Rizzolo rejected it —
*"that would be a sbuild-specific solution, and even in sbuild it's a
configurable parameter."* **You cannot recognise a build path by its shape,
because the shape is a property of whoever built it.**

rpmlint #1350 is the controlled experiment: rpm 4.20 stopped defining
`%{buildroot}`, a shape regex replaced the oracle, and a decade-old script
false-positived within one release.

### 10.2 Measured on this repository

A bare `(?:/[A-Za-z0-9._+-]+){2,}` over 276 files: **650 matches, of which 23
are genuinely local** (`/tmp/…`, `/home/someone/run3`) **and 627 (96%) need a
suppression rule** — 147 system paths, 401 URLs whose path component looks
absolute, 79 single-segment prose fragments. And it **misses**
`C:\Users\bob\run` entirely without a second alternation.

A ten-sample illustration of why the suppressions are not a tidy list:

| Sample | Flagged | Should be |
|---|---|---|
| `#!/usr/bin/env python3` | yes | a shebang |
| `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` | yes | a system path |
| `https://example.com/a/b` | yes | a URL |
| `/opt/rocm/bin` | yes | a vendor prefix |
| `/workspace/repo/build` | yes | container-internal, portable by construction |
| `./data/in.json` | yes | relative — matched via the `/data/in.json` substring |
| `cd /home/someone/run3` | yes | **genuinely local** |
| `out=/tmp/x/results.json` | yes | **genuinely local** |

### 10.3 What the check therefore is

An **anchored allow-list on the raw string**, delocate's shape —
`_SANITARY_RPATH = re.compile(r"^@loader_path/|^@executable_path/")`, no
filesystem access, and negatives pinned as executable tests. Plus the oracle
inputs we actually have, which is more than lintian has:

| Oracle | Source |
|---|---|
| The producing agent's **playground root** | `env_mgr`, known at production time |
| The **store root** | §6.2 |
| The declared **container image**'s internal prefixes | the kind's `dependencies` (spec §7) |

```python
def check(content_dir: Path, kind: HandoffKind, *, oracles: Oracles) -> None:
    """Raise Malformed naming the file, the line, and the offending path."""
```

An oracle hit is **certain**: a playground path in a published artefact is a
record of one machine's afternoon by construction. The shape regex runs only
where no oracle applies, and its output is subject to the allow-list.

**Two traps pre-empted.** *Self-application*: conda splits its own placeholder
literal across two source strings *"such that running this program on itself
will leave it unchanged"*, and bandit `# nosec: B108`s its own default list
twice — our checker's source, tests and error messages will contain example
absolute paths, so the reject patterns are data, never literals in the module
that scans. *Stale exclusions fail closed*: conda-build raises `RuntimeError`
when a declared `has_prefix_files` entry turns out to contain no prefix, so an
allow-list entry that stops matching is an error rather than a widening blind
spot.

### 10.4 The limits, stated

**The check is sound on oracle hits and best-effort otherwise.** Three specific
false negatives, none closed by more regex:

- **Compression.** Nix's manual documents uncompressing man pages *"to not miss
  references hidden by compression"*. A gzipped log carrying a playground path
  passes.
- **Runtime concatenation.** `os.path.join(HOME, "run3")` in a script is a
  locality dependence the checker cannot see. Nix's own scanning has the same
  gap and it is unstudied there.
- **`allowedReferences = []` asserts the scan found nothing, not that there is
  nothing to find.** Nix's exact caveat, and ours.

**Nix has considered our exact check — issue #9549, "Relocatable store
objects" — and has not implemented it**, musing that it *"could also just be
done 'in user space' with a script."* That is the closest thing to a precedent
for the whole idea, and it is an open issue.

**File role is a real axis and almost nobody uses it.** Our artefact holds both
logs and scripts, and a path in a log is weaker evidence than a path in a script
that will be re-executed. lintian is the only surveyed tool with the mechanism
(`Item.pm::mentions_in_operation`, gated on ELF-or-`#!`) and **it does not wire
it into the build-path check** — its own fixture expects a hit under
`usr/share/doc/`. This design follows lintian: **the check does not weight by
role**, because a playground path in a changelog is still a record of one
machine. §15 O5 records it as a question rather than settling it.

**The meta-lesson, and why this section is long.** Debian's detection apparatus
is *differential* — diffoscope requires two inputs and has no single-artefact
path heuristic — and once they froze the build path (commit `8c2c7fb42d5`,
*"stop variying the build path, we want reproducible builds"*), differential
detection of build-path leakage became structurally impossible. **They kept the
leak and stopped being able to see it.** The residual is still visible: `gcc
captures build path` spans **1841 packages**. A check that claims more than it
delivers is how that happens, so this one states its limits in the failure
message: `Malformed` says whether the hit was an oracle match or a heuristic.

**Every project that made a path check mandatory acquired false positives within
a release or two**: delocate #255, Bazel #26150, lintian #1002451, rpmlint
#1350, and conda-build #1409 — the last a false positive that produced a
*corrupt package*.

---

## 11. Build versus adopt

| Module | Considered | Chosen | Why |
|---|---|---|---|
| `digest` | `checksumdir`, `dirhash`, `git hash-object`, tar + sha256 | **own, ~60 lines** | §4.2. `checksumdir`'s own docstring: *"taking into account only file contents and not filenames"* — measured, two trees with contents swapped between filenames hash **identically**. `dirhash` is a sound Merkle design but **has no option for the executable bit at all**, needs `empty_dirs=True` and `is_link` set explicitly, and has an open pickling bug (#34) breaking `--jobs > 1`. tar is rejected in §4.4 |
| canonicalisation | `json.dumps(sort_keys=True)`, `canonicaljson`, `jcs`, hand-rolled | **`rfc8785`, wrapped by a rejecting encoder** | §4.6. `sort_keys` is not a canonicalisation. `canonicaljson` is a *different* specification **and does not implement it** (#22, open since 2019). `jcs` silently rounds. `rfc8785` raises, which is what we want — the wrapper adds the `-0.0` and NaN rejections it does not cover |
| `pointer` | `jsonpath-ng`, `jsonpointer`, `referencing`, `python-jsonpath` | **`python-jsonpath`'s `JSONPointer`** | §8.4. The only one of six that separates malformed, missing, and null. One zero-runtime-dependency wheel that also carries JSONPath, so §8.4's lost features are an import away |
| `readme` | a regex, `markdownlint`/MD043, `remark-lint`, `markdown-it-py` | **`markdown-it-py`** | §9.2. The regex is wrong in both directions, measured. MD043 is the only off-the-shelf required-headings rule and cannot express per-kind requirements (#32); remark-lint has 85 rules and no such rule at all. `markdown-it-py` 4.0.0 is **already present** — transitively, via `rich`. §15 O2 |
| `store` | `fsspec`, MLflow's `ArtifactRepository`, a CAS | **own Protocol, ~10 methods** | §6.1. `fsspec` is 68 public callables where ~6 must be written, and its own author called the API *"incomplete"* on apache/arrow#4225. A CAS fails v1: REAPI cannot list by prefix, cannot name, and blob lifetime is only a **SHOULD** (`remote_execution.proto:337`) — fourteen years of `Missing digest` downstream (bazel#10880, #16660, #17711, #18694, still-live #30218) |
| `verdict` | in-toto attestations, OCI referrers, a sidecar file | **a sidecar file, with in-toto's outer shape** | §4.1, §6.4. The subject/predicate split is adopted so a heterogeneous history stays queryable; the transport is not, because v1 storage is a filesystem tree |
| `locality` | diffoscope, a shape regex alone, Nix's reference scan | **an anchored allow-list plus oracles** | §10. diffoscope is differential and needs two inputs. A shape regex alone is 96% noise, measured |

**Three new runtime dependencies**: `python-jsonpath`, `markdown-it-py`,
`rfc8785`. §15 O2 records that none is declared, and that two are already
installed only as transitive dependencies of something else.

---

## 12. Test plan

`pytest`. Tests in `agent_sys/tests/handoff/`, with an `__init__.py` for the
import-mode reason [`../../task_graph/docs/design.md`](../../task_graph/docs/design.md)
§11 gives. Every test builds its own store against `tmp_path`; nothing is
process-global.

### 12.1 Spec criteria, mapped

| # | Criterion | Test | File |
|---|---|---|---|
| 1 | A spec omitting a key, or with an invalid `items_schema`, is rejected with path and key | `test_missing_key_rejected`, `test_items_schema_not_a_schema_is_one_error` | `test_kind.py` |
| 2 | No README, or a missing required section, is malformed | `test_missing_readme`, `test_missing_section`, `test_section_in_code_fence_is_not_a_section`, `test_setext_heading_counts` | `test_readme.py` |
| 3 | Each content type accepts its items and rejects an undefined one | `test_four_types_accept_and_reject` | `test_content.py` |
| 4 | `reproducible` + `script` with no `env` is rejected | `test_script_without_env_rejected_at_kind_admission` | `test_kind.py` |
| 5 | A runtime key is accepted where the schema permits, rejected where not | `test_runtime_key_follows_additional_properties` | `test_content.py` |
| 6 | The digest reproduces after storage and a copy into a playground | `test_digest_survives_round_trip`, `test_digest_matches_reference_vectors` | `test_digest.py` |
| 7 | **Recording a verdict does not change the digest** | `test_verdict_does_not_move_digest`, `test_rewriting_verdict_does_not_move_digest` | `test_verdict.py` |
| 8 | The history names, per validator, each result plus task, agent, environment | `test_history_is_complete` | `test_verdict.py` |
| 9 | A validator over two handoffs, and a handoff with three validators, both resolve | `test_binding_resolves_both_directions` | `test_binding.py` |
| 10 | **A binding conflict crashes at load, naming both sides** | `test_conflict_names_both_sides_and_paths` | `test_binding.py` |
| 11 | Lookup by uuid resolves; a pointer returns the addressed value; two inputs of one kind are unambiguous | `test_lookup_by_uuid`, `test_pointer_three_way`, `test_two_inputs_same_kind` | `test_pointer.py` |
| 12 | A kind naming no validator is rejected unless the flag is set — and then reported | `test_no_validator_rejected`, `test_flag_reports_by_name` | `test_kind.py` |
| 13 | Knowledge handoffs land in the separate storage, with restricted validators | `test_knowledge_instance_is_separate` | `test_store.py` |
| 14 | The four scope tags produce their storage, permission and retention — **asserted by where the artefact lands, not by reading the tag back** (§3.4) | `test_scope_tags_land_where_declared` | `test_store.py` |
| 15 | An `addons` handoff does not satisfy a `fixed.required` input — checked on the **binding**, at load (§3.4) | `test_addons_does_not_satisfy_required` | `test_scope.py` |
| 16 | **Permission is containment**, against the real layout | `test_own_subtree_and_subtasks_reachable`, `test_sibling_denied`, `test_dotdot_denied`, `test_prefix_sibling_denied` | `test_containment.py` |
| 17 | Content declaring an absolute local path fails | `test_oracle_hit_rejected`, `test_system_path_allowed`, `test_url_allowed` | `test_locality.py` |

Criterion 16's `test_prefix_sibling_denied` is named separately because it is the
`/a/b`-versus-`/a/bc` case, which the other three would all pass.

### 12.2 Tests beyond the criteria

Measured facts a future change could silently break:

| Test | Guards |
|---|---|
| `test_digest_matches_reference_vectors` | §4.2. A checked-in table of (tree, digest) pairs, so a refactor that changes the algorithm fails loudly rather than re-deriving a new "correct" answer. This is the only defence against the Matrix failure mode |
| `test_sort_is_byte_order_not_str_order` | §4.3. Builds a tree containing `b"\x80name"` and `"éname"` and asserts the order. Agrees with `sorted(str)` for all valid UTF-8, so nothing else would catch it |
| `test_exec_bit_in_digest_full_mode_not` | §4.4. `chmod 0600` does not move the digest; `chmod -x` does |
| `test_empty_directory_is_recorded` | §4.5 |
| `test_copy_out_refuses_to_return_store_path` | §5.1. Asserts `copy_out`'s signature has no default for `dst` — the guarantee is the signature, so the signature is what is tested |
| `test_staging_is_a_sibling` | §6.3. Asserts the staging directory shares a parent with the destination, so nobody "tidies" it into `/tmp` |
| `test_concurrent_put_allocates_distinct_versions` | §6.3. Threads, because MLflow's read-max+1 loses a version and ours must not |
| `test_large_int_refused_not_rounded` | §4.6. The specific value `12345678901234567168`, because a library swap could silently start rounding it |
| `StoreConformance` | §6.1. A suite any backend runs against itself, so v2 finds the leaks at implementation time rather than in production |

`test_copy_out_refuses_to_return_store_path` and
`test_sort_is_byte_order_not_str_order` deserve a note: both test a claim that
is structural rather than behavioural, and they exist because §5.1 and §4.3
argue that the structure *is* the guarantee. A claim of that kind should fail
loudly when someone adds a convenience overload.

---

## 13. Implementation order

Test first, in dependency order. Each step green before the next.

| # | Module | Depends on |
|---|---|---|
| 1 | `errors` | — |
| 2 | `digest` — the tree walk and the canonical encoder | 1 |
| 3 | `readme` | 1 |
| 4 | `pointer` | 1 |
| 5 | `locality` | 1 |
| 6 | `content` — the four types, the item split | 2–5 |
| 7 | `kind` + `registry` — including the reverse index | 6, `spec_loader` |
| 8 | `verdict` | 1 |
| 9 | `store` — `FilesystemStore` and `StoreConformance` | 6, 8 |
| 10 | the closure-pass contribution — the agreement check | 7 |

Steps 2–5 are pure functions and independently testable, which is why §2.1's
import rule matters: they can all be written before anything else exists. Step 2
carries the most weight — its reference vectors are checked in, and a change to
them after the alpha is a `v2` (§4.7), not an edit.

---

## 14. Deviations from the spec

None changes an acceptance criterion.

| # | Spec says | Design does | Why |
|---|---|---|---|
| **D1** | ~~"a jsonpath into the content" (§5.1)~~ | **No longer a deviation.** Spec §5.1 rev. 5 and `validator` spec §4.1 rev. 7 both say RFC 6901 JSON Pointer | Rev. 1 reported it and declined to edit the spec, which was right. Decided by the user in the stage-three consistency pass, on the argument rev. 1 made: RFC 9535 §2.5.1.2 **forbids a valid JSONPath query from erroring**, so no implementation can distinguish a wrong path from an absent value. The reason now lives in the spec, where the next reader will meet it before the library choice rather than after |
| **D2** | "the two-way binding … crashes at load" (§5.1) | Implemented as written, **with no precedent** | §8.3. SQLAlchemy's `back_populates` — the closest analogue — **does not check that the two sides agree** (verified on 2.0.44: a flat contradiction configures with no error or warning; `relationships.py::_add_reverse_property` never compares `other.back_populates == self.key`). GraphQL Federation **deleted** the requirement in Fed 2 (`KEY_NOT_SPECIFIED` in the removed-codes list). Kubernetes treats a dangling ownerRef as *absent*, not an error. Django derives one side. **The design implements the spec; the deviation is that nobody else does this**, and §15 O6 states the strongest argument against |
| **D3** | Content is "a README plus a dict" (§3.1) | The dict is split — file-valued items are files, data-valued items live in `items.json` | §3.1. Forcing a `reproducible` handoff's `logs` through JSON is a decision nobody could undo. The spec's model is preserved; only its on-disk realisation is two files |
| **D4** | "the validators are limited to schema conformance and internal consistency" for knowledge handoffs (§4.1) | The restriction lives in the **kind**, not in a `KnowledgeStore` | §6.5. It is a property of what a knowledge kind may declare, and enforcing it in the store would put a policy in the layer that has no access to the validator registry |

---

## 15. New open questions

Found by this design, and **not** in spec §10.

| # | Question |
|---|---|
| **O1** | **Nothing in the spec set names the tree-digest algorithm.** §3.3 fixes sha256 and its scope but not what the walk covers, and §4.2 had to specify it. in-toto registers `dirHash` with a shell equivalent precisely because leaving it implicit makes implementations disagree silently — and note that in-toto's own definition's `-type f` **silently drops symlinks and empty directories**, which is exactly the kind of exclusion that must be written down. Should §3.3 name `agent_sys.handoff.tree.v1` and point here? |
| **O2** | **Three runtime dependencies are undeclared, and the accident is wider than rev. 1 thought.** Re-measured in the stage-three pass: `python-jsonpath` is **still not installed** — it is the one library this design actually chose, and the two it rejected (`jsonpath-ng` 1.8.0, `jsonpointer` 3.1.1) both are. `markdown-it-py` 4.0.0 is installed transitively via `rich`. **`rfc8785` 0.1.4 IS installed**, which rev. 1 recorded as absent. So a test would pass today, on this machine, using a library this design rejected — and fail on a clean install. This compounds [`../../docs/design.md`](../../docs/design.md) §12 O1, which now carries the full list, and [`../../docs/interfaces.md`](../../docs/interfaces.md) §7 carries the declaration |
| **O3** | **Garbage collection between an artefact and its verdict is unsolved everywhere, and this design does not solve it.** `delete_version` is deliberately absent from §6.1's Protocol. OCI distribution-spec#378 — *"How are registries expected to behave when a subject is deleted?"* — has been open since 2023 with maintainers disagreeing in-thread (*"They're decoupled"* versus *"you wind up with zombie objects"*); mark-and-sweep does not traverse `subject`, so GitLab#966 notes referrers *"would be GC'ed soon after being pushed"*, and zot#4271 is a **single dangling reference that silently disabled GC estate-wide** while ~90 GiB accumulated. REAPI#138 (retraction) has been open ~7 years. **Nix's direction is the one design that makes the harmful case unrepresentable** — GC roots point *at* content, so a dead root is auto-unlinked and a content-orphan cannot arise. Worth deciding before anything is deleted, rather than after |
| **O4** | **NFC and NFD are two distinct keys, and JCS mandates that they stay so** (§3.1: *"MUST preserve Unicode string data 'as is'"*). Two `items` keys that render identically on screen produce different digests, with no warning. They also coexist as two visually identical **filenames** on Linux and **collide on macOS**. Nothing in the spec set says a key must be NFC, and §7.3's mint-time allow-list is where it would go |
| **O5** | **Should the locality check weight by file role?** §10.4. A path in a re-executable script is stronger evidence than one in a log. lintian has the mechanism (`Item.pm::mentions_in_operation`) and **does not wire it in**; this design follows lintian and does not either. It is the first refinement anyone will ask for after the first false positive |
| **O6** | **Rust's orphan rule is the strongest argument against D2, and it survives our disanalogies.** Coherence makes disagreement *unrepresentable* rather than detectable — at most one crate is ever eligible to state the fact (`coherence.rs`: *"2 mutually-unknowing crates … can't implement the same trait-ref"*). Three things weaken the analogy: Rust's hazard is two *different behaviours* claiming one slot where ours is two *identical assertions*; Rust's world is open with no point of total visibility where ours is closed and loaded in one process — which is what the closure pass *is*; and Rust cannot show the other span where we always can. But the deeper claim survives all three: **the possibility of two declarations is itself the defect.** Our closed world makes the crash tolerable; it does not make the redundancy necessary. The counter-argument is equally real and comes from the same domain — **SQLAlchemy 2.0 moved *away* from deriving one side**, marking `backref` *"legacy"* because *"all arguments are explicit"* and PEP 484 *"take[s] advantage of attributes being explicitly present in source code"*. Fifteen years of the derived form, then a reversal |
| **O7** | **`addons.temp` retention interacts with resumability, and nothing sweeps a playground.** Inherited from spec §10, sharpened here: §6.3's `.staging-` directories are cleaned on failure, but a playground is `env_mgr`'s and is also what makes an agent resumable (`env_mgr` spec §6). The sweep cannot be unconditional, and no owner has been named |
| **O8** | **A second implementation is the only real test of §4.2.** §12.2's reference vectors catch a *change*, not a *misreading*. in-toto ships a shell equivalent for this reason. Fifteen lines of shell alongside the Python would turn a specification into a checkable one, and it is not written here |
