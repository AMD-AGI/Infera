#!/usr/bin/env python3
"""Reading a published handoff from the store, without importing `handoff`.

A validator body runs as a subprocess in a fresh zone and is handed
`inputs.json` — handoff **ids**, as strings — and nothing else. There is no
route from an id to the content it names: `docs/interfaces.md` §5.8 leaves
*"who materialises the value a JSON Pointer addresses"* open, and this is that
question one level wider. Reported as F-D5 in `examples/demo/README.md`.

**This file is a verbatim copy of `examples/demo/assets/lib/store.py`**, and is
kept verbatim on purpose. The agreement test named below pins *that* copy
against `handoff`'s real constants; a copy here that drifted would be a second,
unpinned reader of a layout `handoff` owns. Fix the original and copy it across.

So this module reads the store's on-disk layout directly, through
`AGENT_SYS_DEMO_STORE`. That layout is `handoff`'s — `<root>/<hid>/v<N>/` with
`content/` and `manifest.json` inside — and reading it here is a **second
reader of a fact `handoff` owns**. It is admissible only because the
alternative is a task package importing a component, and it is bounded: this
file reads directory names and `manifest.json`'s `kind`, and nothing else.

`tests/cli/test_isolation_shown.py::test_the_store_layout_this_package_reads_is_handoffs`
is the price of that duplication, on the terms `docs/interfaces.md` §8.1 sets:
it fails the day `handoff.version_dir` disagrees with what is read here.

**Most of this is now a fallback rather than the route.** `validator` landed
`materials.json` — a declared, body-facing file naming the staged copies of what
the phase validates — so a body reading its *own* inputs needs none of the
below. What is left needing it is the case `materials.json` does not cover: a
validator that must reach a handoff it was **not** handed, which is what
*grounded in its input* means and which the staging does not stage.
"""

import json
import os
import re
from pathlib import Path

import yaml

#: `handoff.store.version_dir`'s naming, duplicated. The agreement test is why
#: this constant is spelled out rather than guessed at each call site.
VERSION_DIR = re.compile(r"^v(\d+)$")

#: **`handoff.store.MANIFEST_FILE`, duplicated.** It is `manifest.yaml` and this
#: said `manifest.json` until `handoff` measured it: `kind_of` returned `''` and
#: `latest_of_kind('facts')` returned `None` for a handoff that was genuinely
#: published. The agreement test did not catch it because it built its fixture
#: with *this* constant and then asserted against this module — comparing a thing
#: to itself. It compares against `handoff`'s now.
MANIFEST = "manifest.yaml"

#: `handoff.store.CONTENT_DIR`, duplicated. Same test, same reason.
CONTENT = "content"


def store_root() -> Path:
    return Path(os.environ["AGENT_SYS_DEMO_STORE"])


def handoff_dir(hid: str) -> Path:
    """This handoff's directory in the store. `handoff.store.handoff_dir`, duplicated.

    The store names a directory ``handoff.<kind>.<uuid>`` so that a person
    reading a run tree can tell what an artefact is. **Nothing resolves through
    the label**: a directory is this handoff's when its name *is* the uuid — the
    shape written before labels existed — or ends with ``.<uuid>``. The same
    admissible duplication as `MANIFEST` and `CONTENT` above, and covered by the
    same agreement test.
    """
    root = store_root()
    suffix = f".{hid}"
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.name == hid or entry.name.endswith(suffix):
                return entry
    return root / hid


def hid_of(dirname: str) -> str:
    """The handoff id in a store directory name — the last field, labelled or not."""
    return dirname.rsplit(".", 1)[-1]


def versions(hid: str) -> list[int]:
    """**Published versions only, and the gaps are the point.**

    `handoff.FilesystemStore.list_versions`'s rule, duplicated: since
    `interfaces.md` §4.14 a version directory is allocated **at dispatch**,
    before the body runs, so `<hid>/v3/` can exist while the attempt that owns
    it is still running or has failed and left it empty. **The manifest is what
    makes a version published**, because it is written last and by the seal.

    Without the filter, an unpublished directory on top — the common case, since
    the newest attempt is the one that failed — made `content_dir` return `None`
    for a handoff with perfectly good content in `v2`.

    A failed attempt leaves a hole: v3 is absent for ever and v4 is allocated
    next. Holes are skipped and never compacted, because renumbering would move
    an artefact a digest already names.
    """
    base = handoff_dir(hid)
    if not base.is_dir():
        return []
    found = [
        (m, e)
        for e in base.iterdir()
        if e.is_dir() and (m := VERSION_DIR.match(e.name)) and (e / MANIFEST).is_file()
    ]
    return sorted(int(m.group(1)) for m, _ in found)


def content_dir(hid: str, version: int | None = None) -> Path | None:
    """The `content/` of one version, latest by default. `None` if absent."""
    numbers = versions(hid)
    if not numbers:
        return None
    chosen = numbers[-1] if version is None else version
    path = handoff_dir(hid) / f"v{chosen}" / CONTENT
    return path if path.is_dir() else None


def kind_of(hid: str, version: int | None = None) -> str:
    numbers = versions(hid)
    if not numbers:
        return ""
    chosen = numbers[-1] if version is None else version
    manifest = handoff_dir(hid) / f"v{chosen}" / MANIFEST
    if not manifest.is_file():
        return ""
    # **YAML, not JSON**, and this read `json.loads` until the filename was
    # fixed — the third half of the same finding: `handoff._write_manifest` does
    # `yaml.safe_dump`. `yaml` is a declared dependency and a third-party
    # library, so importing it here is what any task package could do; importing
    # `handoff` is not.
    return str(yaml.safe_load(manifest.read_text(encoding="utf-8")).get("kind") or "")


def declared_dir(kind: str, *, direction: str = "INPUT") -> Path | None:
    """The content of a handoff **named by the declared environment variable.**

    `env_mgr.grants` exports `AGENT_SYS_INPUT_<KIND>` and
    `AGENT_SYS_OUTPUT_<KIND>` for every slot the *producing task* had, and a
    validation body runs with that task's resolved configuration — spec §8.2's
    producer row, measured: `config.source == ConfigSource.PRODUCER`.

    **So a validator checking `describe`'s output can reach `describe`'s input
    by its declared name**, which is the artefact the summary must be grounded
    in. Measured in demo-1's `describe` output phase, which is where the
    original of this file lives:

        AGENT_SYS_DEMO_OUTSIDE, AGENT_SYS_INPUT_FACTS,
        AGENT_SYS_OUTPUT_SUMMARY, AGENT_SYS_TASK_PACKAGE

    The equivalent here is a validator on `solutions_a` reaching
    `AGENT_SYS_INPUT_PROBLEMS` — the problem set its producer was handed.

    **This is narrower than `latest_of_kind` and that is the point.** The scan
    answers *the newest `facts` anywhere in the store*, which its own docstring
    calls crude and wrong in a real graph; this answers *the `facts` this
    producer actually consumed*. Replacing a guess with the declared fact is the
    same disposal as `materials()` — the route existed and nothing used it.

    It also needs no store root, which matters beyond tidiness: `env_mgr`
    measured `EACCES` on the store root from a confined body, so the scan
    **cannot work under confinement at all** and this can.

    The value is the content directory itself — `env_mgr` narrowed `stage` to
    `content/`, so there is no `content/` hop below it.
    """
    where = os.environ.get(f"AGENT_SYS_{direction}_{_env_name(kind)}")
    if not where:
        return None
    path = Path(where)
    return path if path.is_dir() else None


def _env_name(kind: str) -> str:
    """`env_mgr.grants._env_name`, duplicated — uppercased, non-alphanumerics to
    `_`. The same admissible duplication as `MANIFEST` and `CONTENT` above, and
    covered by the same agreement test."""
    return "".join(c if c.isalnum() else "_" for c in kind).upper()


def latest_of_kind(kind: str) -> Path | None:
    """The newest published content directory of a given kind, anywhere.

    **Crude, and it is the seam showing through.** A validator is handed the
    ids of what it is validating and nothing else, so a check that has to
    compare two handoffs — which is what *grounded in its input* means — can
    only go looking. With one `facts` handoff in the demo there is one answer;
    in a real graph there would be several and this would be wrong. That is
    F-D5, and it is why the demo reports it rather than tidying it away.
    """
    root = store_root()
    if not root.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for entry in sorted(root.iterdir()):
        hid = hid_of(entry.name)
        if not entry.is_dir() or kind_of(hid) != kind:
            continue
        found = content_dir(hid)
        if found is None:
            continue
        stamp = found.stat().st_mtime
        if best is None or stamp > best[0]:
            best = (stamp, found)
    return best[1] if best else None


def inputs() -> list[str]:
    """The handoff ids this body is validating, as `ScriptBodyRunner` wrote them."""
    return list(json.loads(Path("inputs.json").read_text(encoding="utf-8")))


def materials() -> dict[str, Path]:
    """The staged copies of what this phase validates, **by handoff id.**

    `validator` writes `materials.json` into the body's `cwd` — one entry per
    slot the phase covers, written **unconditionally** so that an empty mapping
    is a record rather than an absence.

    **It is a JSON object, and this read it as a list until it was measured.**
    Iterating a `dict` yields its **keys**, so `[Path(e) for e in json.loads(…)]`
    returned the handoff *ids* as relative paths — `Path("03831442-…")` — which
    do not exist, so `staged_content` answered `None` and every call fell
    through to the `AGENT_SYS_DEMO_STORE` fallback and died there. **No
    exception anywhere**: a dict iterates, a missing directory is falsy, and the
    whole failure surfaced three layers away as `KeyError` on an environment
    variable that was never the problem.

    That is the shape to remember rather than the bug: **a container whose
    element type changed still iterates.** The list form was real — it is what
    `staged_content`'s docstring below called *"the one guess left"* — and
    `validator` replaced it with exactly the mapping that docstring asked for.
    The producer improved and the consumer kept parsing, silently.
    """
    path = Path("materials.json")
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(hid): Path(where) for hid, where in loaded.items()}


def staged_content(hid: str) -> Path | None:
    """The staged content of one handoff, **looked up by id.**

    **The positional guess is gone**, and it was deleted rather than defended:
    `materials.json` carries the handoff id, so nothing has to infer which
    material belongs to which input, and a phase whose staged set is a
    different length from `inputs.json` is no longer a case this has to refuse.

    **The staged path is the content directory itself**, not a version
    directory with `content/` inside — measured on a live run, `<materials>/
    <hid>/v0/` holds `README.md` and `items/` directly. `env_mgr` narrowed
    `stage` to `content/` deliberately, so a body sees what it validates and
    not the manifest beside it, and appending `CONTENT` here looked for a
    directory that is correctly absent.

    `None` when the phase staged nothing for this id, which the caller must
    treat as *no content* and never as a pass.
    """
    staged = materials().get(hid)
    return staged if staged is not None and staged.is_dir() else None


def args() -> dict:
    path = Path("args.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_verdict(results: dict[str, bool]) -> None:
    """`verdict.json`, which is what makes a script body and an agent body
    substitutable at `PhaseRunner`'s seam. One entry per declared handoff: a
    missing one raises there rather than folding as falsy."""
    Path("verdict.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
