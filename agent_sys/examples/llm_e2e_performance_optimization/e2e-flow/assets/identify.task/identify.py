#!/usr/bin/env python3
"""What `identify` runs: locate each selected operator's framework-level entry.

DESIGN.md section 4.4. Three resolution levels, tried in order, and the level
reached is recorded rather than hidden:

| level | `source_resolution_method` | evidence |
|---|---|---|
| 1 | `trace_python_stack` | the `kernel_table` row carried a `launcher` block |
| 2 | `symbol_search` | the symbol's compound name was found in an indexed repository |
| 3 | `agent_recovered` | neither did; direction plus a hint, entry function read from source by `build_workset` |

**Level 3 is not a failure.** AgentKernelArena's own KDA and TileLang tasks live
there: a Triton-JIT or TileLang device symbol has no source file, because it is
a compilation artefact. What is editable is the Python function that generated
it, and finding that means reading the framework rather than matching a string.

Everything written into the handoff is locality-independent. `handoff/locality.py`
refuses to seal content carrying an absolute path outside an anchored allow-list,
and `/sgl-workspace/` is not on it — so `image_repo_path` carries a **placeholder**
like `${AITER_ROOT}`, and `assets/lib/container_roots.yaml` says what it expands
to for this image. Host checkout paths are never recorded at all: a repository is
named, not located.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PACKAGE = Path(
    os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ["AGENT_SYS_DEMO_PACKAGE"]
)
_LIB = _PACKAGE / "assets" / "lib"
sys.path.insert(0, str(_LIB))

import schema as schema_lib  # noqa: E402
import store  # noqa: E402
import symbols  # noqa: E402

#: Owner -> `${PLACEHOLDER}`, loaded from `container_roots.yaml`. The concrete
#: container path stays in the package; only the placeholder reaches a handoff.
_ROOTS = yaml.safe_load((_LIB / "container_roots.yaml").read_text(encoding="utf-8"))
def _container_roots(document: dict) -> tuple[dict[str, str], dict[str, str]]:
    """`(owner -> placeholder, placeholder -> description)` from either shape of
    `container_roots.yaml`.

    Two shapes exist in this repository and they carry the same four roots with
    the same four paths: `analyze-demo`'s is keyed by owner and names a
    `placeholder` field; **this package's is keyed by the placeholder itself**
    and carries no owner. Deriving the owner from the placeholder name --
    `AITER_ROOT` -> `aiter`, `SGL_KERNEL_ROOT` -> `sgl_kernel` -- is exact for
    all four and needs no edit to a file another module owns.

    The sigil follows the file: this package writes `@NAME@` rather than
    `${NAME}` because `locality._CANDIDATE`'s lookbehind excludes `@` and not
    `}`, so `${SGLANG_ROOT}/srt/models/x.py` leaves `/srt/models/x.py` as a
    fresh two-segment candidate and is flagged anyway. Both forms are accepted
    downstream, because the sealed artefact this package mocks against carries
    the `${NAME}` one.
    """
    if document.get("owners"):
        pairs = [(owner, spec["placeholder"], spec.get("description", ""))
                 for owner, spec in document["owners"].items()]
        sigil = "${%s}"
    else:
        pairs = [(name[:-5].lower() if name.endswith("_ROOT") else name.lower(), name,
                  spec.get("description", ""))
                 for name, spec in (document.get("roots") or {}).items()]
        sigil = "@%s@"
    return ({owner: sigil % placeholder for owner, placeholder, _ in pairs},
            {sigil % placeholder: description for _, placeholder, description in pairs})


CONTAINER_ROOTS, CONTAINER_ROOT_DESCRIPTIONS = _container_roots(_ROOTS)

#: Which owner a symbol belongs to. Narrower than the taxonomy's `fellow`
#: mapping, because this answers "whose repository" rather than "which language".
OWNER_PATTERNS = [
    ("aiter", [r"^_ZN5aiter", r"^void aiter::", r"^aiter", r"^mfma_"]),
    ("sgl_kernel", [r"^_ZN7sgl_hip", r"sgl_hip", r"topk_transform", r"anonymous namespace"]),
    ("tilelang", [r"^main_kernel$"]),
    ("sglang", [r"^_fused_", r"^_gluon_", r"^_gemm_", r"_kernel_BLOCK_SIZE", r"^triton_"]),
]

README = """# operator_identity

## Purpose

Where each selected operator lives: which repository owns it, what language it
is written in, which files carry it, and which functions are the editable entry
points.

The field names are AgentKernelArena's `config.yaml` vocabulary, not invented
here — `image_repo_path`, `repo_subdir`, `repository_language`,
`source_file_path`, `editable_sources`, `target_kernel_functions`,
`kernel_identity`. Adopting an existing vocabulary is what lets the next step
emit Arena-shaped output with no translation layer.

Resolution reached on this run:

{resolution_table}

## Schema

`items/text.json`:

```json
{{"generated_at": "...",
  "resolver": {{"kernel_finder": "available|unavailable", "indexed_repos": []}},
  "summary": {{"operators": 0, "resolved": 0, "resolve_ratio": 0.0}},
  "operators": [
    {{"kernel_id": "k002", "name": "<device symbol>",
      "logical_operator": "moe_gemm_silu_mul",
      "image_repo_path": "${{AITER_ROOT}}",
      "repo_subdir": "aiter",
      "repository_language": "ck",
      "source_file_path": [],
      "editable_sources": [],
      "target_kernel_functions": [],
      "kernel_identity": {{"logical_operator": "...", "kernel_kind": "ck",
                          "source_owner": "aiter"}},
      "source_resolution_method": "agent_recovered",
      "resolution_hint": "...",
      "baseline_ref_file": "", "baseline_ref_symbol": "", "baseline_ref_kind": "",
      "test_file": "", "test_cmd": "",
      "dtypes": {{"activation": "fp4"}},
      "fellow": "ck-fellow"}}
  ]}}
```

`source_resolution_method` is one of `trace_python_stack`, `symbol_search`,
`agent_recovered`. The third means the resolver gave direction and no entry
point; `resolution_hint` says where to look, and the next step reads the source
to fill `target_kernel_functions`.

## Watch out

**No absolute path appears here, host or container.** `image_repo_path` carries
a placeholder — `${{AITER_ROOT}}` and the like — and `assets/lib/container_roots.yaml`
in the producing package says what each expands to for the declared image.

`handoff/locality.py` refuses to seal content naming an absolute path outside an
anchored allow-list, and `/sgl-workspace/` is not on it. The mechanism meant for
this case exists — `Oracles.image_prefixes`, "prefixes the declared container
image makes portable" — and `handoff.schema.json` has the matching `dependencies`
field, but nothing connects the two; see
`temp/bugs/002-handoff-dependencies-never-reach-locality-check.md`. Keeping the
expansion in the package is also right on its own terms: a different engine image
moves every owner at once, so it is one edit rather than one per operator.

Placeholders in use on this run:

{placeholder_table}

Where a repository happens to be checked out on the machine that produced this
is deliberately not recorded at all: it would be wrong anywhere else.
"""



def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; this body has nowhere to write.")
    return value


#: `handoff/locality._CANDIDATE` and `_URL`, duplicated. The same bounded
#: duplication `demo/assets/lib/store.py` makes of the store layout: the
#: alternative is a task package importing a framework component.
_ABS_PATH = re.compile(
    r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)"
)
_ALLOWED = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/", "/opt/",
    "/proc/", "/sys/", "/dev/", "/var/lib/", "/var/log/", "/run/", "/srv/",
    "/workspace/", "/app/",
)


def _scrub(text: str) -> str:
    """Replace any absolute path the seal would refuse with `<path elided>`.

    Applied to free text — resolver hints and notes — where a path can arrive
    from a tool this package does not control. Failing here with the string in
    hand beats `output was never delivered` after the fact.
    """
    def replace(match: re.Match) -> str:
        path = match.group(0)
        return path if path.startswith(_ALLOWED) else "<path elided>"

    return _ABS_PATH.sub(replace, text or "")


def owner_of(name: str) -> str:
    for owner, patterns in OWNER_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, name):
                return owner
    return ""


def defines_symbol(relative: str, symbol: str, repos: list[str]) -> bool:
    """Does `relative` actually define `symbol`?

    A definition, not a mention: `def name(`, `class name`, or a C-style
    `name(`. A file that merely calls the function is not where it lives, and
    pointing `build_workset` at a call site would make it write a reference
    implementation against the wrong signature.
    """
    if not relative or not symbol:
        return False
    pattern = re.compile(
        rf"(^|\n)\s*(def|class)\s+{re.escape(symbol)}\b|(^|\n)[\w:<>,\s*&]*\b{re.escape(symbol)}\s*\("
    )
    for repo in repos:
        candidate = Path(repo) / relative
        if not candidate.is_file():
            continue
        try:
            return bool(pattern.search(candidate.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            return False
    return False


def logical_operator(row: dict) -> str:
    """A stable, readable operator name.

    Built from the category plus the distinguishing token of the symbol, because
    the raw symbol carries tile sizes and template arguments that change between
    builds and would make the identity unstable. `mfma_moe1_silu_mul_afp4_...`
    becomes `moe_gemm_moe1_silu_mul`.
    """
    category = row.get("category") or "unknown"
    stem = re.sub(r"^(_ZN\d*|void\s+|\(anonymous namespace\)::)", "", row["name"])
    stem = re.split(r"[<(]", stem)[0]
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t and not t.isdigit()]
    # Drop tile/tuning tokens: they encode a build, not an operator.
    noise = re.compile(r"^(t\d+x|v\d+|pm\d+|cu\d+|fix\d+|BLOCK|SIZE|kernel|async|persist)", re.I)
    kept = [t for t in tokens if not noise.match(t)][:4]
    name = "_".join(kept).lower()
    return f"{category}_{name}"[:96].strip("_") if name else category


def kernel_finder(magpie_root: Path, symbols: list[str], repos: list[str], timeout: int) -> dict:
    """Run Magpie's `amd_kernel_finder` out of process, and survive its absence.

    Out of process on purpose: the finder clones repositories and builds an
    index, so a failure there is a subprocess exit code rather than an exception
    inside this body. The result is a `{symbol: KernelSourceInfo-as-dict}` map,
    or `{}` when the tool is not usable — in which case every operator falls to
    resolution level 3, which is a supported outcome and not an error.
    """
    if not magpie_root.is_dir():
        return {"available": False, "reason": f"magpie root absent: {magpie_root.name}", "hits": {}}

    probe = magpie_root / "Magpie" / "tools" / "amd_kernel_finder" / "finder.py"
    if not probe.is_file():
        return {"available": False, "reason": "amd_kernel_finder not present", "hits": {}}

    script = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
symbols = json.loads(sys.argv[2])
repos = [r for r in json.loads(sys.argv[3]) if r]
try:
    from Magpie.tools.amd_kernel_finder import KernelSourceFinder, KernelSourceInfo
except Exception as error:
    print(json.dumps({"available": False, "reason": f"import failed: {error}", "hits": {}}))
    sys.exit(0)
try:
    finder = KernelSourceFinder(repos=repos, auto_clone=False, use_index=bool(repos))
    headers = KernelSourceInfo.csv_headers()
    hits = {}
    for symbol in symbols:
        try:
            hits[symbol] = dict(zip(headers, finder.search(symbol).to_list()))
        except Exception as error:
            hits[symbol] = {"notes": f"search failed: {error}"}
    print(json.dumps({"available": True, "reason": "", "hits": hits,
                      "indexed_repos": [r.rsplit("/", 1)[-1] for r in repos]}))
except Exception as error:
    print(json.dumps({"available": False, "reason": str(error), "hits": {}}))
"""
    try:
        done = subprocess.run(
            [sys.executable, "-c", script, str(magpie_root), json.dumps(symbols), json.dumps(repos)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"timed out after {timeout}s", "hits": {}}
    if done.returncode != 0:
        return {
            "available": False,
            "reason": f"exit {done.returncode}: {(done.stderr or '').strip()[:400]}",
            "hits": {},
        }
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {
            "available": False,
            "reason": f"unparseable output: {(done.stdout or '').strip()[:300]}",
            "hits": {},
        }


#: Magpie's `amd_kernel_finder` reports paths in a repo-variable form —
#: `$AITER_DIR/aiter/ops/rmsnorm.py` — using the `var_name` values in its
#: `repo_config.py`. Stripping the variable leaves exactly the repository-relative
#: path Arena's `source_file_path` holds.
_REPO_VAR = re.compile(r"^\$[A-Z][A-Z0-9_]*/")


def _repo_relative(path: str, roots: list[str]) -> str:
    """Strip a checkout prefix or a Magpie repo variable, leaving a relative path.

    A repository-relative path is both portable and what Arena's
    `source_file_path` holds. A path that cannot be made relative is dropped
    rather than recorded: keeping it would fail the seal and would be wrong on
    any other machine.
    """
    path = (path or "").strip()
    if not path:
        return ""
    if _REPO_VAR.match(path):
        return _REPO_VAR.sub("", path)
    if not path.startswith("/"):
        return path
    for root in roots:
        if root and path.startswith(root.rstrip("/") + "/"):
            return path[len(root.rstrip("/")) + 1:]
    for marker in ("/aiter/", "/sglang/", "/vllm/", "/pytorch/", "/triton/", "/site-packages/"):
        index = path.find(marker)
        if index >= 0:
            return path[index + len(marker):]
    return ""


def bind_launcher(launcher: dict, repo_map: dict) -> tuple[str, str]:
    """`(repo_relative_path, evidence)` for a launcher frame, or `("", why)`.

    **This exists because the producer cannot finish the job and this can.** torch
    writes a frame path with the longest matching `sys.path` entry stripped off,
    and which entry that was is not recoverable from the string: `aiter/ops/x.py`
    is equally consistent with a `sys.path` entry of `/sgl-workspace/aiter` and
    one of `/sgl-workspace`. So `profiling-demo` reports the path it saw and marks
    it `path_form: sys_path_relative`, and the binding happens here, where the
    repository is checked out and a candidate can be tested against the
    filesystem instead of asserted.

    Confirmed against the checkout even for a `container_absolute` frame, which is
    already exactly root-relative. The two roots can still disagree: this package
    copies `/sgl-workspace/sglang/python/sglang` out of the image, so a frame
    relative to `/sgl-workspace/sglang/python` is one segment longer than the
    checkout wants.

    An empty path drops the caller to symbol search, which is the honest outcome:
    a path this cannot find is one the next stage cannot open either.
    """
    recorded = (launcher.get("source_file") or "").strip()
    if not recorded:
        return "", "the launcher block carries no source_file"
    if recorded.startswith("/"):
        # The producer splits the container root off precisely so this cannot
        # happen. An absolute path here would fail the seal downstream and cannot
        # be resolved against a checkout, so it is refused rather than guessed at.
        return "", f"launcher path {recorded!r} is absolute"

    # Candidates, most specific first: the path as recorded, then the same path
    # with its leading segment dropped. The second covers a frame stripped against
    # the *parent* of the checkout root, which is what an editable install of
    # sglang produces.
    candidates = [recorded]
    head, _, tail = recorded.partition("/")
    if tail:
        candidates.append(tail)

    if not repo_map:
        # Nothing indexed to check against. The recorded path is still the best
        # answer available and is reported as unverified rather than dropped.
        return recorded, "no indexed repository to confirm the path against"

    for candidate in candidates:
        for name, root in repo_map.items():
            if (Path(root) / candidate).is_file():
                return candidate, f"confirmed in {name}"

    return "", (
        f"launcher path {recorded!r} is in none of the indexed repositories "
        f"({', '.join(sorted(repo_map)) or 'none'})"
    )


def resolve(row: dict, hit: dict, roots: list[str], repo_map: dict) -> dict:
    """One selected kernel -> one operator identity record."""
    owner = owner_of(row["name"]) or (hit.get("source_repo") or "").strip().lower()
    language = row.get("language") or (hit.get("kind") or "").replace("_jit", "")

    launcher = row.get("launcher") or {}
    source_files: list[str] = []
    entry_functions: list[str] = []
    method = "agent_recovered"
    hint = ""
    search: dict = {}
    launcher_note = ""

    if launcher.get("source_file"):
        # Level 1. The profile carried the Python call site.
        #
        # **Bound to a file before it is claimed, and it falls through when it
        # cannot be.** Level 1 used to be entered on the presence of the block
        # alone, which meant a frame this package could not place still reported
        # `trace_python_stack` with an empty `source_file_path` — a claim
        # `check_identity_resolved` rejects, and one that had already skipped the
        # symbol search that would have answered.
        bound, launcher_note = bind_launcher(launcher, repo_map)
        if bound:
            source_files.append(bound)
            if launcher.get("function"):
                entry_functions.append(launcher["function"])
            method = "trace_python_stack"

    if method == "agent_recovered" and repo_map:
        # Level 2. Search the indexed repositories for the symbol's compound
        # name kept contiguous. `assets/lib/symbols.py` explains why that is the
        # unit: splitting into tokens is what made the earlier attempt answer
        # `triton_store_cache.py` for an `add_rmsnorm_quant` kernel.
        search = symbols.find_source(row["name"], repo_map)
        if search["relative"]:
            source_files.append(search["relative"])
            method = "symbol_search"
        else:
            hint = search["why"]

    if method == "agent_recovered":
        hint = hint or (
            f"Device symbol {row['name'][:60]!r} could not be located by name"
            + (f" ({search.get('why')})" if search.get("why") else "")
            + f". Read {CONTAINER_ROOTS.get(owner, 'the serving container')} and name the "
            f"{language or ''} entry function that launches it. Category is "
            f"{row.get('category')}; observed shapes are in the worklist cases."
        )
        if launcher.get("source_file"):
            # A frame was recorded and could not be bound. Naming it is the most
            # useful thing here: it says which file to look at even though this
            # step could not confirm it, which is strictly more than the symbol
            # name gives.
            hint = hint.rstrip()
            hint += "" if hint.endswith(".") else "."
            hint += (
                f" The profile recorded a call site at "
                f"{launcher.get('source_file')}:{launcher.get('line')} in "
                f"{launcher.get('function') or 'an unnamed function'} "
                f"(under {launcher.get('container_root') or 'an unknown root'}), "
                f"which this step could not bind to an indexed file: {launcher_note}."
            )

    if method == "trace_python_stack" and launcher.get("owner"):
        # **The launcher's owner wins over the symbol's.** `owner_of` reads the
        # device symbol, which names whichever toolchain compiled the kernel;
        # the launcher names the repository the editable entry point is actually
        # in, and those differ exactly where it matters. `main_kernel` is
        # TileLang's symbol for everything it generates, so the symbol says
        # `tilelang` while the frame says the function to edit is in sglang's
        # attention backend — and DESIGN.md section 4.4 is that the second is the
        # workset's unit.
        owner = launcher["owner"]
    # The hint is free text and a resolver may have quoted a path into it. The
    # seal scans it like any other string, so scrub it here rather than losing
    # the whole handoff twenty seconds later.
    hint = _scrub(hint)

    # The PyTorch reference (mission 3.2.4) comes from the same grep and carries
    # the same risk, so it gets the same treatment: a symbol is kept only when
    # its file really defines it. `build_workset` imports what survives and
    # writes its own reference for what does not.
    baseline_file = _repo_relative(hit.get("baseline_ref_file", ""), roots)
    baseline_symbol = (hit.get("baseline_ref_symbol") or "").strip()
    baseline_verified = bool(
        baseline_file and baseline_symbol and defines_symbol(baseline_file, baseline_symbol, roots)
    )

    return {
        "kernel_id": row["kernel_id"],
        "name": row["name"],
        "rank": row.get("rank"),
        "pct_total": row.get("pct_total"),
        "calls": row.get("calls"),
        "logical_operator": logical_operator(row),
        "image_repo_path": CONTAINER_ROOTS.get(owner, ""),
        "repo_subdir": owner,
        "repository_language": language,
        "source_file_path": source_files,
        # Everything resolved is editable until something says otherwise; Arena
        # narrows this when a generated file sits beside a hand-written one.
        "editable_sources": list(source_files),
        "target_kernel_functions": entry_functions,
        "kernel_identity": {
            "logical_operator": logical_operator(row),
            "kernel_kind": language,
            "source_owner": owner,
        },
        "source_resolution_method": method,
        "resolution_hint": hint,
        "resolution_probe": search.get("probe", ""),
        "resolution_evidence": launcher_note or search.get("why", ""),
        # The call site as the profile recorded it, kept even when the binding
        # above failed. A line number and a function name are evidence a reader
        # can check, and dropping them because this step could not confirm the
        # file would throw away the only thing a name search never provides.
        "launcher": {
            "source_file": launcher.get("source_file", ""),
            "line": launcher.get("line"),
            "function": launcher.get("function", ""),
            "launch_api": launcher.get("launch_api", ""),
            "sample_count": launcher.get("sample_count"),
            "container_root": launcher.get("container_root", ""),
            "path_form": launcher.get("path_form", ""),
        } if launcher else {},
        "baseline_ref_file": baseline_file if baseline_verified else "",
        "baseline_ref_symbol": baseline_symbol if baseline_verified else "",
        "baseline_ref_kind": (hit.get("baseline_ref_kind") or "").strip() if baseline_verified else "",
        # Kept even when unverified: a candidate a human can check beats nothing,
        # as long as it is not presented as a fact.
        "baseline_ref_candidate": (
            "" if baseline_verified else (f"{baseline_file}::{baseline_symbol}" if baseline_symbol else "")
        ),
        "triton_ref_symbol": (hit.get("triton_ref_symbol") or "").strip(),
        "test_file": _repo_relative(hit.get("test_file", ""), roots),
        "test_cmd": (hit.get("test_cmd") or "").strip(),
        "category": row.get("category"),
        "dtypes": row.get("dtypes") or {},
        "precision": row.get("precision") or "",
        "fellow": row.get("fellow") or "",
        "cases": row.get("cases") or [],
    }


def main() -> int:
    staged = store.declared_dir("kernel_worklist", direction="INPUT")
    if staged is None:
        raise SystemExit("AGENT_SYS_INPUT_KERNEL_WORKLIST does not name a readable directory.")
    worklist = json.loads((staged / "items" / "text.json").read_text(encoding="utf-8"))
    selected = [k for k in worklist["kernels"] if k.get("selected")]
    if not selected:
        raise SystemExit("the worklist selected no kernels; nothing to identify")

    magpie_root = Path(os.environ.get("E2E_MAGPIE_ROOT") or "/nonexistent")
    repos = [r for r in (os.environ.get("E2E_SGLANG_SRC"), os.environ.get("E2E_AITER_SRC")) if r]
    timeout = int(os.environ.get("E2E_RESOLVE_TIMEOUT_S") or 1800)

    finder = kernel_finder(magpie_root, [k["name"] for k in selected], repos, timeout)
    hits = finder.get("hits") or {}

    repo_map = {Path(r).name: Path(r) for r in repos if Path(r).is_dir()}
    operators = [resolve(row, hits.get(row["name"]) or {}, repos, repo_map) for row in selected]
    resolved = sum(
        1
        for o in operators
        if o["source_resolution_method"] in {"trace_python_stack", "symbol_search"}
    )

    out = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resolver": {
            "kernel_finder": "available" if finder.get("available") else "unavailable",
            "reason": _scrub(finder.get("reason", "")),
            # Repository *names*, never checkout paths: see the module docstring.
            "indexed_repos": finder.get("indexed_repos") or [
                r.rstrip("/").rsplit("/", 1)[-1] for r in repos
            ],
        },
        # Placeholder -> what it means. Deliberately no expansion: the concrete
        # container path lives in the producing package's
        # `assets/lib/container_roots.yaml`, so a consumer resolves it against
        # the image it is actually running.
        "container_root_placeholders": dict(CONTAINER_ROOT_DESCRIPTIONS),
        "summary": {
            "operators": len(operators),
            "resolved": resolved,
            "resolve_ratio": round(resolved / len(operators), 3),
        },
        "operators": operators,
    }

    dst = Path(_required("AGENT_SYS_OUTPUT_OPERATOR_IDENTITY"))
    items = dst / "items"
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # CONTRACT.md 3.4 -- byte-identical to the package's, and copied rather than
    # re-serialised, because the check is on bytes precisely so a private fork
    # cannot hide behind equal semantics.
    (items / "schema").write_bytes(schema_lib.schema_path("operator_identity").read_bytes())

    # CONTRACT.md 2 -- passed through from the worklist unchanged. This task
    # learns nothing new about the machine, so re-deriving the record could only
    # introduce a disagreement with the profile it came from.
    (items / "env").mkdir(parents=True, exist_ok=True)
    (items / "env" / "environment.yaml").write_text(
        (staged / "items" / "env" / "environment.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    try:
        schema_lib.validate("operator_identity", out)
    except schema_lib.SchemaError as error:
        raise SystemExit(f"identify produced an identity document that does not validate:\n{error}")

    log = [
        f"kernel_finder: {out['resolver']['kernel_finder']}"
        + (f" ({finder.get('reason')})" if finder.get("reason") else ""),
        f"indexed repositories: {', '.join(out['resolver']['indexed_repos']) or 'none'}",
        "",
    ]
    for operator in operators:
        log.append(
            f"{operator['kernel_id']} {operator['logical_operator']}: "
            f"{operator['source_resolution_method']}"
            + (f" -> {', '.join(operator['source_file_path'])}" if operator["source_file_path"] else "")
        )
        if operator["resolution_hint"]:
            log.append(f"    hint: {operator['resolution_hint']}")
    (items / "resolution_log.txt").write_text(_scrub("\n".join(log)) + "\n", encoding="utf-8")

    table = "\n".join(
        f"| `{o['logical_operator']}` | {o['source_resolution_method']} | "
        f"{len(o['source_file_path'])} file(s) | {len(o['target_kernel_functions'])} entry point(s) |"
        for o in operators
    )
    placeholders = "\n".join(
        f"| `{placeholder}` | {description} |"
        for placeholder, description in CONTAINER_ROOT_DESCRIPTIONS.items()
    )
    (dst / "README.md").write_text(
        README.format(
            resolution_table=(
                "| operator | method | files | entry points |\n|---|---|---|---|\n" + table
            ),
            placeholder_table="| placeholder | what it holds |\n|---|---|\n" + placeholders,
        ),
        encoding="utf-8",
    )

    print(
        f"identify: {len(operators)} operators, {resolved} resolved by evidence "
        f"(ratio {out['summary']['resolve_ratio']}), finder "
        f"{out['resolver']['kernel_finder']}"
    )
    for operator in operators:
        print(f"  {operator['kernel_id']} {operator['logical_operator']}: {operator['source_resolution_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
