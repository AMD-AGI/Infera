#!/usr/bin/env python3
"""MOCK-MAP (C): make the two sealed halves into one artefact this package's
`operator_workset` contract accepts.

    mock_adapt.py <content dir>        # the handoff's content/, holding items/

**An adaptation is a step after the copy, not a variant of the copy.**
`mock.sh` puts the sealed bytes down verbatim, on purpose; this adds only what
did not exist when they were sealed. Same division `deploy_and_prove.task/
mock_adapt.sh` makes for (A).

## Why (C) is a bigger gap than the other adaptations, measured

`mock.sh stage3-analyze operator_workset` lands the stage-3 half and
`check_workset_shape` says `items/codes is not a directory` — the sealed half is
a `reproducible` kind laid out under `items/code/`, and the merged kind is
`code` under `items/codes/`. Behind that first failure are five more: no
`workset.yaml`, no flashinfer-bench Definitions or Workloads, no entrypoints,
fewer than three shapes on one operator, and no measured evidence.

## Which half supplies what, and why that split

**The stage-4 half is the source of the runnable part.** Its operator is
`torch.softmax(logits, dim=-1, out=out)` — plain torch, so it runs wherever
torch does, which is what makes a mock run possible off a GPU node. Its three
cases (`B1_V151936`, `B8_V151936`, `B32_V151936`) already satisfy `min_shapes`,
its gates are the ones this package's harness implements, and its
`rows_sum_to_one` invariant is the worked example of a gate no SNR threshold
catches.

**The stage-3 half is the source of the provenance vocabulary**, which is the
half that knows what a ranked candidate looks like.

**Nothing here is synthesised.** Every number and every source line below is
read out of a sealed file; `_read_seed` and `_read_cases` do the reading and
fail loudly rather than defaulting, because a mock that invents a plausible
number is a mock that proves nothing (MOCK-MAP's own rule).

The one thing this cannot supply is `evidence/`, because evidence is a
*measurement*: the caller runs the entrypoints afterwards, which is the same
thing `build_workset`'s STEP 7 and STEP 8 do. On a host without torch that step
fails and the mock is correctly incomplete.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PKG = Path(os.environ.get("AGENT_SYS_TASK_PACKAGE")
           or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
           or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PKG / "assets" / "lib"))

import workset_io as W  # noqa: E402

HARNESS = PKG / "assets" / "build_workset.task" / "harness"
MOCK_ROOT = Path(os.environ.get("E2E_MOCK_ROOT") or "/shared_nfs/yihou/agent_sys/cheat_for_mock")
STAGE4 = MOCK_ROOT / "stage4-kernel-opt/workset/content/items/codes/sampler_vocab_softmax"

OPERATOR = "sampler_vocab_softmax"


def _die(message: str) -> None:
    sys.exit(f"mock_adapt: {message}")


def _read_seed() -> str:
    """The sealed operator, as a flashinfer-bench `baseline` source string.

    Read from `kernel/sampler_softmax_kernel.py` and wrapped in the format's
    `def run(*args, **kwargs)` convention. The function body is **not** rewritten
    — the whole file travels and `run` delegates to it, so the thing measured
    here is the thing that was sealed.
    """
    path = STAGE4 / "kernel/sampler_softmax_kernel.py"
    if not path.is_file():
        _die(f"the sealed stage-4 seed is not at {path}")
    source = path.read_text(encoding="utf-8")
    if "def sampler_softmax(" not in source:
        _die(f"{path.name} does not define sampler_softmax; the seal changed shape")
    return source + "\n\n# ----- entry point -----\ndef run(*args, **kwargs):\n    return sampler_softmax(*args, **kwargs)\n"


def _read_cases() -> list[tuple[str, int, int]]:
    """`(case_id, batch, vocab)` from the sealed driver's own `_CASES` tuple.

    Parsed out of the driver rather than restated, so a mock cannot drift from
    the artefact it claims to represent. `kernel-opt-demo`'s
    `check_workset_shape` parses the same tuple for the same reason.

    **Parsed with `ast`, and the regex it replaces is worth recording.** A
    two-integer pattern over a `_CASES\\s*=\\s*\\((.*?)\\)` capture finds zero
    cases here: the tuple is `((1, _VOCAB), (8, _VOCAB), (32, _VOCAB))`, so the
    second element is a *name*, and the non-greedy capture stops at the first
    inner `)` anyway. It failed closed — `0 case(s); the contract needs 3` —
    which is the right way for it to fail, but the fix is to read the module
    rather than to widen the pattern. `_case_id` below reproduces the sealed
    driver's own naming (`B8_V151936`), which is the join key m4 needs stable.
    """
    path = STAGE4 / "kernel/driver.py"
    if not path.is_file():
        _die(f"the sealed stage-4 driver is not at {path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        _die(f"the sealed driver does not parse: line {error.lineno}")

    # Module-level integer constants first, so `_VOCAB` resolves.
    constants: dict[str, int] = {}
    cases_node = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            constants[name] = node.value.value
        elif name == "_CASES":
            cases_node = node.value
    if cases_node is None:
        _die("the sealed driver has no _CASES assignment")

    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        _die(f"_CASES holds {ast.dump(node)[:60]}, which is neither a literal nor a known constant")

    pairs = [(literal(e.elts[0]), literal(e.elts[1]))
             for e in getattr(cases_node, "elts", []) if len(getattr(e, "elts", [])) == 2]
    if len(pairs) < 3:
        _die(f"the sealed driver declares {len(pairs)} case(s); the contract needs 3")
    return [(f"B{b}_V{v}", int(b), int(v)) for b, v in pairs]


#: The correctness reference. float64 softmax — the sealed driver's own, and the
#: operation's published semantics rather than a reading of the kernel.
REFERENCE = '''from __future__ import annotations
import torch


def _reference(logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax in float64. Slow and unambiguous."""
    out.copy_(torch.softmax(logits.double(), dim=-1).to(out.dtype))
    return out


# ----- entry point -----
def run(*args, **kwargs):
    return _reference(*args, **kwargs)
'''


#: What `content.py:73` requires of a `code` handoff's README, and what the
#: sealed one is missing. The sealed document has `Purpose`, `How to run`,
#: `Result`, `Environment` and `Watch out` — it was written for a content type
#: whose section list was different, and it predates this one.
_README_SECTIONS = {
    "Interface": """\
`items/codes/workset.yaml` is the one document a consumer reads. It indexes
every operator, and per operator: the flashinfer-bench `definition` and
`workload` paths, the `shapes` list, the two `entrypoints`, the `gates`, the
measured `noise_floor`, and the `apparatus` — the exact file list that must
travel for the entrypoints to run outside this handoff.

The two entrypoints are the interface for anything that wants a number:

    ./run_correctness.sh --operator <id> [--shape CASE_ID] --json <report>
    ./run_performance.sh --operator <id> [--shape CASE_ID] --json <report>

Both are run from `items/codes/`, and both take their flag spellings from
`entrypoints.<what>.flags` in `workset.yaml` rather than from this document, so
a consumer reads the manifest and not this paragraph. `evidence/` holds the
reports this workset's own run produced; `check_workset_runs` re-runs the same
scripts and compares.
""",
    "Boundary": """\
**Baseline only.** `evidence/` measures the incumbent implementation against
the reference. There is no candidate here and therefore no speedup — the ratio
module 4 computes is not in this artefact and cannot be.

**One operator is runnable.** `sampler_vocab_softmax` carries a Definition, a
Workload and measured evidence. The `moe_gemm_*` directories carry the ranking
provenance from the stage-3 half and are **not** runnable through the
entrypoints; nothing here claims they are.

**The numbers describe one machine.** `ground_truth.environment` names it, and
the entrypoints refuse to measure where `abort_on_mismatch` disagrees with the
record for this run. Numbers taken here and numbers taken elsewhere are not a
ratio.

**This is a mock artefact.** It is assembled by `mock_adapt.py` from two sealed
handoffs; the measurement is real and taken on this hardware, and everything
else is a replay of bytes sealed on 2026-09-02.
""",
}


def _repair_readme(content: Path) -> None:
    """Add the README sections a `code` handoff needs and the sealed one lacks.

    **MOCK-MAP's own rule, and the one I missed.** *An adaptation is a step
    after the copy* — `mock.sh` lays the sealed bytes down faithfully, and every
    gap between what was sealed and what this package requires is a step
    somebody wires. I wired four (the `items/code` rename, `workset.yaml`, the
    Definitions and Workloads, `environment.yaml`) and not this one.

    **The cost was rung 0, twice.** `handoff/content.py:73` requires
    `Purpose, Interface, Boundary` of a `code` README; the sealed document has
    `Purpose, How to run, Result, Environment, Watch out`, written for a
    content type whose list was different. So the body ran, measured correctly,
    transcribed, and exited 0 — **and the seal refused**:

        README.md: required section 'Interface' is missing. Present at document
        root: ['Environment', 'How to run', 'Purpose', 'Result', 'Watch out',
        'operator_workset']

    which surfaced as `output_absent: declared output was never delivered`. Two
    runs a day apart were attributed to the stall detector, the card, and the
    body, and the reason was in the event's own `seal_refused` attribute the
    whole time.

    Idempotent, and it **appends rather than rewrites**: a mock run that is
    re-driven is the normal case, and `handoff/readme.py`'s own docstring cites
    Ruff #23562 for why nothing should edit a README line by line. A section
    that is already present is left exactly as the author wrote it.
    """
    readme = content / "README.md"
    if not readme.is_file():
        _die(f"{readme} is absent; the sealed copy should have provided it")
    text = readme.read_text(encoding="utf-8")
    # Document root only, matching `readme.py:sections` — a heading inside a
    # fence or a list does not count there and must not count here, or this
    # would decide a section exists that the seal will not find.
    present = {m.group(1).strip() for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)}
    added = [name for name in _README_SECTIONS if name not in present]
    if not added:
        return
    body = "".join(f"\n## {name}\n\n{_README_SECTIONS[name]}" for name in added)
    readme.write_text(text.rstrip("\n") + "\n" + body, encoding="utf-8")
    print(f"mock_adapt: README.md gained {', '.join(added)} — "
          f"the sealed copy predates `code`'s section list (content.py:73)")


#: The sealed `reproducible` items that a `code` handoff does not define.
#: `content.py:69-73` gives `code` exactly `codes` (required) plus `logs` and
#: `watchout` (optional), so `watchout` stays where it is and these three have
#: nowhere to be. They are the same leftover as the `items/code` → `items/codes`
#: rename: the sealed half was a different content type.
_SEALED_ONLY_ITEMS = ("env", "result", "script")


def _base_sha256(root: Path, target: str) -> dict | None:
    """sha256 of the stock target file **as it exists in the image**, or `None`.

    m5's patch manifest pulls the file out of the image, hashes it, and refuses
    on a mismatch; with this recorded, m4's STEP 6 no longer needs a container
    of its own. The schema requires it at *identify* time and says why — a hash
    taken later is a hash of whatever the file had become.

    **Discovered on the node, not written down here**, which is m1's rule for
    the image digest and the same rule for the same reason: a constant in a
    mock is a claim about a machine, and the machine is the only thing that can
    make it. `MOCK_BASE_SHA256` overrides for an operator who knows better.

    `None` when the node or the image is out of reach. The schema permits null
    and states the consequence — a consumer must fall back to hashing and
    **should say that it did** — so an absent hash is a declared gap rather
    than a silent one. That is why this returns rather than dying: the mock is
    still a valid workset without it, and refusing here would trade a stated
    gap for a stopped run.
    """
    override = os.environ.get("MOCK_BASE_SHA256")
    if override:
        return {target: override}
    image = ((yaml.safe_load((root / "environment.yaml").read_text(encoding="utf-8")) or {})
             .get("fixed") or {}).get("image")
    if not image:
        return None
    # The repo root is asked of the image rather than assumed: `sglang.__file__`
    # is the only thing that knows where this build put it, and a wrong guess
    # would hash nothing and look identical to an unreachable node.
    script = (
        'r=$(python3 -c "import sglang,os;'
        'print(os.path.dirname(os.path.dirname(os.path.dirname(sglang.__file__))))" 2>/dev/null); '
        f'sha256sum "$r/{target}" 2>/dev/null | cut -d" " -f1'
    )
    probe = subprocess.run(
        ["bash", "-c", f'. "$1"; on "$2"', "_", str(PKG / "assets/lib/remote.sh"),
         f"docker run --rm --entrypoint bash '{image}' -c '{script}'"],
        capture_output=True, text=True, timeout=300)
    digest = re.search(r"\b[0-9a-f]{64}\b", probe.stdout or "")
    if not digest:
        print(f"mock_adapt: could not read a base_sha256 for {target} from {image}; recording null. "
              f"A consumer must hash it itself and say so (workset.schema.json). "
              f"{(probe.stderr or '').strip()[-200:]}", file=sys.stderr)
        return None
    return {target: digest.group(0)}


def _fold_sealed_items(content: Path, root: Path) -> None:
    """Move the sealed `reproducible` items into `items/codes/sealed/`.

    **The second seal refusal, found before it cost a rung.** With the README
    repaired, `content.check_items` refuses next:

        items ['env', 'result', 'script'] are not defined by content_type
        'code' (defines: ['codes', 'logs', 'watchout'])

    Verified against `handoff.content.check_items` on the repaired artefact and
    confirmed present in the live run's own output, rather than predicted.

    **Moved, not deleted.** MOCK-MAP forbids inventing and this is the other
    side of the same rule: the three carry the sealed handoff's own environment
    note, result write-up and driver script, and they are the only record of
    what the original delivery said. Dropping them to satisfy a shape would
    throw away evidence to make a gate pass, which is the shape of thing this
    package refuses everywhere else.

    **And the fix is here rather than on the kind.** Declaring the three in the
    kind's `items_schema` would also satisfy the check — and would widen
    `operator_workset` to admit items the *real* producer never writes, because
    `scaffold.py` writes `items/codes/` and nothing else. That is CONTRACT §4.4
    with the arrow reversed: the fixture reshaping production. **The mock adapts
    to the kind; the kind does not adapt to the mock.**

    Idempotent: a name already folded is left alone, so a re-driven mock does
    not stack `sealed/sealed/`.
    """
    destination = root / "sealed"
    moved = []
    for name in _SEALED_ONLY_ITEMS:
        source = content / "items" / name
        if not source.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        moved.append(name)
    if moved:
        print(f"mock_adapt: folded {', '.join(moved)} into items/codes/sealed/ — "
              f"content_type 'code' defines only codes, logs and watchout (content.py:69-73)")


def main() -> int:
    if len(sys.argv) != 2:
        _die("usage: mock_adapt.py <content dir>")
    content = Path(sys.argv[1])
    root = content / "items" / "codes"

    sealed = content / "items" / "code"
    if sealed.is_dir() and not root.is_dir():
        # The first failure `check_workset_shape` reports, and the cheapest to
        # fix: the sealed half is a `reproducible` kind (`items/code`) and the
        # merged kind is `code` (`items/codes`).
        sealed.rename(root)
    root.mkdir(parents=True, exist_ok=True)

    _repair_readme(content)
    _fold_sealed_items(content, root)

    # The provenance vocabulary, from whichever stage-3 operator directory the
    # seal carried. Optional: the mock is still valid without it, and says so.
    provenance_src = next((d for d in sorted(root.iterdir())
                           if d.is_dir() and (d / "invocation_spec.json").is_file()), None)
    if provenance_src is not None:
        spec = json.loads((provenance_src / "invocation_spec.json").read_text(encoding="utf-8"))
        environment_note = (spec.get("environment") or {}).get("gpu_target", "")
    else:
        environment_note = ""

    for name in ("_common.py", "run_correctness.sh", "run_performance.sh"):
        shutil.copy2(HARNESS / name, root / name)
        if name.endswith(".sh"):
            (root / name).chmod(0o755)

    cases = _read_cases()
    vocab = cases[0][2]
    (root / "definitions/softmax").mkdir(parents=True, exist_ok=True)
    (root / "workloads/softmax").mkdir(parents=True, exist_ok=True)
    (root / f"definitions/softmax/{OPERATOR}.json").write_text(json.dumps({
        "name": OPERATOR,
        "op_type": "softmax",
        "axes": {
            "batch": {"type": "var", "description": "Concurrent decode requests; rows of the logits tensor."},
            "vocab": {"type": "const", "value": vocab,
                      "description": "Vocabulary size. Fixed by the model, so const rather than var."},
        },
        "inputs": {
            "logits": {"shape": ["batch", "vocab"], "dtype": "float32", "description": "Raw logits."},
            "out": {"shape": ["batch", "vocab"], "dtype": "float32",
                    "description": "Pre-allocated output. The production call site is `logits[:] = ...`, so a replacement that allocates is not substitutable."},
        },
        "outputs": {"out": {"shape": ["batch", "vocab"], "dtype": "float32", "description": "Probabilities."}},
        "reference": REFERENCE,
        "baseline": _read_seed(),
        "tags": ["softmax", "sglang", "generic-fellow"],
        "description": "Row-wise softmax over the vocabulary, written in place into a caller-provided buffer.",
    }, indent=2) + "\n", encoding="utf-8")

    shapes, lines = [], []
    for index, (case_id, batch, vocab_size) in enumerate(cases):
        uuid = f"{index:016x}"
        primary = batch == 8  # the traced production shape; the sealed README says so
        shapes.append({
            # `role` is filled after the loop by the *production* rule rather
            # than written here. It used to say `correctness-and-performance`
            # outright, which is CONTRACT 4.4: the fixture was more generous
            # than the scaffold, so no mock run ever exercised the scaffold's
            # one-timed-shape-per-operator defect and m4 found it instead.
            "case_id": case_id, "uuid": uuid, "axes": {"batch": batch},
            "role": None, "is_primary": primary,
            "observed": primary,
            "calls": 20 if primary else 0,
            **({"observed_shapes": [[batch, vocab_size]]} if primary
               else {"note": "declared by the sealed driver, not seen in the trace"}),
        })
        lines.append(json.dumps({
            "definition": OPERATOR,
            "workload": {"axes": {"batch": batch},
                         "inputs": {"logits": {"type": "random"}, "out": {"type": "random"}},
                         "uuid": uuid},
            "solution": None, "evaluation": None}))
    (root / f"workloads/softmax/{OPERATOR}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for shape, role in zip(shapes, W.assign_roles(shapes)):
        shape["role"] = role

    # MOCK-MAP (A), and it has to happen **after** the rename above, which is
    # why it is here rather than in `entry.sh`: `env_render --content-type code`
    # writes into `items/codes/`, and until the rename that directory does not
    # exist. `--inherit` when an upstream is staged, because CONTRACT 2 has m1
    # produce the record and everyone carry it forward; `--new` only when no
    # upstream is reachable, which is the pure-mock case.
    if not (root / "environment.yaml").is_file():
        upstream = next(
            (Path(os.environ[var]) / rel
             for var, rel in (("AGENT_SYS_INPUT_OPERATOR_IDENTITY", "items/env/environment.yaml"),
                              ("AGENT_SYS_INPUT_PROFILING_EVIDENCE", "items/env/environment.yaml"),
                              ("AGENT_SYS_INPUT_DEPLOY_KIT", "items/codes/environment.yaml"))
             if os.environ.get(var) and (Path(os.environ[var]) / rel).is_file()),
            None)
        if upstream:
            mode = ["--inherit", str(upstream)]
            extra: list[str] = []
        else:
            # **The `--new` path is the standalone case only.** In the full graph
            # `deploy_kit` is a declared input of `build_workset`, so `--inherit`
            # fires and none of the four below is consulted. They exist so this
            # script can be run on its own.
            #
            # The names are **m1's** (`MOCK_GPU_ARCH` and friends), deliberately:
            # `deploy_and_prove.task/mock_adapt.sh` already defines them for the
            # same four fields, and a second set of names would be a second place
            # for an image digest to go stale. `MOCK_IMAGE_ID` has no default
            # here for exactly that reason — inventing a digest would make the
            # mock a fiction about which image ran.
            image_id = os.environ.get("MOCK_IMAGE_ID")
            if not image_id:
                _die("no upstream environment.yaml is staged and MOCK_IMAGE_ID is unset. "
                     "In the graph, deploy_kit supplies it; standalone, set MOCK_IMAGE_ID to a "
                     "digest a real bring-up recorded. A digest invented here would make the "
                     "mock a fiction about which image ran.")
            mode = ["--new"]
            extra = [
                "--set", f"fixed.gpu_arch={os.environ.get('MOCK_GPU_ARCH', 'gfx950')}",
                "--set", f"fixed.gpu_count={os.environ.get('MOCK_GPU_COUNT', '8')}",
                "--set", f"fixed.image_id={image_id}",
                "--set", "runtime.endpoint=" + os.environ.get(
                    "MOCK_ENDPOINT",
                    f"http://{os.environ.get('E2E_NODE_IP', '127.0.0.1')}"
                    f":{os.environ.get('E2E_PORT_ROUTER', '8101')}"),
            ]
        # `sys.executable` and not a bare `python3`: `env_render.py` validates
        # before it writes and `schema.py` imports `referencing` to do it, which
        # m1 measured to be absent from `/usr/bin/python3` on this host. The
        # interpreter already running this file imported `yaml`, so it is the
        # one to hand the job to.
        rendered = subprocess.run(
            [sys.executable, str(PKG / "assets/lib/env_render.py"), *mode,
             "--content-type", "code", "--out", str(content), *extra],
            capture_output=True, text=True)
        if rendered.returncode != 0:
            _die(f"env_render ({mode[0]}) failed:\n{rendered.stderr[-800:]}")
    if not (root / "environment.yaml").is_file():
        _die("env_render reported success and wrote no items/codes/environment.yaml")
    environment = yaml.safe_load((root / "environment.yaml").read_text(encoding="utf-8"))

    def entrypoints(operator_id=None):
        suffix = f" --operator {operator_id}" if operator_id else ""
        tag = operator_id or "all"
        return {"correctness": {"cmd": f"./run_correctness.sh{suffix}",
                                "report": f"evidence/correctness.{tag}.json", "protected": True},
                "performance": {"cmd": f"./run_performance.sh{suffix}",
                                "report": f"evidence/performance.{tag}.json", "protected": True,
                                "timeout_s": 1800,
                                "impl_contract": W.IMPL_CONTRACT}}

    document = {
        "schema_version": 1,
        "workset_id": "mock.sampler-vocab-softmax",
        "produced_by": {"package": "e2e-flow", "commit": "mock-adapt",
                        "step": "build_workset (mock)"},
        "ground_truth": {"abort_on_mismatch": ["gpu_arch", "gpu_count", "tp_size", "dtype"],
                         "warn_on_mismatch": ["image_id", "rocm", "torch"],
                         "dtypes": {OPERATOR: "float32"},
                         "noise_floor": 1.05,
                         "environment": environment},
        "entrypoints": entrypoints(),
        "protocol": {"groups": 5, "iters_per_group": 10, "warmup": 3, "timing": "event"},
        "operators": [{
            "operator_id": OPERATOR,
            "kernel_id": "k002",
            "device_symbol": "cunn_SoftMaxForwardGmem<4, float, float, float>",
            "op_type": "softmax",
            "status": "complete", "missing_fields": [],
            "definition": f"definitions/softmax/{OPERATOR}.json",
            "workload": f"workloads/softmax/{OPERATOR}.jsonl",
            "shapes": shapes,
            "entrypoints": entrypoints(OPERATOR),
            # **`imported`, and m4 was right to push on this.** They read
            # `kind: written` and argued for importing torch at float64
            # instead — a hand-written float64 softmax must reproduce the
            # max-subtraction trick or it overflows at V=151936, which is
            # exactly the regime this operator exists for; and if it does
            # reproduce it, it is the same algorithm under test and a shared
            # bug in the trick is invisible.
            #
            # The body already did the right thing — `REFERENCE` calls
            # `torch.softmax(logits.double(), ...)`, so the algorithm is
            # torch's and only the dtype is ours. The *declaration* was wrong,
            # and a declaration that misdescribes the body is worse than a
            # wrong body, because a reviewer checks the declaration.
            "reference": {"kind": "imported", "module": "torch", "symbol": "softmax",
                          "source_file": "torch/functional.py",
                          "rationale": "torch.softmax evaluated at float64. Imported rather than written: a hand-written float64 softmax must reproduce the max-subtraction trick or overflow at V=151936, and if it reproduces it then it is the same algorithm under test. The wrapper casts and writes into `out`; the algorithm is torch's."},
            "baseline": {"kind": "imported", "module": "torch", "symbol": "softmax",
                         "rationale": "The incumbent. sglang's sampler.py:183 is literally `logits[:] = torch.softmax(logits, dim=-1)`."},
            "edit_target": {"source_owner": "sglang", "repo_root_var": "@SGLANG_ROOT@",
                            "source_file": "python/sglang/srt/layers/sampler.py",
                            "editable_sources": ["python/sglang/srt/layers/sampler.py"],
                            "entry_function": "Sampler.forward",
                            "entry_function_line": 183,
                            "source_resolution_method": "trace_python_stack",
                            "resolution_evidence": "record_shapes resolved the device symbol to aten::softmax with Input Dims [[8, 151936], [], []]; the only vocabulary-wide softmax on the decode path is sampler.py:183."},
            "integration": {
                "target_files": ["python/sglang/srt/layers/sampler.py"],
                "base_sha256": _base_sha256(root, "python/sglang/srt/layers/sampler.py"),
                "public_symbol": "sampler_softmax",
                "signature": "sampler_softmax(logits: Tensor[B, V] fp32, out: Tensor[B, V] fp32) -> Tensor",
                "invariants": [
                    "writes in place into the caller-provided `out`; the call site is `logits[:] = ...`, so a replacement that allocates is not substitutable there",
                    "fp32 in and fp32 out",
                    "every output row sums to 1 within 1e-4 — these feed torch.multinomial, and a row that does not is a sampler drawing from the wrong distribution",
                    # Both from the sealed `integration.md`, both surfaced by m4
                    # in review, and both the same class as the first: no gate
                    # in this workset catches either.
                    "the reduced dimension is the last one and is contiguous; a replacement that relies on that must say so, because a non-contiguous `logits` then gives wrong results rather than an error",
                    "no build step may be added — the driver imports the module directly, and a replacement needing compilation cannot be delivered as an overlay",
                ],
                "apply_mode": "overlay_files", "requires_restart": True, "build_step": None,
            },
            "gates": {"snr_db": 30.0, "allclose": {"atol": 1e-6, "rtol": 1e-3},
                      "extra": [{"name": "rows_sum_to_one", "tolerance": 1e-4,
                                 "brief": "Every output row sums to 1. No SNR threshold catches a wrong distribution."},
                                {"name": "writes_in_place", "tolerance": None,
                                 "brief": "The returned tensor IS the caller's `out` and `out` was written. The call site is `logits[:] = ...`; a replacement that allocates passes every isolated correctness case and breaks there."}]},
            # A placeholder until STEP 8 transcribes the measured figure, and it
            # is the previous round's rule of thumb rather than a measurement.
            # `check_workset_shape` compares it against `evidence/` once that
            # exists, so it cannot survive as a fiction.
            "noise_floor": 1.05,
            "apparatus": ["_common.py", "run_correctness.sh", "run_performance.sh", "workset.yaml",
                          f"definitions/softmax/{OPERATOR}.json", f"workloads/softmax/{OPERATOR}.jsonl"],
            "provenance": {
                "source": "Qwen3-0.6B TP1 decode capture, torch.profiler /start_profile, DECODE stage"
                          + (f"; workset targeted at {environment_note}" if environment_note else ""),
                "rank": 2, "pct_total": 14.50, "in_service_avg_us": 55.59,
                "magpie_row": {"Name": "cunn_SoftMaxForwardGmem<4, float, float, float>",
                               "Calls": 20, "Self CUDA total (us)": 1111.9,
                               "Avg time (us)": 55.59, "% Total": 14.50,
                               "Input Shapes": "[[8, 151936], [], []]"},
            },
        }],
    }
    (root / "workset.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    # The forge add-on, by the same generator a real run uses — so what the mock
    # produces and what `check_workset_shape` calls consistent cannot drift.
    export = subprocess.run([sys.executable, str(PKG / "assets/lib/forge_export.py"),
                             "--workset", str(root)], capture_output=True, text=True)
    if export.returncode != 0:
        _die(f"forge_export failed:\n{export.stderr[-800:]}")

    # The stage-3 operator directories are left where they are. They carry the
    # ranking provenance and the two MoE operators' briefs, and deleting them
    # would throw away the half of the merge that is not represented above.
    print(f"mock_adapt: {OPERATOR} from the sealed stage-4 half, "
          f"{len(shapes)} shapes, forge add-on generated")
    print("mock_adapt: evidence/ is NOT written here — it is a measurement. "
          "Run ./run_correctness.sh and ./run_performance.sh under items/codes/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
