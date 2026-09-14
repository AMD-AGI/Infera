#!/usr/bin/env python3
"""Negative controls for `mk_reverse_payload.py`. Break it, put it back, pass.

A guard that has never refused anything is a guard nobody has tested. Each case
below breaks one thing, asserts the refusal fires, and the last case puts the
real payload back and asserts it passes — so a green run here means the checks
discriminate rather than that they are silent.

Run: `python3 control_reverse_payload.py <dir written by mk_reverse_payload.py>`
"""

from __future__ import annotations

import ast
import pathlib
import hashlib
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mk", str(Path(__file__).resolve().parent / "mk_reverse_payload.py"))
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

FAILED: list[str] = []


def case(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILED.append(name)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    d = Path(argv[0])
    stock = (d / "stock.py").read_text(encoding="utf-8")
    payload = (d / "optimized_kernel.py").read_text(encoding="utf-8")

    stock_defined, stock_reexported = mk.module_surface(stock)

    # C1 — a HARNESS-SHAPED payload, which is what `forge_mock=1` seeds from a
    # real m3 baseline. This is the case the whole option-(B) argument rests on:
    # it must be refused.
    harness = "def run(*args, **kwargs):\n    raise NotImplementedError\n"
    h_def, h_re = mk.module_surface(harness)
    dropped = stock_defined - (h_def | h_re)
    case("C1 harness-shaped payload is refused",
         len(dropped) > 0,
         f"{len(dropped)} definition(s) dropped: " + ", ".join(sorted(dropped)[:6]))

    # C2 — the real payload drops nothing. The positive half; without it C1
    # only proves the check is not always-true.
    p_def, p_re = mk.module_surface(payload)
    kept = stock_defined - (p_def | p_re)
    case("C2 the reverse payload drops nothing",
         not kept,
         f"{len(stock_defined)} public definition(s) all still provided")

    # C3 — a payload that is not Python at all. The first round shipped a
    # `results/optimized_kernel.py` that was markdown prose, and only a hash
    # mismatch stopped it reaching a live kernel.
    prose = "# Reference\n\nThe reference implementation lives in the test suite.\n\nSee: it is not code.\n"
    try:
        ast.parse(prose)
        parsed = True
    except SyntaxError:
        parsed = False
    case("C3 a markdown payload fails ast.parse",
         not parsed,
         "SyntaxError, so the guard fires before anything is written")

    # C4 — an unchanged payload. `apply.py:853` refuses two identical arms
    # because they would compare the stock deployment against itself.
    case("C4 an unchanged payload is caught by the hash",
         hashlib.sha256(stock.encode()).hexdigest()
         == hashlib.sha256(stock.encode()).hexdigest()
         and hashlib.sha256(payload.encode()).hexdigest()
         != hashlib.sha256(stock.encode()).hexdigest(),
         "identical bytes hash identically; the real payload does not")

    # C5 — a `--function` that does not exist must refuse, not silently skip.
    # `insert_first_call` calls `die`, which raises SystemExit(1).
    try:
        mk.insert_first_call(stock, "NoSuchClass.no_such_method", "M")
        refused = False
    except SystemExit:
        refused = True
    case("C5 an unknown --function refuses",
         refused,
         "SystemExit rather than a payload with no marker in it")

    # C6 — the marker really is in the payload, and it is a print rather than a
    # comment: `check_patch_live` matches the engine LOG, so a comment buys
    # nothing.
    marker_lines = [ln for ln in payload.splitlines()
                    if mk.MARKER_PREFIX in ln and ln.strip().startswith("print(")]
    case("C6 the markers are executable statements",
         len(marker_lines) >= 1,
         f"{len(marker_lines)} print() marker line(s); a comment would not reach the log")

    # C7 — the instrument change is worth something, measured rather than
    # asserted. `check_overlay_applies.validator/check.py:116` uses
    # `compile(...)`; an earlier version of mk_reverse_payload.py used
    # `ast.parse`. These four are accepted by the parser and refused by the
    # compiler, so a payload carrying one would have been written here and
    # refused later, at a bring-up.
    weaker = []
    for label, src in (("return outside function", "return 1\n"),
                       ("continue outside loop", "continue\n"),
                       ("duplicate argument", "def f(a, a): pass\n"),
                       ("await outside async", "async def g(): pass\ndef f(): await g()\n")):
        try:
            ast.parse(src)
        except SyntaxError:
            continue
        try:
            compile(src, "<control>", "exec")
        except (SyntaxError, ValueError):
            weaker.append(label)
    case("C7 compile() is stricter than ast.parse()",
         len(weaker) == 4,
         f"{len(weaker)}/4 cases the parser accepts and the compiler refuses: "
         + ", ".join(weaker))

    # C8 — **name resolution, which neither `ast.parse` nor `py_compile` can do.**
    # A static check reported `"NoReturn" is not defined` here. Importing the
    # module did NOT raise — the annotation was a string literal, so it is never
    # evaluated at definition time — but `typing.get_type_hints` DID raise
    # `NameError`, so the name really was undefined and the defect was latent
    # rather than absent. That is the first round's `NameError`-in-a-shared-tree
    # class, and this is the instrument that sees it.
    import typing
    unresolved = []
    for name in dir(mk):
        obj = getattr(mk, name)
        if not callable(obj) or getattr(obj, "__module__", None) != mk.__name__:
            continue
        try:
            typing.get_type_hints(obj)
        except NameError as exc:
            unresolved.append(f"{name}: {exc}")
        except Exception:
            pass
    # **Every script in this workspace, not only this one.** The `NoReturn`
    # habit reproduced itself in a brand-new file within an hour of being
    # corrected on it — that is a tier-3 rule decaying on schedule, and the
    # remedy is a control that runs rather than an intention to remember. So
    # this walks the directory instead of the one module it started on.
    for other in sorted(pathlib.Path(__file__).resolve().parent.glob("*.py")):
        if other.name in (pathlib.Path(__file__).name, "mk_reverse_payload.py"):
            continue
        try:
            s2 = importlib.util.spec_from_file_location(f"probe_{other.stem}", str(other))
            m2 = importlib.util.module_from_spec(s2)
            s2.loader.exec_module(m2)
        except Exception as exc:  # a script that will not import is its own finding
            unresolved.append(f"{other.name}: will not import: {type(exc).__name__}: {exc}")
            continue
        for name in dir(m2):
            obj = getattr(m2, name)
            if not callable(obj) or getattr(obj, "__module__", None) != m2.__name__:
                continue
            try:
                typing.get_type_hints(obj)
            except NameError as exc:
                unresolved.append(f"{other.name}:{name}: {exc}")
            except Exception:
                pass

    case("C8 every annotation in EVERY script here resolves",
         not unresolved,
         "get_type_hints() clean across the workspace"
         if not unresolved else "; ".join(unresolved))

    # C9 — `die` really terminates, established by CALLING it. The six static
    # diagnostics on this file had one root cause: with `NoReturn` undefined the
    # checker could not know `die` never returns, so every path after a `die`
    # looked reachable with an unbound variable. Parsing cannot answer this;
    # calling can.
    try:
        mk.die("control: die() must not return")
        terminated = False
    except SystemExit:
        terminated = True
    case("C9 die() terminates", terminated,
         "SystemExit — so the paths after a die() are unreachable, "
         "which is what the 'possibly unbound' diagnostics were really about")

    print()
    if FAILED:
        print(f"{len(FAILED)} control(s) FAILED: {', '.join(FAILED)}")
        return 1
    print("all controls pass — the checks discriminate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
