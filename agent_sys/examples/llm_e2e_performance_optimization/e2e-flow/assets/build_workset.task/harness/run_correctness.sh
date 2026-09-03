#!/usr/bin/env python3
"""One-click correctness over every operator and every shape (M3.7.4).

    ./run_correctness.sh [--operator ID] [--shape CASE_ID] [--impl PATH] [--json OUT]

Exit 0 iff every gate on every shape passed. **Protected**: a consumer may not
edit this file or `_common.py`; `check_workset_shape` compares both against the
package copy. The argument is in `_common.py`'s header and it is short — an
agent that writes its own oracle controls its own result.

`.sh` and a Python shebang: the entrypoint's *name* is contract (it is written
into `workset.yaml`), its language is not, and a shell wrapper around a Python
program is one more file to keep in step for nothing.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import Ctx, build_inputs, finish, load_definition, load_impl, setup  # noqa: E402


def _snr_db(got, want) -> float:
    """Signal-to-noise in dB, computed in float64 against the reference.

    Reported rather than a bare pass/fail because the margin is the useful part:
    a kernel at 31 dB against a 30 dB gate is one regression away from failing,
    and nothing else in the pipeline would say so.
    """
    import torch

    a, b = got.to(torch.float64), want.to(torch.float64)
    noise = (a - b).pow(2).sum().item()
    signal = b.pow(2).sum().item()
    if noise == 0:
        return float("inf")
    if signal == 0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


#: Filled by the caller immediately around the call under test, because these
#: two facts cannot be recovered from the returned tensor afterwards.
_WITNESS: dict = {}


def _extra(name: str, got, tolerance) -> bool:
    """The operator-specific invariants a Definition declares in `gates.extra`.

    Deliberately a small closed set rather than an eval hook: a gate the workset
    can define in arbitrary code is a gate an optimiser can define to be true.
    An invariant this table does not carry is a change to this file, reviewed.
    """
    import torch

    if name == "rows_sum_to_one":
        # Not decoration. These probabilities feed `torch.multinomial`, so a row
        # that does not sum to 1 is a sampler drawing from the wrong
        # distribution — which no SNR threshold catches.
        return bool(torch.allclose(got.sum(-1), torch.ones_like(got.sum(-1)), atol=float(tolerance or 1e-4)))
    if name == "writes_in_place":
        # **The invariant with no other gate**, and both m4 and m5 named it
        # independently. The production call site is `logits[:] = softmax(...)`,
        # so a replacement that *allocates* its own output is not substitutable
        # there — and it passes every correctness case run in isolation, because
        # in isolation nobody looks at which buffer the answer landed in.
        #
        # Checked by identity and by mutation, both: `returned is out` alone
        # would pass an implementation that returns the buffer without writing
        # it, and "out changed" alone would pass one that writes the buffer and
        # returns a fresh tensor. m5 gates on this; as prose it was only ever a
        # note for a human.
        return bool(got is not None and _WITNESS.get("returned_is_out") and _WITNESS.get("out_mutated"))
    if name == "no_nan":
        return not bool(torch.isnan(got).any() or torch.isinf(got).any())
    if name == "non_negative":
        return bool((got >= 0).all())
    raise SystemExit(f"gates.extra names {name!r}, which this harness does not implement. "
                     f"Known: rows_sum_to_one, no_nan, non_negative.")


def main() -> int:
    ctx = setup("correctness")
    every = True
    for operator in ctx.operators:
        definition = load_definition(operator)
        gates = operator["gates"]
        row = {"operator_id": operator["operator_id"], "ran": False, "failure": None,
               "passed": False, "gates": gates, "shapes": []}
        try:
            reference = load_impl(operator, definition, None, "reference")
            under_test = load_impl(operator, definition, ctx.args.impl, "baseline")
        except Exception as error:  # noqa: BLE001 — the artefact under test may raise anything
            row["failure"] = f"could not load an implementation: {error!r}"
            ctx.report["operators"].append(row)
            every = False
            continue

        row["ran"] = True
        for shape in ctx.shapes(operator):
            entry = {"case_id": shape["case_id"], "uuid": shape["uuid"],
                     "snr_db": None, "allclose": None, "extra": {}, "passed": False, "failure": None}
            try:
                inputs = build_inputs(definition, shape)
                want = reference(**inputs)
                probe = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in inputs.items()}
                # Witness the destination buffer across the call, for
                # `writes_in_place`. A sentinel rather than the original
                # contents: an implementation that happens to write the same
                # values back would otherwise read as not having written.
                destination = probe.get("out")
                _WITNESS.clear()
                if destination is not None and hasattr(destination, "fill_"):
                    destination.fill_(float("-inf"))
                    before = destination
                    got = under_test(**probe)
                    import torch

                    _WITNESS["returned_is_out"] = got is before
                    _WITNESS["out_mutated"] = not bool(torch.isinf(before).all())
                else:
                    got = under_test(**probe)
                entry["snr_db"] = _snr_db(got, want)
                checks = [entry["snr_db"] >= float(gates["snr_db"])]
                if gates.get("allclose"):
                    import torch

                    entry["allclose"] = bool(torch.allclose(
                        got.to(torch.float64), want.to(torch.float64),
                        atol=float(gates["allclose"].get("atol", 1e-6)),
                        rtol=float(gates["allclose"].get("rtol", 1e-3))))
                    checks.append(entry["allclose"])
                for gate in gates.get("extra") or []:
                    ok = _extra(gate["name"], got, gate.get("tolerance"))
                    entry["extra"][gate["name"]] = ok
                    checks.append(ok)
                entry["passed"] = all(checks)
                if not entry["passed"]:
                    entry["failure"] = (f"SNR {entry['snr_db']:.2f} dB against a {gates['snr_db']} dB gate"
                                        if entry["snr_db"] < float(gates["snr_db"])
                                        else "an allclose or extra gate failed")
            except Exception as error:  # noqa: BLE001
                entry["failure"] = repr(error)
            row["shapes"].append(entry)
            every = every and entry["passed"]

        row["passed"] = bool(row["shapes"]) and all(s["passed"] for s in row["shapes"])
        if not row["passed"] and not row["failure"]:
            row["failure"] = "; ".join(s["failure"] for s in row["shapes"] if s["failure"])[:300]
        ctx.report["operators"].append(row)

    ctx.report["passed"] = every and bool(ctx.report["operators"])
    finish(ctx, ctx.report["passed"])


if __name__ == "__main__":
    main()
