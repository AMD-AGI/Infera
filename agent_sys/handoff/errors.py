"""The error vocabulary.

Five of the six classes are **imported** from `protocols.py` rather than
redeclared. A second class of the same name is the failure this package would
be least able to see: `validator` catching `handoff.protocols.Malformed` would
not catch a `handoff.errors.Malformed`, and every test in this package would
still pass.

`BindingConflict` is the one class declared here. It is raised across the seam
by the closure pass (§8.3) and subclasses `SpecInconsistent`, so a caller that
knows only `spec_loader`'s three error classes still catches it.
"""

from __future__ import annotations

from handoff.protocols import (
    DigestMismatch,
    Malformed,
    NotContained,
    PointerInvalid,
    PointerMiss,
)
from spec_loader.protocols import SpecInconsistent

__all__ = [
    "BindingConflict",
    "DigestMismatch",
    "Malformed",
    "NotContained",
    "NotSealable",
    "PointerInvalid",
    "PointerMiss",
]


class NotSealable(RuntimeError):
    """`seal` was asked for a version that is not open to it.

    **Deliberately not a `Malformed`, and the distinction is the caller's to
    act on.** `Malformed`'s own docstring is *"content that does not satisfy
    its kind"*; these refusals say nothing about content at all. They say the
    call could not have succeeded.

    | raised | means | the caller should |
    |---|---|---|
    | `Malformed` | the agent wrote something and it is not a handoff | **catch it.** The attempt produced nothing publishable, which is a fact about the producer and belongs in the record |
    | `NotSealable` | there is no such version, or it is already published | **let it escape.** Nothing is wrong with the artefact; the seal was wired wrong |

    **Found by a caller before it was written, which is why it exists.**
    `agent` was about to wrap `seal` in `except Malformed` so an unpublishable
    attempt could be reported through the gate — correct, and with one
    exception type it would also have swallowed *already published*. That is
    the re-run case: `agent/runner.py:621` re-runs the body inside one attempt
    after a gate failure, against a grant pinned to the same `v<N>`, so a
    second `seal` refuses. Swallowed, the loop would look like it worked while
    the second body's output was discarded — **silent, and only on retries.**
    """


class BindingConflict(SpecInconsistent):
    """A handoff kind and a validator disagree about their binding.

    Spec §5.1: the binding is recorded on both sides and a mismatch crashes at
    load rather than picking a winner. `design.md` §8.3 fixes the seven fields
    of the message, and `registry.py` builds it — the exception carries the
    two names so a caller can report them without parsing prose.
    """

    def __init__(self, kind: str, validator: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.validator = validator
