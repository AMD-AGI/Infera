"""RFC 6901 addressing into a handoff's content, with three-way failure.

Spec §5.1 rev. 5 is a Pointer and not a jsonpath, and the reason is a property
of the standard: RFC 9535 §2.5.1.2 **forbids** a syntactically valid JSONPath
query from erroring, so no JSONPath implementation can distinguish "the path is
wrong" from "the value is absent" — the caller gets an empty result either way,
which is the silent pass this system exists to prevent.

The library is `python-jsonpath`'s `JSONPointer`, the only one of six measured
that separates all three outcomes (`design.md` §8.4).
"""

from __future__ import annotations

import re
from typing import Any

from jsonpath import JSONPointer
from jsonpath.exceptions import JSONPointerError, JSONPointerResolutionError

from handoff.errors import PointerInvalid, PointerMiss

__all__ = ["resolve"]

#: RFC 6901 §3: `~` is only ever followed by `0` or `1`. `python-jsonpath`
#: accepts `~2` at construction and fails it at *resolve* as a missing key,
#: which would report an author's typo as absent content — measured on 2.2.1.
#: Three lines of pre-check keep the two outcomes apart.
_BAD_ESCAPE = re.compile(r"~(?![01])")


def resolve(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer into a document.

    Three outcomes, three answers:

    | | |
    |---|---|
    | malformed pointer | `PointerInvalid` — the binding author's typo |
    | addresses nothing | `PointerMiss` — the content is not what the binding expected |
    | the value is JSON `null` | returns `None`, and it is **not** a miss |

    Two exception types rather than one, for the reason `docs/design.md` §6.2
    separates `SpecNotFound` from `SpecInconsistent`: a validator must be able
    to treat "I was written wrong" and "the artefact is wrong" differently.

    The empty pointer `""` addresses the whole document, and `~0`/`~1` address
    a key containing `~` or `/` — which matters because §3.3's `items` keys can
    be agent-generated and may contain any byte but `/`.
    """
    if _BAD_ESCAPE.search(pointer):
        raise PointerInvalid(
            f"{pointer!r}: '~' must be followed by '0' or '1' (RFC 6901 §3); "
            f"'~0' is a literal '~' and '~1' a literal '/'"
        )
    try:
        ptr = JSONPointer(pointer)
    except JSONPointerError as exc:
        raise PointerInvalid(f"{pointer!r}: {exc}") from exc

    try:
        return ptr.resolve(doc)
    except JSONPointerResolutionError as exc:
        raise PointerMiss(f"{pointer!r} addresses nothing: {exc}") from exc
