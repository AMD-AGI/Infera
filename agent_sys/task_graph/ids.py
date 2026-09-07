"""Typed identities.

A ``TaskId`` and a ``HandoffId`` built from the same bytes are different values.
``typing.NewType`` would give that statically and erase at runtime, so the two
would still compare equal and collide in one dict; subclassing ``uuid.UUID``
gives both, and generation, parsing and formatting come from the stdlib.
"""

import uuid
from typing import Any, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

__all__ = ["Id", "TaskId", "AgentId", "HandoffId"]

# `typing.Self` is 3.11; the repository targets 3.10.
_S = TypeVar("_S", bound="Id")


class Id(uuid.UUID):
    """A UUID that is not interchangeable with a UUID of another kind.

    **Public, and it was not always.** `monitor.record.EventId` subclasses it —
    a fourth id class here would have made this package carry a monitor concept,
    which `engineer_principle.md` §2 forbids, so the edge is right and only the
    name was wrong: a leading underscore says "named in one package" (§1.2) and
    it was named in two. Ruled public in `interfaces.md` §4.7.

    A subclass gets three things and none of them is obvious:

    - `__eq__` compares `type(other) is type(self)`, so two kinds built from one
      UUID are unequal in both directions;
    - `__hash__` keys on `type(self).__name__` and the int, so they land in
      different buckets. Two id classes sharing a `__name__` across modules
      would share a bucket — a collision, not a correctness bug, since equality
      still discriminates;
    - `__get_pydantic_core_schema__` is inherited and `_coerce` is bound to
      `cls`, so a subclass round-trips through a pydantic model as itself.
      Measured against `monitor.EventId`, not assumed.

    **Its shape does not change without messaging `monitor` first.**
    """

    __slots__ = ()

    @classmethod
    def new(cls: type[_S]) -> _S:
        return cls(uuid.uuid4().hex)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self.int == other.int

    def __hash__(self) -> int:
        # Overriding __eq__ alone sets __hash__ to None. The kind is part of the
        # hash so two kinds sharing bytes land in different buckets.
        return hash((type(self).__name__, self.int))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    @classmethod
    def _coerce(cls: type[_S], value: Any) -> _S:
        if isinstance(value, cls):
            return value
        if isinstance(value, uuid.UUID):
            return cls(value.hex)
        return cls(str(value))  # UUID.__init__ rejects a malformed one

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Required, not optional: pydantic raises PydanticSchemaGenerationError
        # on a UUID subclass it has no schema for. `when_used="json"` keeps
        # model_dump() returning real id objects while mode="json" gives strings.
        return core_schema.no_info_plain_validator_function(
            cls._coerce,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema(), when_used="json"
            ),
        )


class TaskId(Id):
    __slots__ = ()


class AgentId(Id):
    __slots__ = ()


class HandoffId(Id):
    __slots__ = ()
