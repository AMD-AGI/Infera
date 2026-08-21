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

__all__ = ["TaskId", "AgentId", "HandoffId"]

# `typing.Self` is 3.11; the repository targets 3.10.
_S = TypeVar("_S", bound="_Id")


class _Id(uuid.UUID):
    """A UUID that is not interchangeable with a UUID of another kind."""

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


class TaskId(_Id):
    __slots__ = ()


class AgentId(_Id):
    __slots__ = ()


class HandoffId(_Id):
    __slots__ = ()
