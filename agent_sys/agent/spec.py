"""The agent spec — design §3.

The *validated* form. The JSON Schema pass in the main design §4 has already
run, so these models restate the shape rather than defend it; what they add is
`extra="forbid"`, which is how criteria 5 and 14 are satisfied structurally —
there is nowhere to put a permission or a runtime configuration knob.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "KNOWLEDGE_TYPES",
    "AgentSpec",
    "BackendDecl",
    "Kind",
    "KnowledgeRef",
    "KnowledgeReport",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Kind(str, Enum):
    """What an agent *is*. Main spec §4.8: every task has an agent, and `kind`
    is what varies.

    **`human` is declarable and unimplemented, deliberately.** A `kind: human`
    spec loads and fails at *selection*, which is the honest outcome: the spec
    is well-formed and this alpha cannot run it. Rejecting it at load would make
    the spec ill-formed, which it is not (design §3.1).
    """

    AI = "ai"
    PROGRAM = "program"
    HUMAN = "human"


#: Spec §3.4's six, as a documented constant used for the coverage report and
#: never as a validator. `knowledge_type` is a `str` because the spec says the
#: list is "extensible … not a closed vocabulary" — an enum would make the
#: seventh type a code change in this module (design D3).
KNOWLEDGE_TYPES: tuple[str, ...] = (
    "few_shot",
    "runnable",
    "official_reference",
    "expert_experience",
    "verifiable_resource",
    "runtime_generated",
)


class BackendDecl(_Model):
    """One declared backend implementation. Design §6.3.

    `backend_entry` takes both forms: a dotted path `pkg.mod:Class`, which names
    *this exact one*, or a bare entry-point name in group `agent_sys.backends`,
    which discovers *what exists*. `err` is the per-entry failure message the
    dotted form buys — fsspec's `"Install adlfs to access Azure Datalake Gen2"`
    rather than a traceback about a missing submodule.
    """

    key: str
    backend_entry: str
    config: dict[str, Any] = Field(default_factory=dict)
    err: str = ""


class KnowledgeRef(_Model):
    """Knowledge is a reference, not content — spec §3.4.

    `kind` is a **handoff kind name**, resolved in the handoff registry at load
    (design §4, check 3). Type 6 is declarable and produced later; whether an
    instance exists is a runtime question and no load check asks it.
    """

    kind: str
    knowledge_type: str
    required: bool = False


class AgentSpec(_Model):
    """Spec §3.1's nine keys.

    **No permissions field, no permissions parameter, anywhere in this package.**
    Permissions live on the task, versioned with it (`task_graph` spec §3.2.2).
    Criterion 5 is satisfied structurally: `extra="forbid"` means there is
    nowhere to put one.
    """

    name: str
    version: str = ""  # maintenance metadata; nothing at runtime reads it
    description: str = ""
    kind: Kind
    backends: list[BackendDecl] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)
    knowledge: list[KnowledgeRef] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    @field_validator("backends", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        """A mapping form is normalised to a list, preserving declaration order.

        Spec §3.1 permits "a list or dict"; design D2 stores a list, because the
        order is load-bearing (criterion 3) and a dict that carries an ordering
        is a dict whose ordering nobody can see at the call site.
        """
        if isinstance(value, Mapping):
            return [{"key": key, **dict(decl)} for key, decl in value.items()]
        return value

    @field_validator("backends")
    @classmethod
    def _keys_unique(cls, value: list[BackendDecl]) -> list[BackendDecl]:
        """matplotlib's entry-point validation, adopted (design §6.3): a key may
        not be duplicated within a spec, and an *identical* duplicate is
        tolerated rather than an error, because duplicates arise from packaging
        outside the declarer's control."""
        seen: dict[str, BackendDecl] = {}
        kept: list[BackendDecl] = []
        for decl in value:
            previous = seen.get(decl.key)
            if previous is None:
                seen[decl.key] = decl
                kept.append(decl)
            elif previous != decl:
                raise ValueError(
                    f"backend key {decl.key!r} is declared twice with different entries: "
                    f"{previous.backend_entry!r} and {decl.backend_entry!r}"
                )
        return kept

    @classmethod
    def of(cls, record: Mapping[str, Any]) -> AgentSpec:
        """Build one from an admitted spec record."""
        return cls.model_validate(dict(record))


class KnowledgeReport(_Model):
    """Design §4.1. **A value, not a log line.**

    The warning is rendered from it and the fatal mode raises from the same
    value, so the two modes cannot drift into disagreeing about what is missing.
    Criterion 2 asserts exactly that.
    """

    spec: str
    missing: list[KnowledgeRef] = Field(default_factory=list)
    types_absent: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """`types_absent` is advisory and never fatal — a spec that declares no
        `runnable` knowledge is not malformed. Only `missing` counts."""
        return not self.missing

    def render(self) -> str:
        parts: list[str] = []
        if self.missing:
            named = ", ".join(f"{ref.kind} ({ref.knowledge_type})" for ref in self.missing)
            parts.append(f"unresolvable knowledge handoff kinds: {named}")
        if self.types_absent:
            parts.append(f"knowledge types not represented: {', '.join(self.types_absent)}")
        return f"agent spec {self.spec!r}: " + "; ".join(parts) if parts else ""


def types_absent(refs: Sequence[KnowledgeRef]) -> list[str]:
    """Which of the six are not represented. Advisory — design §4.1."""
    present = {ref.knowledge_type for ref in refs}
    return [name for name in KNOWLEDGE_TYPES if name not in present]
