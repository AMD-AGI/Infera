"""What leaves `agent/` — `docs/interfaces.md` §4.4.

Two interface levels, kept apart as two protocols rather than one with holes in
it: level 1 is what a task runner talks to and every executor satisfies it;
level 2 is the AI-harness abstraction and only an AI executor has one.

**`backends/claude_sdk.py` is not imported here, and must never be.** The SDK is
a 376 MB extra costing ~1.3 s to import; a missing extra is a
`BackendUnsupported` naming it, not an `ImportError` at start-up.
"""

from agent.backend import (
    TERMINAL,
    AgentBackend,
    AgentHistory,
    AgentResult,
    AgentStatus,
    Assignment,
    BackendUnsupported,
    ConfinementNotApplied,
    Executor,
    ExecutorBase,
)
from agent.gate import GateFailure, run_gate
from agent.registry import AgentSpecRegistry, KnowledgeWarning
from agent.runner import (
    RESUMED,
    WOKEN,
    MonitorUnresolved,
    Runner,
    TaskAttempt,
    ThreadAlreadyHeld,
)
from agent.selection import BackendUnavailable, Rejection, Selection, select_backend
from agent.spec import (
    KNOWLEDGE_TYPES,
    AgentSpec,
    BackendDecl,
    Kind,
    KnowledgeRef,
    KnowledgeReport,
)
from agent.validator_executor import ValidatorExecutor, ValidatorExecutorUnconfigured

__all__ = [
    "KNOWLEDGE_TYPES",
    "RESUMED",
    "TERMINAL",
    "WOKEN",
    "AgentBackend",
    "AgentHistory",
    "AgentResult",
    "AgentSpec",
    "AgentSpecRegistry",
    "AgentStatus",
    "Assignment",
    "BackendDecl",
    "BackendUnavailable",
    "BackendUnsupported",
    "ConfinementNotApplied",
    "Executor",
    "ExecutorBase",
    "GateFailure",
    "Kind",
    "KnowledgeRef",
    "KnowledgeReport",
    "KnowledgeWarning",
    "MonitorUnresolved",
    "Rejection",
    "Runner",
    "Selection",
    "TaskAttempt",
    "ThreadAlreadyHeld",
    "ValidatorExecutor",
    "ValidatorExecutorUnconfigured",
    "run_gate",
    "select_backend",
]
