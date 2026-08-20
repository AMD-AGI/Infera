"""The agent collection.

Two of them: the **spec table** of what kinds of agent exist, and the
**instances** of what has been created. Only the instances persist — the spec
table is configuration, and restoring it would resurrect a spec the operator
has since removed.
"""

from typing import Any

from task_graph.ids import AgentId, TaskId
from task_graph.models import Agent
from task_graph.registry import Registry

__all__ = ["AgentMgr"]

KIND = "agent"


class AgentMgr:
    def __init__(self, registry: Registry) -> None:
        self._r = registry
        self._specs: dict[str, dict[str, Any]] = {}
        self._agents: dict[AgentId, Agent] = {}

    # ---- the spec table ----

    def register(self, spec: str, **config: Any) -> None:
        self._specs[spec] = dict(config)

    def specs(self) -> list[str]:
        return list(self._specs)

    def is_registered(self, spec: str) -> bool:
        return spec in self._specs

    # ---- the instance collection ----

    def instantiate(self, spec: str, task_id: TaskId) -> Agent:
        """Mint an Agent with a fresh AgentId, bound to that task, and keep it.

        A fresh id every call is forced by criterion 21: after a resume the
        stack top must report a different agent than the entry beneath it.
        """
        if spec not in self._specs:
            raise KeyError(f"unknown agent spec {spec!r}; registered: {sorted(self._specs)}")
        return self._mint(spec, task_id)

    def get(self, ref: AgentId | str) -> Agent:
        """By id: that instance. By spec name: a new unbound one.

        The second form is the `get(name) -> agent` the task definition asks for.
        Dispatch calls `instantiate` explicitly — relying on the overload there
        would make "a new agent is created here" invisible at the call site.
        """
        if isinstance(ref, AgentId):
            try:
                return self._agents[ref]
            except KeyError:
                raise KeyError(f"no agent {ref}") from None
        if ref not in self._specs:
            raise KeyError(f"unknown agent spec {ref!r}; registered: {sorted(self._specs)}")
        return self._mint(ref, None)

    def by_spec(self, spec: str) -> list[Agent]:
        return [a for a in self._agents.values() if a.spec == spec]

    def by_task(self, tid: TaskId) -> list[Agent]:
        return [a for a in self._agents.values() if a.task_id == tid]

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def retire(self, aid: AgentId) -> None:
        self.get(aid)
        del self._agents[aid]
        self._store.delete(KIND, str(aid))

    # ---- persistence ----

    def persist(self, aid: AgentId) -> None:
        """Write an agent back after it appended to its `handoffs`."""
        self._store.update(KIND, str(aid), self.get(aid).model_dump(mode="json"))

    def resume_system(self) -> None:
        """Reload instances. The spec table is NOT restored.

        A restored instance is a record, not a live agent: nothing resumes the
        agent's own process and nothing dispatches against one. Restoration
        exists so the audit trail resolves.
        """
        self._agents = {
            agent.id: agent
            for agent in (Agent.model_validate(r) for r in self._store.read_all(KIND))
        }

    # ---- internals ----

    def _mint(self, spec: str, task_id: TaskId | None) -> Agent:
        agent = Agent(spec=spec, task_id=task_id, config=dict(self._specs[spec]))
        self._agents[agent.id] = agent
        self._store.create(KIND, str(agent.id), agent.model_dump(mode="json"))
        return agent

    @property
    def _store(self):
        return self._r.get("store_mgr")
