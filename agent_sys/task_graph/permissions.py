"""What a task's executor may reach — carried here, interpreted in `env_mgr`.

`task_graph` never resolves a path, compares a prefix, or decides containment.
The precedent is inside this module: `Task.resources` is `dict[str, float]` keyed
by pool *name* and the scheduler never learns what a GPU is. Permissions are the
same shape one level up, and carrying them here is what lets the type exist
without `task_graph` and `env_mgr` importing each other.

`Grant.kind` is a handoff **kind name**, never an id. A grant is written at
declaration time, where no instance exists — `HandoffId('trace')` raises — so the
declared name plus a resolution step is the only shape that works. That is also
the shape every surveyed permission model has: Kubernetes RBAC references no UID
anywhere, Dagster carries a stringified `asset_key`, Android lint compares bare
permission names.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict

__all__ = ["Access", "Grant", "Permissions"]


class Access(str, Enum):
    """What an author *declared*. `env_mgr.Mode` is what the kernel gets."""

    READ = "read"
    WRITE = "write"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


class Grant(_Frozen):
    path: str = ""  # opaque here; env_mgr resolves it
    access: Access = Access.READ
    kind: str | None = None  # a handoff KIND NAME, never an id


class Permissions(_Frozen):
    grants: tuple[Grant, ...] = ()

    def covers(self, kind: str, access: Access) -> bool:
        """Does a declared grant name this handoff kind, for this access?

        A lookup over declared entries and nothing more — which is what
        `closure`'s load check 6 asks and the whole of what is owed. A WRITE
        grant implies READ; the reverse does not hold.
        """
        for grant in self.grants:
            if grant.kind != kind:
                continue
            if grant.access is access or grant.access is Access.WRITE:
                return True
        return False
