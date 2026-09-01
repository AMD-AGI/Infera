"""Tuning agent: LLM-driven search for an optimal parallelization
configuration of a workload on a target cluster.

The agent uses the projection tool as an oracle -- memory and performance
projection, simulated by default so the search needs no GPU, with
``--profiling-mode benchmark`` to anchor instead on what fits on a sub-node run
and scale the rest analytically -- and a DSPy.RLM loop with planner +
scratchpad + history to navigate the space efficiently.

Entry point:
    python -m infera.projection.agents.tuning_agent --workload <yaml> --target-cluster <yaml>
"""

from .config import AgentConfig, OptimizationConfig, TargetCluster, load_config
from .workload import ArchitectureRecord, resolve_workload

__all__ = [
    "AgentConfig",
    "TargetCluster",
    "OptimizationConfig",
    "load_config",
    "ArchitectureRecord",
    "resolve_workload",
]
