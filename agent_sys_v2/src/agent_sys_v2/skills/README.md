# Infera performance skills

Procedures and scripts an agent_sys agent reads when a task calls for them. Each `<name>/SKILL.md` starts with the
frontmatter of a Claude Code skill (`name`, `description`); agent_sys injects one index line per skill into every
run's system.md (`project.yaml` `skills:` picks which), and the agent opens the SKILL.md it needs.

| Skill | For |
|---|---|
| `infera-deploy` | Bring a kit up or down, audit the configuration that is really live, guard the run |
| `infera-bench` | Measure one point and write it as an `infera.result/v1` file for the ledger |
| `infera-quality-gate` | Prove the deployment's output is right before any performance claim |
| `infera-profile` | A round's profile: cheap layers first, kernels of the bottleneck layer only; `candidates.yaml` for the plan; engine playbooks |
| `infera-diagnose` | Classify the bottleneck from metrics, logs, GPU traces and CPU profiles; pick candidate levers |
| `infera-experiment` | One-variable A/B: prove the lever is live, judge with `scripts/ab_judge.py`, record |
| `infera-hygiene` | Shared-node manners and the traps of the cluster (VRAM release, foreign containers, NFS) |

Rules for writing one: determinism goes into scripts, judgement stays with the agent; every procedure says how to
prove that what it changed is really in effect; point at the repository's own scripts instead of copying them.
