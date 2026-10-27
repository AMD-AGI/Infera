"""Skills (design §5.7): procedures and scripts an agent reads when it needs them, offered through an index.

  agent_sys_v2/skills/<name>/SKILL.md          shipped with agent_sys; project.yaml `skills: [name, ...]` picks some
                                            (no `skills:` = all of them)
  projects/<p>/skills/<name>/SKILL.md      the project's own, always offered (in place of a shipped one of its name)

SKILL.md starts with the frontmatter of a Claude Code skill (name, description). The index goes into system.md as one
line per skill, like a knowledge index: the agent opens SKILL.md when a task calls for it, so a skill costs a line of
context until it is used.
"""

from __future__ import annotations

from pathlib import Path

from .model import plain, read_task_file

SHIPPED = Path(__file__).resolve().parent / "skills"


def read_skill(path):
    """{name, description, path} of a SKILL.md; None if it has no description (not a skill yet)."""
    try:
        fm, _body = read_task_file(path)
    except ValueError:
        return None
    description = " ".join(str(fm.get("description") or "").split())
    if not description:
        return None
    return {
        "name": str(fm.get("name") or path.parent.name),
        "description": description,
        "path": str(path),
    }


def found(root):
    return [s for p in sorted(Path(root).glob("*/SKILL.md")) if (s := read_skill(p))]


def wanted(project):
    """project.yaml `skills:` as a list of names; None (no `skills:`) for all of them."""
    value = plain(project.config.get("skills"))
    return (
        None if value is None else [str(x) for x in (value if isinstance(value, list) else [value])]
    )


def available(project):
    """The skills offered to the tasks of project: the shipped ones it picks, in its order, then its own (one of its
    own takes the place of a shipped one of the same name)."""
    if not project:
        return []
    own = found(project.path / "skills")
    shipped = [s for s in found(SHIPPED) if s["name"] not in {o["name"] for o in own}]
    names = wanted(project)
    if names is not None:
        shipped = sorted(
            (s for s in shipped if s["name"] in names), key=lambda s: names.index(s["name"])
        )
    return shipped + own


def unknown(project):
    """The names in project.yaml `skills:` that neither agent_sys nor the project has."""
    have = {s["name"] for s in found(SHIPPED) + found(project.path / "skills")}
    return [name for name in wanted(project) or [] if name not in have]


def index(project):
    """The index text: one line per skill."""
    return "\n".join(
        f"- {s['name']} — {s['description']} `{s['path']}`" for s in available(project)
    )


def source(project):
    """Where the skills of the index come from: their directories, comma separated."""
    return ", ".join(dict.fromkeys(str(Path(s["path"]).parents[1]) for s in available(project)))
