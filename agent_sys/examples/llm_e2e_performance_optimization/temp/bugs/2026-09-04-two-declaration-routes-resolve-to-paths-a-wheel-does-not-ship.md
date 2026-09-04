# Two declaration routes resolve to repository paths a wheel does not ship

**Found:** 2026-09-04 by `pkg-impl` during the PR-155 `components/` →
`agent_plugins/` rename, while checking whether the `git mv` could break
packaging. It could not — because neither name was ever packaged.
**Severity:** two documented declaration routes — `recipes: [<bare name>]` and
`agent_plugins: [<name>]` — cannot resolve from a wheel install. **Both fail
loudly**, as a named `PrepareRefused`, so nothing is silently wrong; the run
refuses. From a git checkout, which is how everything runs today, both work.
**Status:** reported, **not fixed**. Pre-existing and older than the rename.

## What was measured, and how

A wheel was built from `agent_sys/pyproject.toml` and its members listed. This
is the artefact that `pyproject.toml` produces; whether anyone installs it that
way today is a separate question this record does not answer.

```
python3 -m pip wheel --no-deps --no-build-isolation -w <dir> .
```

`agent_sys_helper-0.1.0-py3-none-any.whl`, **149 members**. Every non-`.py`
member, in full:

```
agent/protocols.pyi        closure/protocols.pyi      env_mgr/protocols.pyi
handoff/protocols.pyi      monitor/protocols.pyi      spec_loader/protocols.pyi
validator/protocols.pyi
spec_loader/schemas/{_common,agent,closure,handoff,task,validator}.schema.json
agent_sys_helper-0.1.0.dist-info/{METADATA,WHEEL,entry_points.txt,top_level.txt,RECORD}
```

So: **no `agent_plugins/` member, and no `.yaml` member of any kind.** The
schemas ship only because they have an explicit `package-data` entry.

## The two routes

### 1. `agent_plugins: [<name>]`

`env_mgr/agent_assets.py`:

```python
AGENT_PLUGINS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_plugins")
```

From a wheel that is `<site-packages>/agent_plugins`, which does not exist.
`_agent_plugin_trees` then raises:

> `agent 'x' declares component 'envchk-baseline' and '<...>/agent_plugins/envchk-baseline' does not exist. 'agent_plugins:' takes a bare name under agent_sys/agent_plugins/, never a path`

The message is accurate about the rule and misleading about the cause: it tells
the reader they wrote a path when in fact they wrote a correct bare name and the
directory was never installed.

### 2. `recipes: [<bare name>]` — the worse one, and it is not in the rename's blast radius

`_recipe_paths` (`env_mgr/agent_assets.py`) tries a package-relative path first,
then:

```python
shipped = os.path.join(os.path.dirname(__file__), "recipes")
candidates.append(os.path.join(shipped, f"{os.path.basename(declared)}.yaml"))
```

`<site-packages>/env_mgr/recipes/serena.yaml` — and the wheel contains **no
`.yaml` at all**, so both shipped recipes (`serena.yaml`, `sglang.repo.yaml`) are
absent. `recipes: [serena]`, which is what `examples/env_checker` declares and
what `env_mgr/docs/design.md:1277` documents as *"the staged package, then
`env_mgr/recipes/<name>.yaml`"*, refuses:

> `agent 'x' declares recipe 'serena' and none of [...] exists.`

This route has nothing to do with the rename. It is recorded here because it is
the same defect with the same cause, and because it is the one that breaks a
route the shipped example actually uses.

## Why

`agent_sys/pyproject.toml`:

```toml
[tool.setuptools.packages.find]
include = ["env_mgr*", "task_graph*", "agent_sys_helper*",
           "spec_loader*", "handoff*", "validator*", "agent*", "closure*",
           "monitor*", "cli*"]

[tool.setuptools.package-data]
spec_loader = ["schemas/*.json"]
```

Neither directory is a Python package — measured, both lack `__init__.py`:

```
$ ls -a env_mgr/recipes/          .  ..  serena.yaml  sglang.repo.yaml
$ ls -a spec_loader/schemas/      .  ..  _common.schema.json  ...
$ find_packages(include=[...])    env_mgr, env_mgr.fs, env_mgr.installers,
                                  env_mgr.isolation, env_mgr.remote
                                  # no env_mgr.recipes, no agent_plugins
```

So `packages.find` cannot reach either, and the *only* route by which a non-`.py`
file gets into this wheel is `package-data`. `spec_loader/schemas/` — which is
also not a package, and ships anyway — is the proof that route works: a
`package-data` glob is resolved relative to the owning package's directory and
does not require the subdirectory to be one.

That gives the two routes different prognoses:

- **`env_mgr/recipes/` is fixable by one line**, because the working analogue is
  in the same file, three lines above where the fix goes.
- **`agent_plugins/` is not**, because it has no owning package to hang a glob
  on. It sits beside `env_mgr/`, not inside it.

`pyproject.toml`'s own comment about the schemas describes this bug exactly:
*"a bare `agent_sys/schemas/` is not a package, so `find_packages` cannot see it
and setuptools will not install it. Reading it by relative path works from a git
checkout and dies from a wheel."* **It was written about one directory, fixed
for that directory, and the class was not swept for.** Two more instances were
sitting beside it.

## What would fix it upstream

For the recipes, one line beside the existing entry:

```toml
[tool.setuptools.package-data]
spec_loader = ["schemas/*.json"]
env_mgr = ["recipes/*.yaml"]
```

The mechanism is the one `spec_loader/schemas/` already uses successfully from a
directory that is likewise not a package, so this is a strong candidate rather
than a guess. **It was still not run** — no wheel was built with that line in
it, and this record does not claim the fix works.

`agent_plugins/` has no such line available: `package-data` needs an owning
package, and `agent_plugins/` sits beside `env_mgr/` rather than inside it. It
is also deliberately *not* a Python package — `agent_assets.py` calls its root
*"a repository path, not a configurable root"* — so making it one to get it into
a wheel would change what it is. The alternatives are a ruling somebody has to
make: move it under `env_mgr/` and ship it as package data, or state that
`agent_plugins:` is a checkout-only route and say so in the three documents that
currently do not.

## What is NOT claimed here

- That anyone installs `agent_sys` from a wheel today. Not measured. Everything
  in this repository runs from a checkout, where both routes work.
- That the failure is silent. **It is not** — both raise `PrepareRefused` with
  the declaring agent and the missing path named. That is the difference between
  this and a defect worth stopping for.
- That `env_mgr/recipes/` would ship with only the `package-data` line above.
  See the previous section.

## Why it is worth a record rather than a shrug

`docs/design.md:1277` and `spec_loader/schemas/agent.schema.json` both document
the bare-name recipe route without qualification, and `agent_plugins/README.md`
says *"the directory name **is** the name `agent_plugins: [<name>]` resolves"*.
Three documents describe a resolution that holds only under one install mode,
and none of them says which. A reader installing from a wheel meets an error
message that tells them they made a mistake they did not make.

The near-miss is worth stating too: the `components/` → `agent_plugins/` rename
was checked against packaging *because* a move is the classic way to break
package data. The check found nothing to break, which looked like a clean
result and was actually the bug.
