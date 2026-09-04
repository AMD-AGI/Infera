# Two declaration routes resolve to repository paths a wheel does not ship

**Found:** 2026-09-04 by `pkg-impl` during the PR-155 `components/` →
`addons/` rename, while checking whether the `git mv` could break
packaging. It could not — because neither name was ever packaged.
**Severity:** two documented declaration routes — `recipes: [<bare name>]` and
`agent_plugins: [<name>]` — cannot resolve from a wheel install. **Both fail
loudly**, as a named `PrepareRefused`, so nothing is silently wrong; the run
refuses. From a git checkout, which is how everything runs today, both work.
**Status:** **one of the two is FIXED and proven; the other is still open.**
See *Resolution* at the end — do not act on the body below without reading it.
Pre-existing and older than the rename that surfaced it.

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

So: **no `addons/` member, and no `.yaml` member of any kind.** The
schemas ship only because they have an explicit `package-data` entry.

## The two routes

### 1. `agent_plugins: [<name>]`

`env_mgr/agent_assets.py`:

```python
ADDONS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_plugins")
```

From a wheel that is `<site-packages>/agent_plugins`, which does not exist.
`_addon_trees` then raises:

> `agent 'x' declares component 'envchk-baseline' and '<...>/addons/envchk-baseline' does not exist. 'agent_plugins:' takes a bare name under agent_sys/env_mgr/addons/, never a path`

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
- **`addons/` is not**, because it has no owning package to hang a glob
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

`addons/` has no such line available: `package-data` needs an owning
package, and `addons/` sits beside `env_mgr/` rather than inside it. It
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
the bare-name recipe route without qualification, and `addons/README.md`
says *"the directory name **is** the name `agent_plugins: [<name>]` resolves"*.
Three documents describe a resolution that holds only under one install mode,
and none of them says which. A reader installing from a wheel meets an error
message that tells them they made a mistake they did not make.

The near-miss is worth stating too: the `components/` → `addons/` rename
was checked against packaging *because* a move is the classic way to break
package data. The check found nothing to break, which looked like a clean
result and was actually the bug.


---

# Resolution, 2026-09-04

## `agent_plugins:` / addons — **FIXED, and measured fixed**

The directory moved from `agent_sys/agent_plugins/` to **`agent_sys/env_mgr/addons/`**.
That move is what makes the fix possible: `package-data` needs an owning package
and the directory now has one. `pyproject.toml` gained

```toml
env_mgr = [
    "addons/README.md",
    "addons/*/README.md",
    "addons/*/.claude/**/*",
    "addons/*/.claude/.mcp.json",
]
```

Proven by building the wheel and listing its members — **6 of 6 addon files
present**, where the build in the body of this record had 0.

**The fourth pattern is the finding, and it cost a build to see.** With only the
first three, the wheel shipped **4 of 6** and looked correct until they were
counted: `addons/*/.claude/**/*` matched `.claude/servers/envchk_baseline_server.py`
but **not** `.claude/.mcp.json`. The glob crosses a dot-*directory* and will not
match a dot-*filename* at the leaf. Nothing in the pattern hints at that
asymmetry, and a reader who adds a dotfile to an addon will hit it again — which
is why the pattern carries a comment saying so.

This is the second time in this record that reading the configuration was not
enough and building the artefact was. **Count the members.**

## `env_mgr/recipes/*.yaml` — **STILL OPEN**

Unchanged: the wheel ships no `.yaml`, so `recipes: [<bare name>]` still cannot
resolve from a wheel, and `examples/env_checker` declares `recipes: [serena]` as
a live dependency. Deliberately not fixed in the same change — it is a different
route with a different blast radius, and bundling it would have hidden it behind
a rename.

The candidate remains one line, and it is now a *stronger* candidate than when
this record was written, because the addons entry beside it has been proven to
work from a directory that is not a package:

```toml
env_mgr = [..., "recipes/*.yaml", "default.env_recipe.yaml"]
```

**Still unrun.** Whoever runs it should count the members rather than trust the
pattern.

---

# Resolution, 2026-09-04 (second pass) — **recipes FIXED and measured fixed**

Run by `pkg2`. The candidate above was correct as written, including the
`default.env_recipe.yaml` half it added almost in passing — which turned out to
be the more important of the two.

```toml
env_mgr = [
    "default.env_recipe.yaml",
    "recipes/*.yaml",
    ... the addons entries ...
]
```

## Counted, then resolved

A wheel was built and its members listed: **3 of 3** — `default.env_recipe.yaml`,
`recipes/serena.yaml`, `recipes/sglang.repo.yaml`. Then the wheel was
**installed into a clean venv** and the bare name resolved through the real code
path rather than by inspecting the zip:

```
installed at: <venv>/lib/python3.13/site-packages/env_mgr
DEFAULT_RECIPE ships: True
recipes/ dir: ['serena.yaml', 'sglang.repo.yaml']
bare name resolved to: ['default.env_recipe.yaml', 'recipes/serena.yaml']
serena.yaml parses -> 3 items
```

`agent_assets._recipe_paths({"recipes": ["serena"]})` is the function the body of
this record accuses, called against the install. **Counting members proves the
files are in the archive; calling the resolver proves the route works.** The
first run of this check imported the *checkout* rather than the venv, because the
working directory put it on `sys.path` — caught by an assertion on
`env_mgr.__file__`, which is the only thing that could tell the two apart.

## A correction to this record's own *What is NOT claimed*

> *"That the failure is silent. **It is not** — both raise `PrepareRefused` with
> the declaring agent and the missing path named."*

**That was true of the two routes this record examined and false of the one it
did not.** `default.env_recipe.yaml` is the DEFAULT recipe layer;
`_recipe_paths` gates it on `os.path.isfile(DEFAULT_RECIPE)`, and the file's own
header states the rule: *"Nothing declares this file, so its absence is simply
absence."* So from a wheel the entire default layer would have done nothing,
reported nothing and named no cause — **the silent failure this record declared
absent, in a third instance of the class it was written about.**

## The class, third time

`pyproject.toml`'s comment above `[tool.setuptools.package-data]` describes this
failure mode exactly and was written when `spec_loader/schemas` was fixed.
`addons/` was the second instance. These are the third and fourth, and they were
sitting *under the comment that explains them*.

What found them was not reading: it was `find env_mgr -type f ! -name '*.py'`,
i.e. enumerating the class's population instead of the instance in front of me.
**Fixing an instance is not sweeping a class**, and the sweep is one command.

`recipes/README.md` is deliberately still not shipped — nothing reads it at run
time. That is a decision, recorded so the next person counting members does not
read it as a fourth omission.

**This record is now closed.** Both routes named in its title resolve from a
wheel, and the one it did not name is fixed too.
