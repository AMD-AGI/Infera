# `env_mgr/recipes/` — the recipes you name

Everything in this directory is reached by **naming it**:

```yaml
  recipes: [serena]     # -> env_mgr/recipes/serena.yaml
```

`agent_assets.py::_recipe_paths` resolves a bare name to `<this dir>/<name>.yaml`.
It applies `os.path.basename` to the declared name deliberately, so a name may
not select a file outside this directory by spelling a path — which also means
**a subdirectory here is unreachable**: `recipes: [demo/x]` resolves to `x`.

## The default recipe is **not** here, and that is the distinction

It is at **`env_mgr/default.env_recipe.yaml`**, one level up.

`recipes/` is the namespace of things you *name*. The default is the one you
**never** name, because it always applies — so it is not in the directory of
nameable things. Nothing marks it with a field or a flag; its path is what says
which layer it is. That is the same rule the item schema follows: the layer is
*where the file is*, and a field repeating it would be a second writer of one
fact.

The three layers, most general to most specific, all in
`agent_assets.py::_recipe_paths`:

| layer | where | declared |
|---|---|---|
| default | `env_mgr/default.env_recipe.yaml` | never |
| package | `<staged package>/assets/main.env_recipe.yaml` | never — auto-detected |
| agent | `recipes: [...]` on the agent spec | **here**, by name, or by package-relative path |

## What is here

| file | |
|---|---|
| `serena.yaml` | installs serena from upstream — see the caveat below |
| `sglang.repo.yaml` | an sglang development environment |

**Both are classed as demonstrations by the owner. `serena.yaml` is not only a
demonstration in practice, and you need to know that before you touch it:**
`examples/env_checker/steps/check.yaml` declares `recipes: [serena]` as a live
dependency, and its capability 7 fails without it. The label and the usage
disagree; that disagreement is recorded here rather than resolved, because which
one gives is the owner's call and not this file's.

## Adding one

Drop a `<name>.yaml` here in `env_mgr/recipe.py`'s format — `target` plus
`items`, each item naming an `installer` and an `importance`. There is nothing
to register: the filename is the name `recipes: [<name>]` resolves.

If what you are adding should apply to **every** agent without being named, it
does not belong here — it belongs in the default recipe one level up.
