# `env_mgr/recipes/` — the recipes you name

Everything in this directory is reached by **naming it under the `agent_sys:`
root**:

```yaml
  recipes: [agent_sys:serena]     # -> env_mgr/recipes/serena.yaml
```

`agent_assets.py::_recipe_paths` resolves `agent_sys:<name>` to
`<this dir>/<name>.yaml`. **A reference names its root and there is no bare
form**: until 2026-09-04 `recipes: [serena]` was resolved by trying a
package-relative path first and falling back here, so which root was meant
depended on which file existed — measured, a package carrying a file named
exactly `serena` shadowed this directory's `serena.yaml` in silence. A bare name
is now a dated error.

`<name>` may not contain a separator — **a subdirectory here is unreachable**,
and `agent_sys:demo/x` is refused rather than silently read as `x`, which is
what the previous `os.path.basename` call did.

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
| agent | `recipes: [...]` on the agent spec | **here** via `agent_sys:<name>`, or the package's own via `package:<relpath>` |

## What is here

| file | |
|---|---|
| `serena.yaml` | installs serena from upstream — see the caveat below |
| `sglang.repo.yaml` | an sglang development environment |

**Both are classed as demonstrations by the owner. `serena.yaml` is not only a
demonstration in practice, and you need to know that before you touch it:**
`examples/env_checker/steps/check.yaml` declares `recipes: [agent_sys:serena]` as a live
dependency, and its capability 7 fails without it. The label and the usage
disagree; that disagreement is recorded here rather than resolved, because which
one gives is the owner's call and not this file's.

## Adding one

Drop a `<name>.yaml` here in `env_mgr/recipe.py`'s format — `target` plus
`items`, each item naming an `installer` and an `importance`. There is nothing
to register: the filename is the name `recipes: [<name>]` resolves.

If what you are adding should apply to **every** agent without being named, it
does not belong here — it belongs in the default recipe one level up.
