# Running `agent-sys` on a Slurm compute node — three faults, all environmental

**Found:** 2026-09-01, `crsuse2-m2m-188` (amd-spur, MI355X), reaching the node
with `spur exec <jobid> bash <script>`.
**Method:** `examples/demo` as a harness smoke test — no GPU, ~90 s per attempt.
Each fault below cost one of those attempts instead of one 40-minute run of the
real package, which is the only reason to write this down.

None of the three is an `agent_sys` bug. All three are things a caller must do
that nothing tells them to do, and the third has a sharp edge worth knowing.

## 1. `claude` is not on `PATH`

```
done  the 'claude' backend is not on PATH.
```

`spur exec` builds its own `PATH` and it does not include `~/.local/bin`, which
is where a per-user `claude` install puts the binary.

**Fix:** `export PATH="$HOME/.local/bin:$PATH"` in the run script.

## 2. `HOME` is `/opt/spur`, so `claude` has no credentials

```
done  claude exited 1 unconfined, so this operator cannot authenticate at all
stdout: Not logged in · Please run /login
```

Measured directly (`ws/claude_auth_probe.sh`):

| environment | `ANTHROPIC_API_KEY` | `claude -p 'reply PONG'` |
|---|---|---|
| ambient `spur exec` — `HOME=/opt/spur` | absent | `Not logged in · Please run /login` |
| same, with `~/.claude/settings.json`'s `env` block exported | present | `PONG` |

The credentials are not in the ambient environment on a compute node; they are
in `$HOME/.claude/settings.json`, which `claude` reads for itself and which
`env_mgr.harness.harness_env` reads to build its by-name allow-list. With
`HOME` pointing at `/opt/spur`, neither finds it.

**Fix:** `export HOME=/home/<user>` in the run script, before anything else.
That is preferable to exporting the key: the value then never appears on a
command line, in a process listing, or in a log.

## 3. `extensions.preciousObjects` in a **git worktree** hits the shared config

```
done  <repo> does not set extensions.preciousObjects, and `env_mgr.workspace.cut`
      refuses without it — so every output-producing task would die in `prepare`.
Note: in a git worktree this lands in the SHARED common config, so it affects the
      main checkout and every other worktree, and `git gc` will refuse in all of
      them until it is unset.
```

The CLI's own message is accurate and is the reason this is not filed as a bug.
The consequence is still worth stating plainly: `_main_repo` walks up from the
package directory to the first `.git`, and in a worktree that resolves to the
shared common directory. Setting the key — by hand or with `--allow-repo-config`
— changes the state of **every other worktree and the main checkout**, on a box
where other people's sessions are using them.

**What we did instead of asking for that:** a standalone clone.

```bash
git clone /home/yihou/dev/git/infera.aiopt.real.task_package \
          /shared_nfs/yihou/agent_sys_debug/repo          # 94 MB
git -C /shared_nfs/yihou/agent_sys_debug/repo config extensions.preciousObjects true
```

Runs use `--package <clone>/agent_sys/examples/...`, so `_main_repo` is the
clone and the key lands there. Authoring stays in the worktree; each run syncs
with `git -C <clone> fetch <worktree> <branch> && git checkout FETCH_HEAD`,
which has the side benefit that **a run names the commit it ran**.

A stub repository holding only the package would also work and is worse: the
agent's workspace is cut from this repository, and the task brief tells the
agent to read `examples/` and `docs/` for the launch recipes. A stub takes that
away.

## Not a fault, but the same class

`spur exec` runs at `pwd=/`. `agent_sys` is imported from the current directory
in an editable install that only exposes `agent_sys_helper`, so a script that
does not `cd` into the repository first gets `ModuleNotFoundError: No module
named 'agent_sys'` — on a node where the module is plainly present.
