# Repository conventions

## DCO sign-off is required on every commit

This repository enforces the [Developer Certificate of Origin](https://developercertificate.org/).
CI blocks any PR containing a commit without a `Signed-off-by:` trailer, so an
unsigned commit is a broken PR, not a style nit.

Commit with `-s`, always:

```bash
git commit -s -m "..."
git commit -s -F -   # when writing a longer message from a heredoc
```

That appends a trailer matching `user.name` / `user.email`:

```
Signed-off-by: Zhang, Jiejing <jiejing.zhang@amd.com>
```

**Sign off as the repository's configured identity** (`git config user.name` /
`user.email`). The DCO is an assertion about the *right to submit* the code, so
it must name the human submitting it — never a bot or assistant identity.

### Fixing commits that are already missing it

Sign off a range without disturbing commits that already carry the trailer —
rebasing from a point that includes signed commits appends duplicates:

```bash
git rebase --signoff <last-already-signed-commit>
git push --force-with-lease origin <branch>
```

Pick `<last-already-signed-commit>` as the newest commit that already has the
trailer. Check what you are about to touch first:

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
```

Cherry-picks do **not** inherit the trailer — `git cherry-pick -s`, or sign off
afterwards. This is the easiest way to reintroduce the problem on a second branch
after fixing it on the first.

## Related trailers

`Signed-off-by` is the DCO assertion and is mandatory. `Co-Authored-By` is
separate, is not a substitute, and some upstreams reject assistant co-author
trailers outright — when contributing to a third-party repository (e.g. ROCm/aiter),
do not add them.
