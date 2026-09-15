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

That appends a trailer built from **your own** `user.name` / `user.email`:

```
Signed-off-by: Your Name <you@example.com>
```

**Sign off as yourself.** The DCO is an assertion that *you* have the right to
submit this code, so the trailer must name the person making the commit. Never
copy a colleague's line from an existing commit, and never use a bot or assistant
identity — contributors here sign off under several different addresses, and the
trailer has to match the commit's actual author.

Check what `-s` will produce before your first commit in a fresh clone or
container, where git may have inherited a default from the environment:

```bash
git config user.name && git config user.email
```

If those are empty or wrong, set them (add `--global` outside a container):

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

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
