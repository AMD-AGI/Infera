#!/bin/bash
# Stage the branch source + the build driver onto shared NFS so both held nodes
# can build the same tree.
#
# `git archive` of HEAD would drop the uncommitted ROCm hicache patch, which is
# exactly the delta this cluster needs -- so this stages the WORKTREE for the
# tracked file set instead, and records the delta explicitly in the manifest.
set -eu
REPO=/home/yihou/dev/git/infera.merge.liying.kv.mtp
OUT=/shared_nfs/yihou_agbench_mtp/build
mkdir -p "$OUT"

cd "$REPO"
# Tracked files + the two staged-but-uncommitted additions. `git ls-files` keeps
# .dockerignore's spirit (no manual/, tests/, rust/target) without shipping the
# packup folders, which are huge and irrelevant to the build.
git ls-files -c -o --exclude-standard \
  | grep -vE '^(agenticbench\.|glm52\.|kvaware_kvd_pr\.|liying_rest_pr56\.|merge_kvaware_mtp_pd\.|work\.|manual/|tests/|images/|partial\.debug)' \
  > "$OUT/filelist.txt"
tar -cf "$OUT/src.tar" -T "$OUT/filelist.txt"
echo "staged $(wc -l < "$OUT/filelist.txt") files -> $OUT/src.tar ($(du -h "$OUT/src.tar" | cut -f1))"

{
  echo "branch     $(git rev-parse --abbrev-ref HEAD)"
  echo "commit     $(git rev-parse HEAD)"
  echo "dirty      $(git status --porcelain | wc -l) paths"
  git status --porcelain
} > "$OUT/manifest.txt"
cat "$OUT/manifest.txt"
