#!/usr/bin/env bash
# Token diff of the rendered engine argv (+env) per leg between two dryrun files.
# Usage: tokdiff.yihou.sh A.txt B.txt   (round names normalised to ROUND)
leg(){ awk -v want="## $2-0" '/^## /{on=index($0,want)==1} on && /^ \+ docker/' "$1" |
       sed 's/ + /\n/g' | sed -E 's/-yihou-[0-9]+-[A-Za-z0-9]+-/-yihou-NODE-ROUND-/' ; }
for r in prefill decode; do
  echo "=== $r: < $(basename "$1")   > $(basename "$2")"
  diff <(leg "$1" $r) <(leg "$2" $r) | grep '^[<>]'
done
