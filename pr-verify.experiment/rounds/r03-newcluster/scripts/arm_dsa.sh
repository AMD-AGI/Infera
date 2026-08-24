#!/usr/bin/env bash
# Select and VERIFY the sglang arm for the #33973 DSA A/B, on the DECODE leg.
#
#   ARM=stock    -> git checkout -- dsa_backend.py   (restores seq_lens.max().item())
#   ARM=patched  -> re-apply infera's DSA patch
#
# Guard: `seq_lens.max().item()` in dsa_backend.py -- present on stock, absent on
# patched. Same discipline as arm.sh: refuse rather than run a mislabelled A/B.
#
# Only the decode leg matters: #33973 is a DECODE DP-divergent branch. Prefill runs
# DPA=0 here and never enters it.
#
# CONFOUND TO KEEP IN MIND: there is a SECOND, unrelated deadlock on gfx950 -- the
# aiter custom all-reduce during EAGLE verify (engine/leg.sh:138). leg.sh passes
# --disable-custom-all-reduce by default, which is what keeps it out of the way. Do
# NOT set CUSTOM_AR=1 while running this A/B, or a hang becomes unattributable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${ARM:?ARM=stock|patched}"
SSH_CMD="${SSH_CMD:-$HERE/spur_ssh.sh}"
CTR="${CTR:-glm52_pd}"
DECODE_NODE="${DECODE_NODE:-crsuse2-m2m-106}"

DSA_FILE="python/sglang/srt/layers/attention/dsa_backend.py"
PATCH_SRC="${PATCH_SRC:-/home/yihou/dev/git/infera.upstream.pr.verify/deploy/docker/patches/sglang_dsa}"

rc=0
if [ "$ARM" = "stock" ]; then
  $SSH_CMD "$DECODE_NODE" "docker exec $CTR bash -c 'cd /sgl-workspace/sglang && git checkout -- $DSA_FILE'" \
    || { echo "arm_dsa: could not revert $DSA_FILE" >&2; rc=1; }
else
  # The image ships this patch applied, so "patched" normally means: put back what
  # the stock arm reverted. Applied from the .orig-free tree via the diff the image
  # was built with.
  $SSH_CMD "$DECODE_NODE" "docker exec $CTR bash -c 'cd /sgl-workspace/sglang && git checkout -- $DSA_FILE && git apply /dsa.patch'" \
    || { echo "arm_dsa: could not restore the DSA patch" >&2; rc=1; }
fi

n=$($SSH_CMD "$DECODE_NODE" "docker exec $CTR bash -c \"grep -c 'seq_lens.max().item()' /sgl-workspace/sglang/$DSA_FILE\"" 2>/dev/null | tr -d '\r\n ')
echo "  $DECODE_NODE: 'seq_lens.max().item()' occurs ${n}x  (stock>=1, patched=0)"
case "$ARM:$n" in
  patched:0)  ;;
  stock:0)    echo "  REFUSING: stock arm still has 0 occurrences -- the checkout did not restore it" >&2; rc=1 ;;
  stock:*)    ;;
  *)          echo "  REFUSING: not in the '$ARM' arm (count=$n)" >&2; rc=1 ;;
esac

[ $rc -eq 0 ] && echo "dsa arm=$ARM verified on the decode leg"
exit $rc
