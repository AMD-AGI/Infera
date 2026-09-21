#!/usr/bin/env bash
# Text-coherence gate. Runs 3 temperature-0 probes against the live router and
# FAILS (exit 1) if any reply is degenerate: a leading "1!" (the original
# garbled-decode signature) or any short substring repeated 5+ times in a row
# (e.g. "4,4,4,4,4", "answer, answer, answer, ...", "1!au!au!au!").
#
# Credit for the gate idea: the parallel TP8 session, which lost two full
# 3600 s points to garbled output that a forced acceptance gauge could not see.
#
# Usage: ROUTER=http://host:port OUTFILE=path ./coherence_gate.yihou.sh [label]
#   ROUTER  default http://10.245.148.209:28000
#   MODEL   default glm5.2-mxfp4
#   OUTFILE default <workspace>/notes/coherence-<label>.txt (appended)
# Exit: 0 all coherent, 1 a probe degenerated, 2 a probe could not be obtained.
set -uo pipefail
ROUTER="${ROUTER:-http://10.245.148.209:28000}"
MODEL="${MODEL:-glm5.2-mxfp4}"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:-adhoc}"
OUTFILE="${OUTFILE:-$W/notes/coherence-$LABEL.txt}"

PROMPTS=(
  "What is 2+2? Answer with a single digit."
  "List the first 10 prime numbers in order, comma separated."
  "Reply with exactly: PROBE_OK_7391"
)

ts="$(date -u +%FT%TZ)"
echo "=== coherence gate [$LABEL] $ts router=$ROUTER ===" >> "$OUTFILE"
rc=0
for p in "${PROMPTS[@]}"; do
  # max_tokens 512, not 64: GLM-5.2 is a REASONING model — it emits into
  # reasoning_content first and only then into content, so a small cap can leave
  # content empty even on a healthy reply. We inspect BOTH fields.
  body=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':sys.argv[1]}],'temperature':0,'max_tokens':512}))" "$p")
  resp=$(curl -s -m 90 "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" 2>/dev/null)
  # Two outputs from the parser: a one-word TAG on stdout (for the exit code) and
  # the FULL raw reply written to $OUTFILE (evidence that the numbers came from a
  # coherent deployment -- exactly what the t2e numbers lack).
  tag=$(printf '%s' "$resp" | OUTFILE="$OUTFILE" PROMPT="$p" python3 -c "
import sys,json,re,os
raw=sys.stdin.read()
try:
    d=json.loads(raw)
    m=d['choices'][0]['message']
    c=(m.get('content') or '')
    rc_=(m.get('reasoning_content') or '')
    fr=d['choices'][0].get('finish_reason')
except Exception:
    with open(os.environ['OUTFILE'],'a') as f:
        f.write('  [NORESP] <- '+os.environ['PROMPT']+'\n')
        f.write('    RAW: '+repr(raw[:400])+'\n')
    print('NORESP'); sys.exit(0)
# Effective text = content if present, else reasoning_content. GLM-5.2 garble
# shows up in reasoning_content (content stays empty), so BOTH must be checked.
eff=(c if c.strip() else rc_)
def degen(t):
    t=t or ''
    return t.strip().startswith('1!') or bool(re.search(r'(.{1,8}?)\1{4,}', t))
if not eff.strip():
    tag='EMPTY'          # no content and no reasoning -> not coherent, not evidence
elif degen(c) or degen(rc_):
    tag='DEGENERATE'
else:
    tag='OK'
with open(os.environ['OUTFILE'],'a') as f:
    f.write('  ['+tag+'] finish='+str(fr)+' <- '+os.environ['PROMPT']+'\n')
    f.write('    content         : '+repr(c[:400])+'\n')
    f.write('    reasoning_content: '+repr(rc_[:400])+'\n')
print(tag)
")
  case "$tag" in
    OK)          : ;;
    DEGENERATE)  rc=1 ;;
    *)           [[ $rc -eq 0 ]] && rc=2 ;;
  esac
done
echo "  gate result: rc=$rc" >> "$OUTFILE"
exit $rc
