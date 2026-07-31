set -e
P=/opt/infera/infera/kvd/storage_classify.py
cp $P /tmp/sc_new.py
python3 - <<'PY'
p="/opt/infera/infera/kvd/storage_classify.py"
s=open(p).read()
needle='    bracket = source.find("[")\n    if bracket > 0:\n        source = source[:bracket]\n'
assert needle in s, "fix marker not found"
open("/tmp/sc_old.py","w").write(s.replace(needle,""))
PY
echo "--- WITHOUT fix (pre-patch behaviour) ---"
cp /tmp/sc_old.py $P
python3 -m infera.kvd classify /kvd-long 2>&1 | grep -E "mount |devices |rationale"
echo "--- WITH fix ---"
cp /tmp/sc_new.py $P
python3 -m infera.kvd classify /kvd-long 2>&1 | grep -E "mount |devices |rationale"
