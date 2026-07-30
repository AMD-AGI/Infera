#!/bin/bash
# Verify a patch marker survives into the compiled bytecode.
# Source showing the fix while CPython runs a stale .pyc invalidated a full
# experiment once -- never trust the .py alone.
MOD="${1:?module path}"
MARK="${2:?marker}"
D=$(dirname "$MOD"); F=$(basename "$MOD" .py)
echo "src count : $(grep -c "$MARK" "$MOD")"
rm -f "$D/__pycache__/$F."*.pyc
python3 -c "import py_compile,sys; py_compile.compile('$MOD', doraise=True)" || exit 1
PYC=$(ls "$D/__pycache__/$F."*.pyc 2>/dev/null | head -1)
echo "pyc file  : $PYC"
echo "pyc count : $(strings "$PYC" | grep -c "$MARK")"
