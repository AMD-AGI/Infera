#!/bin/bash
# extract runtime KV peaks from a decode log. Arg: <logfile>
L="$1"
echo "peak_#full_token(per-rank)=$(grep -aoE '#full token: [0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)"
echo "peak_#swa_token(per-rank)=$(grep -aoE '#swa token: [0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)"
echo "peak_full_usage=$(grep -aoE 'full token usage: [0-9.]+' "$L" | grep -oE '[0-9.]+' | sort -n | tail -1)"
echo "peak_swa_usage=$(grep -aoE 'swa token usage: [0-9.]+' "$L" | grep -oE '[0-9.]+' | sort -n | tail -1)"
echo "peak_#running=$(grep -aoE '#running-req: [0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)"
echo "total_#retracted=$(grep -aoE '#retracted-req: [0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)"
echo "peak_#queue=$(grep -aoE '#queue-req: [0-9]+' "$L" | grep -oE '[0-9]+' | sort -n | tail -1)"
