#!/bin/bash
# total python CPU% + top python threads
echo "top python LWPs by CPU:"
ps -eLo pid,lwp,pcpu,stat,comm --sort=-pcpu | grep -i python | head -8
tot=$(ps -eLo pcpu,comm | grep -i python | awk '{s+=$1} END{print s}')
echo "total_python_cpu%=$tot"
