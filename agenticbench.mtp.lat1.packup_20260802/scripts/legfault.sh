#!/bin/bash
L="$1"
echo "faults=$(strings $L | grep -cE 'HSA_STATUS_ERROR|Fatal Python error|Expected lengths.size|Memory access fault|Scheduler hit an exception') retract_nonzero=$(strings $L | grep -cE '#retracted-req: [1-9]') lastline=$(strings $L | tail -1 | cut -c1-80)"
