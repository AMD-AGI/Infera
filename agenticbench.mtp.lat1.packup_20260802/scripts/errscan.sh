#!/bin/bash
L="$1"
echo "  HTTP 500 : $(strings $L | grep -cE '" 500|HTTP/1.1\" 500')"
echo "  HTTP 502 : $(strings $L | grep -c ' 502 ')"
echo "  HTTP 503 : $(strings $L | grep -c ' 503 ')"
echo "  non-200  : $(strings $L | grep -oE '\- \"POST [^\"]+\" [0-9]{3}' | awk '{print $NF}' | sort | uniq -c | tr '\n' ' ')"
echo "  timeout/abort words: $(strings $L | grep -ciE 'timed out|timeout|aborted|cancelled|disconnect')"
