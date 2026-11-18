#!/usr/bin/env bash
# Per-request chat-template render time (vllm:request_render_duration_seconds, chat) and engine TTFT / queue time over
# an interval, for the prefill and the decode API servers.   render_delta.sh IP [SECONDS=300]
ip=$1 secs=${2:-300}
snap() { for p in 27201 27301; do curl -s -m 20 "http://$ip:$p/metrics" | awk -v p=$p '
  /^vllm:request_render_duration_seconds_(sum|count)\{.*request_type="chat"/ {split($1, a, "{"); v[a[1]] += $2}
  /^vllm:(time_to_first_token_seconds|request_queue_time_seconds|request_prefill_time_seconds)_(sum|count)\{/ {split($1, a, "{"); v[a[1]] += $2}
  END {for (k in v) print p, k, v[k]}'; done; }
snap > /tmp/rd0.txt; sleep "$secs"; snap > /tmp/rd1.txt
join <(awk '{print $1"|"$2, $3}' /tmp/rd0.txt | sort) <(awk '{print $1"|"$2, $3}' /tmp/rd1.txt | sort) | awk '
  {split($1, k, "|"); d[k[1]" "k[2]] = $3 - $2}
  END {for (p in a) ; for (key in d) { split(key, q, " "); port = q[1]; name = q[2]; sub(/_(sum|count)$/, "", name)
         if (key ~ /_sum$/) s[port" "name] = d[key]; else c[port" "name] = d[key] }
       for (x in s) printf "%s: %.3f s per request over %d requests\n", x, s[x] / (c[x] > 0 ? c[x] : 1), c[x] }' | sort
