# Task v0521cap2 packup C: per-window max/mean of SGLang scheduler log fields (Prefill batch / Decode batch lines, all DPs pooled).
# Usage: awk -v W1S=HH:MM:SS -v W1E=HH:MM:SS -v W2S=.. -v W2E=.. -v TAG=name -f logstats.yihou.awk server.log
# Window membership uses the container log timestamp (UTC, field 1, chars 12-19). Fields reported:
#  prefill: #queue-req, #inflight-req, token usage, #running-req, #bootstrap-req, #optimistic-req
#  decode : #running-req, token usage, pre-allocated usage, #prealloc-req, #transfer-req, #retracted-req, #queue-req
function grab(line, key,   i, s) { i = index(line, key ": "); if (!i) return ""; s = substr(line, i + length(key) + 2); sub(/[ ,].*/, "", s); return s }
function acc(w, k, v) { if (v == "") return; v = v + 0; n[w, k]++; sum[w, k] += v; if (!((w, k) in mx) || v > mx[w, k]) mx[w, k] = v; keys[k] = 1 }
{
  t = substr($1, 12, 8); w = ""
  if (t >= W1S && t <= W1E) w = "W1"; else if (W2S != "" && t >= W2S && t <= W2E) w = "W2"
  if (w == "") next
  if (index($0, "Prefill batch")) { split("#queue-req,#inflight-req,token usage,#running-req,#bootstrap-req,#optimistic-req", ks, ",")
    for (j in ks) acc(w, "prefill " ks[j], grab($0, ks[j])); lines[w, "prefill"]++ }
  else if (index($0, "Decode batch")) { split("#running-req,token usage,pre-allocated usage,#prealloc-req,#transfer-req,#retracted-req,#queue-req", ks, ",")
    for (j in ks) acc(w, "decode " ks[j], grab($0, ks[j])); lines[w, "decode"]++ }
}
END {
  for (ww = 1; ww <= 2; ww++) { w = "W" ww
    for (k in keys) if (n[w, k] > 0) printf "%s\t%s\t%s\tn=%d\tmax=%g\tmean=%.3f\n", TAG, w, k, n[w, k], mx[w, k], sum[w, k] / n[w, k]
  }
}
