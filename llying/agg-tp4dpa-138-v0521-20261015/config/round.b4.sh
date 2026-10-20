# b4 (crsuse2-m2m-140): b0r's engine, still running, scored again with the router's session
# affinity off (ROUTER_ENV in operations/b4.steps): every request is placed by the kv-aware score
# instead of staying on its session's DP rank. b2/b0r showed the DP ranks' prefill load uneven.
source "$AGG/config/round.b0r.sh"
