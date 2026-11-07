#!/usr/bin/env bash
# v2 kit 1P2D C128 again: the 11-06 21:03 point (35231, n04-25 -> n04-29) lost one warmup request (both decodes answered
# 1,420 / 1,420 sent with 200 OK, the client got 1,419; prefill idle from 21:30:43, one KV transfer in flight), and the sweep
# watchdog stopped AgentX after 15 min. C96 of the same sweep: 38,094 @ p90 84.3.
ITEM="$0" exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-1p2d-c128 sweep 1p2d 128
