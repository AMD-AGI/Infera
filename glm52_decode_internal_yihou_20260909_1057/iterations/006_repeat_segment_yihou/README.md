# Iteration006: matched long-context repeat

## Goal and change
Repeat the first256 measured steps of full run005 with identical code/config/seed and warmup10. Only maxsteps=256 is added; input70000/output10000 capacity retained. Compare matched steps, not whole-run average.

## Result
PASS exit0. All8 ranks agree, all three graph execute counters256. Acceptance and context trajectory exactly match005 first256 steps.14864 useful tokens,final context70929,complete=false as expected. Overall elapsed7.184349s; matched step sum7.183488s vs7.185985s in005(-0.034756%). Median27.928701ms vs27.996378ms. One bounded rerun only; no statistical-equivalence claim.

Evidence: command.txt,runtime.log,result_yihou.json,rank JSONs,steps_yihou.jsonl,source snapshots; comparison in results/repeat_comparison_yihou.json.
