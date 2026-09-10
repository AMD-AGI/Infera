# Iteration004: target long-context and capacity smoke

## Hypothesis
The verified graph path remains valid with16 resident70000-token prefixes and10000 output-token capacity per request, switching DSA from short-context dense graph to sparse graph.

## Change
Only workload length/warmup/check length change from003: ISL70000,OSL10000,warmup5,maxsteps16. Same frozen implementation and optimized image. Full command in command.txt, actual runtime output in runtime.log.

## Pass criteria
Physical full-target KV mappings allocate successfully; finite bootstrap;16 complete speculative graph iterations; worker sequence lengths match actual acceptance; all8 ranks agree; each graph stage executes; complete=false because smoke intentionally stops early. This is not the final160000-output-token test.

## Result
PASS exit0,all8 rank JSON equal.16 target/draft/draft-extension executes each,1008 useful tokens,context70000->70063,complete=false.1,281,024 mapped token slots(80,064/request,page64,reserve12). Finite/bootstrap/layout/state checks pass.0.453703s is a short16-iteration smoke sample; realized accept length3.9375, not yet the full expected3.61 workload.
