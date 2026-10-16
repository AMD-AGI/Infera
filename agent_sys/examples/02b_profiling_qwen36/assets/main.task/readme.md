# main — profile Qwen3.6 with the profiler off and on

This task runs nothing itself. `m1_deploy` deploys the model and publishes a
`deploy_kit`; `m2_profiling` uses that kit to bring the service up, replays the
load with the profiler off (the online numbers), then again with it on (the
torch trace and kernel table), and merges them into `profiling_evidence`.
