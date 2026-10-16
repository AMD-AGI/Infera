# main — serve Qwen3.6 with sglang

This task runs nothing itself. Its one subgraph entry is `m1_deploy`, e2e-flow's
stage 1, which deploys the model and publishes a `deploy_kit`: the scripts that
bring the service up, wait for it and tear it down, plus an `environment.yaml`
recording where and how it ran.
