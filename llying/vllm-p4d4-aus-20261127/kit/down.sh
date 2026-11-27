#!/usr/bin/env bash
# Remove the deployment's containers (and the client's, if still running).
docker rm -f vpd-agentx vpd-eval-gsm8k vpd-eval-algate vpd-router vpd-decode vpd-prefill vpd-etcd > /dev/null 2>&1
exit 0
