#!/bin/sh
# Copy the fixture into the two output handoffs. The outputs already exist and
# are the only places this task may write.
set -eu
FIXTURE="$AGENT_SYS_TASK_PACKAGE/assets/fixture"
cp -a "$FIXTURE/deploy_kit_qwen36/." "$AGENT_SYS_OUTPUT_DEPLOY_KIT/"
cp -a "$FIXTURE/workset/." "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET/"
echo "published deploy_kit and operator_workset from $FIXTURE"
