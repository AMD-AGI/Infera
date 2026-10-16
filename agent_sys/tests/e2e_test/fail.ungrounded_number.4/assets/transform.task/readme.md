# transform

Read the facts manifest from `$AGENT_SYS_INPUT_FACTS`, write a summary into
`$AGENT_SYS_OUTPUT_SUMMARY` that includes a fabricated "elapsed_ms: 99999" —
a number guaranteed to not appear in the facts. Copy the facts into
`items/grounding/` so check_grounded can compare without reaching outside.
