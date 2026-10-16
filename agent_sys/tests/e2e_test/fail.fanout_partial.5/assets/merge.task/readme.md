# merge

Read thing_a, thing_b, thing_c and merge them. This task never runs because
thing_b is sealed INVALID by check_thing, leaving merge in WAITING_HANDOFF.
