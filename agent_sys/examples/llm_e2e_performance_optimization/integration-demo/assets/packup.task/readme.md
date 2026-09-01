# packup

Assemble the deliverable in the `experiment-result-packup` layout: README,
REPRODUCE, environment, notes, results, logs, scripts.

## content_type is `code`, not `reproducible`

Laying a packup into `reproducible` renames `results/` to `items/result` and
leaves `REPRODUCE.md` with no item to be, which destroys the thing
`check_packup_shape` exists to check. `code`'s `codes` item is unconstrained
inside, so the layout survives intact. `single_real_task` reached the same
conclusion for the same reason.

## REPRODUCE.md is written to be executed

`check_packup_shape` counts its command lines rather than its prose, because it
is the one file in a packup somebody actually runs. It also defines a success
condition narrower than "reproduce the whole comparison": bring the stock arm up,
pass smoke, retrieve the gated needle, and score one 200-question gsm8k run.
Roughly twenty-five minutes, most of it the cold start — against an hour for the
full two-arm run.

## What it does not carry

Model weights, container images, source trees. Only the patch, the hashes that
pin it to one image, and the evidence. `scripts/` holds each handoff's `command`
item, which is executable and takes its site paths as shell variables.
