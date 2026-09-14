# `check_deploy_kit` — completeness, strong

The handoff carries exactly one `<name>.packup_<YYYYMMDD>/` directory, that
directory holds every entry this package makes mandatory, each mandatory
document carries substance rather than a heading and a blank line, the kit
freezes no identifier it then binds on a shared host, and — the part that is
new here — **it carries the four things a standardised deploy kit is for**.

`strong` without qualification: every rule is decided by opening a file that
either is there or is not, and the substance rules are counted rather than
judged. It cannot be *approximately* right about whether `REPRODUCE.md` exists.

## Inherited, unchanged, from `single_real_task/check_packup_shape`

This body is that one with four rules added. The reasoning for everything below
lives in that package's readme and is not repeated:

- exactly one packup directory under `items/codes/`;
- `README.md`, `REPRODUCE.md`, `environment.md`, `notes.md` all present, each
  over its content-line floor, none carrying an unfilled placeholder;
- `scripts/` and `results/` present and non-empty;
- at least *n* command lines inside code blocks in `REPRODUCE.md`;
- a `## Result` heading in `README.md`; a digit somewhere in `environment.md`;
- no identifier frozen in `scripts/` and then bound into a host-wide namespace.

## The four added rules, and why each exists

Each was a fault in a real kit before it was a rule here.

| rule | arg | why |
|---|---|---|
| `REPRODUCE.md` has an **`Expected output`** section | `require_expected_output` | It is the *only* criterion `check_deploy_reproduces` hands its reproducer. Vague here means a correct reproduction can be judged a failure and an incorrect one a success, and neither verdict is recoverable afterwards |
| `results/` holds ≥ *n* non-empty **`.json`** files | `min_json_results` | Stage 2 of this flow (profiling) consumes results, and it consumes files rather than paragraphs. A kit whose evidence is prose cannot be diffed against the next run |
| `environment.md` names the **GPU architecture, the image and the model** | `environment_facts` | These three are what a reproduction fails without, and their absence is the commonest reason a kit does not travel. The patterns are deliberately generous — the rule is "this fact is stated", not "stated the way I would have written it" |
| no evidence file shows the model served under a **filesystem path** | `require_served_name_not_a_path` | Measured: one kit registered `Qwen/Qwen3.6-27B`, the next registered `/data/<user>/…/Qwen3.6-27B`, purely from `--served-model-name` being absent — baking one machine's directory layout into the one field every caller copies |

## What it cannot catch, stated so nobody assumes otherwise

- **Correctness of the commands.** It counts command lines; it does not parse
  shell, and a `REPRODUCE.md` full of plausible nonsense passes here and fails
  at `check_deploy_reproduces`, which is the right place for it.
- **Whether the evidence is *true*.** A hand-written `completion.json` with the
  right shape passes. The corroboration that it was produced by a running server
  is the reproduction check's, and that check is `weak`.
- **The served-name rule scans `results/` only.** A kit is *encouraged* to
  explain the path trap in `notes.md`, and a rule that fired on prose would
  punish it for doing so. A kit that puts its evidence somewhere other than
  `results/` escapes the rule — and fails the `results/` floor instead.
- **The `environment.md` fact patterns match a word, not a claim.** A line
  saying "no image was used" satisfies the image rule. The floor is against
  omission, not against dishonesty.
