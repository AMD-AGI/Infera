#!/usr/bin/env python3
"""Write a per-DP-rank ISL list (for --input-len-spec list:@FILE) matching measured decode-shape moments.

The llying decode step log (decode_shape_summary.json, steady window) records per step and per DP rank only the
running-request count, the MEAN context per request and the MAX context in the batch; the per-request
distribution itself is not recorded. This writes the simplest list that reproduces those two moments:
`--long` requests at --max-context, the rest drawn from a clamped normal around the mean that keeps the overall
mean at --mean (seeded, sorted descending is NOT applied: the order is the draw order). It is a model of the
shape, not the measured distribution; say so wherever it is used.
Shape A defaults: 224 requests per DP rank, mean 173297 (steady context_per_req mean), one request at 829824
(steady context_max p50), others N(mu, 0.15*mu) clamped to [32768, 262144].
"""
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=224)
parser.add_argument("--mean", type=float, default=173297)
parser.add_argument("--max-context", type=int, default=829824)
parser.add_argument("--long", type=int, default=1)
parser.add_argument("--rel-std", type=float, default=0.15)
parser.add_argument("--lo", type=int, default=32768)
parser.add_argument("--hi", type=int, default=262144)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rng = random.Random(args.seed)
rest = args.count - args.long
mu = (args.mean * args.count - args.long * args.max_context) / rest
values = [args.max_context] * args.long
values += [min(args.hi, max(args.lo, round(rng.gauss(mu, args.rel_std * mu)))) for _ in range(rest)]
rng.shuffle(values)
with open(args.output, "w") as handle:
    handle.write(f"# make_isl_shape.py count={args.count} mean_target={args.mean} max={args.max_context} long={args.long} "
                 f"rel_std={args.rel_std} clamp=[{args.lo},{args.hi}] seed={args.seed}; realized mean "
                 f"{sum(values) / len(values):.1f}\n")
    handle.write("\n".join(str(v) for v in values) + "\n")
print(f"{len(values)} values, mean {sum(values) / len(values):.1f}, min {min(values)}, max {max(values)}")
