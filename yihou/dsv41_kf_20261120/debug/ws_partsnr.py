# Per-part SNR of baseline vs reference for one workset shape (debug of the 5 dB cases). Run in the image, cwd = workset codes.
import json, sys, torch
sys.path.insert(0, ".")
from _common import build_inputs
op, T = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"definitions/mhc/{op}.json"))
shape = {"axes": {"T": T}}
torch.manual_seed(0)
def load(src):
    ns = {}; exec(compile(src, op, "exec"), ns); return ns
ref, base = load(d["reference"]), load(d["baseline"])
for seed in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3):
    torch.manual_seed(seed)
    inp = build_inputs(d, shape)
    w = ref["run"](**inp); g = base["run"](**{k: (v.clone() if hasattr(v, "clone") else v) for k, v in inp.items()})
    H = 5120; sizes = [T*4*H, T*4, T*16, T*H, T*4] + ([T*H] if op.endswith("mxfp8") else [])
    names = ["residual_out", "post_mix", "comb_mix", "layer_input", "next_pre"] + (["deq"] if op.endswith("mxfp8") else [])
    o = 0; out = []
    for n, s in zip(names, sizes):
        a, b = g[o:o+s].double(), w[o:o+s].double(); o += s
        err = (a-b).pow(2).sum().item(); sig = b.pow(2).sum().item()
        bad = ((a-b).abs() > 1e-2 * b.abs().max()).nonzero()
        out.append(f"{n}: snr {10*torch.log10(torch.tensor(sig/max(err,1e-300))).item():.1f} bad {bad.numel()}"
                   + (f" first {bad[0].item()} row {bad[0].item() // (s // T)}" if bad.numel() else ""))
    print(f"seed {seed} T {T}:", "; ".join(out), "| ref nan", torch.isnan(w).sum().item(), "base nan", torch.isnan(g).sum().item())
