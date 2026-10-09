#!/usr/bin/env python3
"""Create lightweight read-only-shard overlay for the synthetic expert8 model."""
import json
from pathlib import Path
import shutil

W = Path(__file__).resolve().parents[1]
SOURCE = Path('/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4')
DEST = W / 'model'
DEST.mkdir(exist_ok=True)
for p in SOURCE.iterdir():
    if not p.is_file():
        continue
    d = DEST / p.name
    if d.exists() or d.is_symlink():
        raise RuntimeError(f'will not overwrite existing model artifact: {d}')
    if p.name == 'config.json':
        config = json.loads(p.read_text())
        config.update(n_routed_experts=8, num_experts_per_tok=8, n_group=1,
                      topk_group=1, ep_size=4, yihou_synthetic_expert8=True,
                      index_share_for_mtp_iteration=False)
        d.write_text(json.dumps(config, indent=2) + '\n')
    elif p.suffix == '.safetensors':
        d.symlink_to(p)
    else:
        shutil.copy2(p, d)
print('Prepared synthetic expert8 overlay:', DEST)
