#!/usr/bin/env python3
"""Continue the ten-point sweep only after the initial checkpoint validates."""
import json
from pathlib import Path
import subprocess
import sys
import time

W = Path(__file__).resolve().parents[1]


def main():
    checkpoint = W / 'rounds/dpa_off_c16_yihou/status.json'
    deadline = time.monotonic() + 9500
    while time.monotonic() < deadline:
        if checkpoint.exists():
            status = json.loads(checkpoint.read_text())
            if status['state'] == 'failed':
                raise SystemExit('Initial checkpoint failed; no other points launched')
            if status['state'] == 'passed':
                break
        time.sleep(30)
    else:
        raise SystemExit('Checkpoint timeout; server left untouched')
    subprocess.run([sys.executable, str(W / 'scripts/collect_yihou.py')], check=True)
    for mode in ['off', 'on']:
        for c in [4, 8, 16, 20, 24]:
            if (mode, c) == ('off', 16):
                continue
            print(f'Launching {mode}/C{c}', flush=True)
            result = subprocess.run([sys.executable, str(W / 'scripts/run_point_yihou.py'), mode, str(c)])
            subprocess.run([sys.executable, str(W / 'scripts/collect_yihou.py')], check=True)
            if result.returncode:
                raise SystemExit(f'{mode}/C{c} failed; sweep paused with all artifacts retained')
    print('All ten points measured and validated.', flush=True)


if __name__ == '__main__':
    main()
