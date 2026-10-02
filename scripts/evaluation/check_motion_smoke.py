#!/usr/bin/env python3
"""Check finite CHORD smoke checkpoints and updates between two saved steps."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, default=3)
    args = parser.parse_args()
    report = {}
    for obj in ['obj_0', 'obj_2']:
        states = []
        for step in [args.first, args.last]:
            path = args.run_dir / 'deform' / f'deform_{step}' / f'{obj}.pth'
            data = torch.load(path, map_location='cpu', weights_only=False)
            tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
            tensors += [data[k].cp_deform for k in ['cp_deform', 'c_cp_deform']
                        if data.get(k) is not None]
            if not all(torch.isfinite(t).all() for t in tensors):
                raise ValueError(f'Nonfinite checkpoint: {path}')
            states.append(data['cp_deform'].cp_deform.detach())
        change = (states[1] - states[0]).abs().max().item()
        if change <= 0:
            raise ValueError(f'No motion parameter update: {obj}')
        report[obj] = dict(finite=True, motion_max_change=change,
                           first_step=args.first, last_step=args.last)
    (args.run_dir / 'debug_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
