"""Count occupied 128-byte packet slots in the fixed 32x32 USTC images."""
import argparse
import json
from pathlib import Path

import numpy as np

from pilot.pairing import USTC_LABELS, sha_file


def inspect(dataset_dir):
    dataset_dir = Path(dataset_dir)
    sources = [dataset_dir / f'USTC_1c_{part}.npz' for part in ('train', 'test')]
    counts = {name: {'rows': 0, 'multi_slot_rows': 0, 'eight_slot_rows': 0}
              for name in USTC_LABELS}
    for path in sources:
        with np.load(path, allow_pickle=False) as archive:
            x, y = archive['data'], archive['target']
            if x.dtype != np.uint8 or x.shape[1:] != (32, 32) or len(x) != len(y):
                raise ValueError('Unexpected USTC schema: ' + path.name)
            occupied = np.any(x.reshape(-1, 8, 128) != 0, axis=2).sum(axis=1)
            for name, label in USTC_LABELS.items():
                slots = occupied[y == label]
                counts[name]['rows'] += len(slots)
                counts[name]['multi_slot_rows'] += int(np.count_nonzero(slots > 1))
                counts[name]['eight_slot_rows'] += int(np.count_nonzero(slots == 8))
    return {'schema': 'ustc-image-packet-slot-audit-v1',
            'sources': [{'file': p.name, 'sha256': sha_file(p)} for p in sources],
            'slot_bytes': 128, 'class_counts': counts,
            'caution': ('An occupied image slot indicates bytes in that position, not an independently '
                        'verified packet or a recoverable IAT. The source preprocessing defines the slots.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.dataset_dir)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print('Rows with bytes in more than one 128-byte slot:',
          sum(item['multi_slot_rows'] for item in result['class_counts'].values()))
