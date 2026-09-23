"""Locate original image first-packet blocks in a source capture.

An anchor is only a byte-level clue. It does not pair a full image with a flow.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot.pairing import USTC_LABELS, sha_file
from pilot.trace_preprocessing import legacy_block


def build_anchors(dataset_dir):
    anchors = defaultdict(list)
    sources = []
    for part in ('train', 'test'):
        path = Path(dataset_dir) / f'USTC_1c_{part}.npz'
        with np.load(path, allow_pickle=False) as archive:
            x, y = archive['data'], archive['target']
            if x.dtype != np.uint8 or x.shape[1:] != (32, 32) or len(x) != len(y):
                raise ValueError('Invalid USTC NPZ')
            blocks = x.reshape(-1, 8, 128)
            lengths = np.any(blocks != 0, axis=2).sum(axis=1)
            for i, (image, label, length) in enumerate(zip(blocks, y, lengths)):
                key = hashlib.sha256(image[0].tobytes()).digest()
                anchors[key].append((path.name, i, int(label), int(length)))
        sources.append({'file': path.name, 'sha256': sha_file(path)})
    return anchors, sources


def probe(dataset_dir, captures, packet_limit):
    from scapy.all import PcapReader
    from data.Preprocessing.utils import raw_packet_to_string
    anchors, sources = build_anchors(dataset_dir)
    results = []
    for path, name in captures:
        label = USTC_LABELS[name]
        scanned = 0
        observations = {method: {'packet_hits': 0, 'unique_anchor_packet_hits': 0,
                                 'matching_image_rows': set(), 'matching_multi_slot_rows': set(),
                                 'unique_image_rows': set()}
                        for method in ('current', 'legacy')}
        with PcapReader(str(path)) as packets:
            for i, packet in enumerate(packets):
                if i >= packet_limit:
                    break
                scanned += 1
                current = None
                try:
                    header, payload = raw_packet_to_string(packet)
                    current = bytes.fromhex(header + payload)
                except ValueError:
                    pass
                for method, block in (('current', current), ('legacy', legacy_block(packet))):
                    if block is None:
                        continue
                    matches = anchors.get(hashlib.sha256(block).digest(), ())
                    valid = [m for m in matches if m[2] == label]
                    if not valid:
                        continue
                    out = observations[method]
                    out['packet_hits'] += 1
                    out['unique_anchor_packet_hits'] += len(matches) == 1
                    for file, row, _, length in valid:
                        key = (file, row)
                        out['matching_image_rows'].add(key)
                        if length > 1:
                            out['matching_multi_slot_rows'].add(key)
                        if len(matches) == 1:
                            out['unique_image_rows'].add(key)
        results.append({'class': name, 'capture': Path(path).name,
                        'capture_sha256': sha_file(path), 'packets_scanned': scanned,
                        'methods': {name: {key: len(value) if isinstance(value, set) else value
                                           for key, value in out.items()}
                                    for name, out in observations.items()}})
    return {'schema': 'first-packet-anchor-probe-v1', 'packet_limit': packet_limit,
            'sources': sources, 'captures': results,
            'scope': 'First 128 image bytes only; packet hash match is not a full temporal pairing.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--capture', nargs=2, action='append', metavar=('PCAP', 'CLASS'), required=True)
    p.add_argument('--packet-limit', type=int, default=10000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = probe(a.dataset_dir, a.capture, a.packet_limit)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    for row in report['captures']:
        print(row['class'], row['methods'], flush=True)
