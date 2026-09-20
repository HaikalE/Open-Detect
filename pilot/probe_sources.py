"""Bounded source investigation without rebuilding or changing P1 artifacts."""
import argparse
import json
from pathlib import Path

from pilot.pairing import make_index, extract_candidates, annotate_matches, temporal_coverage


def probe(dataset_dir, captures, packet_limit=10000):
    index, datasets = make_index(sorted(Path(dataset_dir).glob('*.npz')))
    results = []
    for path, label in captures:
        try:
            rows, _, scan = extract_candidates(path, label, packet_limit=packet_limit)
            matches = annotate_matches(rows, index)
            results.append({'scan': scan, 'matches': matches, 'coverage': temporal_coverage(rows)})
        except Exception as exc:
            results.append({'file': str(path), 'error': str(exc)})
    return {'datasets': datasets, 'packet_limit': packet_limit, 'results': results,
            'training_ready': False, 'scope': 'bounded source probe, not full-source uniqueness or split validation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--capture', nargs=2, action='append', metavar=('PCAP', 'CLASS'), required=True)
    parser.add_argument('--packet-limit', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.packet_limit < 1:
        parser.error('--packet-limit must be positive')
    result = probe(args.dataset_dir, args.capture, args.packet_limit)
    # Never overwrite a previous source investigation.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
