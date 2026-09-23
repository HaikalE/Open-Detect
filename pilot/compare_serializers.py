"""Compare two documented byte encodings on identical source packet windows."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pilot.pairing import USTC_LABELS, annotate_matches, extract_candidates, make_index
from pilot.trace_preprocessing import legacy_block


def compare(dataset_dir, captures, packet_limit):
    from scapy.all import PcapReader
    index, sources = make_index(sorted(Path(dataset_dir).glob('*.npz')))
    results = []
    for path, name in captures:
        path = Path(path)
        try:
            rows, _, scan = extract_candidates(path, name, packet_limit=packet_limit,
                                                session_limit=packet_limit)
            current = annotate_matches(rows, index)
            wanted = {i for row in rows for i in row['packet_indices']}
            blocks = {}
            with PcapReader(str(path)) as packets:
                for i, packet in enumerate(packets):
                    if i in wanted:
                        blocks[i] = legacy_block(packet)
                    if len(blocks) == len(wanted):
                        break
            legacy = defaultdict(int)
            for row in rows:
                key = hashlib.sha256(b''.join(blocks[i] for i in row['packet_indices'])
                                     .ljust(1024, b'\x00')).hexdigest()
                matches = [m for m in index.get(key, []) if m['dataset'] == 'USTC']
                if matches and all(m['label'] == USTC_LABELS[name] for m in matches):
                    legacy['matching_candidates'] += 1
                    legacy['matching_multi_packet'] += int(row['length'] > 1)
                if matches and not all(m['label'] == USTC_LABELS[name] for m in matches):
                    legacy['label_conflicts'] += 1
            results.append({'class': name, 'capture': path.name, 'scan': scan,
                            'current': current, 'legacy': dict(legacy)})
        except Exception as exc:
            results.append({'class': name, 'capture': path.name, 'error': str(exc)})
    return {'schema': 'serializer-comparison-v1', 'packet_limit': packet_limit,
            'datasets': sources, 'results': results,
            'scope': 'Same P1 session/window packet indices; not a new sessionization rule.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--capture', nargs=2, action='append', metavar=('PCAP', 'CLASS'), required=True)
    p.add_argument('--packet-limit', type=int, default=10000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = compare(a.dataset_dir, a.capture, a.packet_limit)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    for row in report['results']:
        print(row['class'], row.get('current'), row.get('legacy'), row.get('error'), flush=True)
