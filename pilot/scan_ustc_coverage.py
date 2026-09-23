"""Bounded USTC coverage scan: one source capture for each of its 20 classes.

Read-only source probe. Per-class results are checkpointed for interrupted runs.
This intentionally does not construct train/test splits or fusion inputs.
"""
import argparse
import csv
import json
from pathlib import Path

from pilot.pairing import USTC_LABELS, annotate_matches, extract_candidates, make_index, temporal_coverage


def select_captures(root):
    root = Path(root)
    options = {}
    for path in root.rglob('*.pcap'):
        parts = path.relative_to(root).parts
        if len(parts) < 2 or parts[0] not in ('Benign', 'Malware'):
            continue
        name = parts[1] if len(parts) > 2 else path.stem
        if name not in USTC_LABELS:
            continue
        rank = (path.stat().st_size, str(path))
        if name not in options or rank < options[name][0]:
            options[name] = rank, path
    missing = sorted(set(USTC_LABELS) - set(options))
    if missing:
        raise ValueError('Missing USTC class PCAP: ' + ', '.join(missing))
    return [(name, options[name][1]) for name in sorted(options)]


def scan(root, dataset_dir, output, packet_limit, classes=None, timestamp_tolerance_us=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, datasets = make_index(sorted(Path(dataset_dir).glob('*.npz')))
    selected = select_captures(root)
    if classes is not None:
        requested = set(classes)
        missing = requested - {name for name, _ in selected}
        if missing:
            raise ValueError('Unknown selected class: ' + ', '.join(sorted(missing)))
        selected = [(name, path) for name, path in selected if name in requested]
    if not selected:
        raise ValueError('No classes selected')
    policy = ('biflow-idle60s-syn-seq-rst-v1-first8' if timestamp_tolerance_us == 0 else
              f'biflow-idle60s-syn-seq-rst-v1-first8-tol{timestamp_tolerance_us}us')
    rows = []
    for name, path in selected:
        destination = output / (name + '.json')
        if destination.exists():
            previous = json.loads(destination.read_text(encoding='utf-8'))
            # A completed source scan is immutable on resume. Never silently
            # accept another file, code policy, or limit in an old output dir.
            if (previous['source_path'] != str(path.resolve()) or
                    previous['packet_limit'] != packet_limit or
                    previous['source_bytes'] != path.stat().st_size or
                    previous['policy'] != policy):
                raise ValueError('Incompatible previous scan for ' + name)
            item = previous
        else:
            try:
                candidates, _, source = extract_candidates(path, name, packet_limit=packet_limit,
                    session_limit=packet_limit, timestamp_tolerance_us=timestamp_tolerance_us)
                matches = annotate_matches(candidates, index)
                coverage = temporal_coverage(candidates).get(name, {})
                item = {'class': name, 'source_path': str(path.resolve()),
                        'source_bytes': path.stat().st_size,
                        'packet_limit': packet_limit,
                        'policy': policy,
                        'source': source, 'matches': matches, 'coverage': coverage}
            except Exception as exc:
                item = {'class': name, 'source_path': str(path.resolve()),
                        'source_bytes': path.stat().st_size,
                        'packet_limit': packet_limit,
                        'policy': policy,
                        'error': str(exc)}
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(item, handle, indent=2)
        coverage = item.get('coverage', {})
        rows.append({'class': name, 'capture': path.name,
                     'packets_scanned': item.get('source', {}).get('packets_scanned'),
                     'capture_complete': item.get('source', {}).get('complete'),
                     'candidate_sessions': coverage.get('candidates'),
                     'timestamp_adjustments': item.get('source', {}).get('timestamp_adjustments'),
                     'unique_pilot_matches': coverage.get('unique_pilot_matches'),
                     'matched_multi_packet': coverage.get('matched_multi_packet_candidates'),
                     'matched_positive_iat': coverage.get('matched_with_positive_iat'),
                     'error': item.get('error', '')})
        print(f"{name}: {rows[-1]['matched_multi_packet']} multi-packet matches; "
              f"{rows[-1]['packets_scanned']} packets scanned", flush=True)
    with (output / 'coverage.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'schema': 'ustc-coverage-v1', 'packet_limit_per_capture': packet_limit,
               'selected_classes': [name for name, _ in selected],
               'policy': policy,
               'source_datasets': datasets, 'classes': rows,
               'classes_scanned': sum(not r['error'] for r in rows),
               'classes_with_multi_packet_matches': [r['class'] for r in rows if (r['matched_multi_packet'] or 0) > 0],
               'total_matched_multi_packet': sum(r['matched_multi_packet'] or 0 for r in rows),
               'training_ready': False,
               'coverage_note': 'One selected capture per class, first N packets only. Uniqueness is local to each capture.'}
    (output / 'coverage_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ustc-root', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet-limit', type=int, default=10000)
    parser.add_argument('--classes', nargs='+', help='Only these USTC classes')
    parser.add_argument('--timestamp-tolerance-us', type=int, default=0)
    args = parser.parse_args()
    if args.packet_limit <= 0:
        parser.error('packet limit must be positive')
    result = scan(args.ustc_root, args.dataset_dir, args.output, args.packet_limit,
                  classes=args.classes, timestamp_tolerance_us=args.timestamp_tolerance_us)
    print('Classes with multi-packet matches:', result['classes_with_multi_packet_matches'])
    print('Total:', result['total_matched_multi_packet'])
