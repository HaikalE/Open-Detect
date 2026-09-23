"""Build a bounded, flow-disjoint USTC feasibility cohort; not a thesis benchmark."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from pilot.pairing import USTC_LABELS, sha_file


SPLIT_ID = {'train': 0, 'val': 1, 'test': 2}
NPZ_FILES = ('USTC_1c_train.npz', 'USTC_1c_test.npz')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _check_temporal(row):
    n = row['length']
    keys = ('packet_indices', 'timestamps_seconds', 'payload_bytes', 'direction', 'iat_seconds')
    if not 2 <= n <= 8 or any(len(row[key]) != n for key in keys):
        raise ValueError('Invalid sequence length')
    if len(set(row['packet_indices'])) != n or row['packet_indices'] != sorted(row['packet_indices']):
        raise ValueError('Packet indices not strictly increasing')
    if row['direction'][0] != 0 or any(v not in (0, 1) for v in row['direction']):
        raise ValueError('Invalid direction')
    if any(not isinstance(v, int) or v < 0 for v in row['payload_bytes']):
        raise ValueError('Invalid payload length')
    raw = [Decimal(t) for t in row['timestamps_seconds']]
    if any(not t.is_finite() for t in raw):
        raise ValueError('Nonfinite timestamp')
    adjusted = [raw[0]]
    for t in raw[1:]:
        if t < adjusted[-1] - Decimal('0.000002'):
            raise ValueError('Timestamp reversal exceeds 2 us')
        adjusted.append(max(adjusted[-1], t))
    expected = [0.0, *(float(b-a) for a, b in zip(adjusted, adjusted[1:]))]
    if any(abs(a-b) > 1e-9 for a, b in zip(expected, row['iat_seconds'])):
        raise ValueError('IAT does not match source timestamps')
    if not any(v > 0 for v in expected[1:]):
        raise ValueError('No positive IAT')


def build(dataset_dir, report_paths, unknown_class='Tinba', seed=2026):
    if unknown_class not in USTC_LABELS:
        raise ValueError('Unknown holdout class')
    dataset_dir = Path(dataset_dir)
    sources = {}
    for name in NPZ_FILES:
        path = dataset_dir / name
        with np.load(path, allow_pickle=False) as archive:
            images = archive['data']
            labels = archive['target']
            if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (32, 32):
                raise ValueError(f'Unexpected image schema: {name}')
            if labels.shape != (len(images),):
                raise ValueError(f'Unexpected label schema: {name}')
            sources[name] = (images, labels)
    records = []
    seen_reports = set()
    seen_rows, seen_hashes, seen_packets = set(), set(), set()
    report_identities = []
    for path in report_paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding='utf-8'))
        name = report.get('class')
        if report.get('schema') != 'exact-packet-block-recovery-v1' or name not in USTC_LABELS:
            raise ValueError(f'Invalid report: {path}')
        if name in seen_reports:
            raise ValueError(f'Duplicate class report: {name}')
        seen_reports.add(name)
        report_identities.append({'name': path.name, 'sha256': sha_file(path),
                                  'capture_sha256': report['capture_sha256'], 'class': name,
                                  'packet_limit': report['packet_limit']})
        rows = report['recovered']
        if len(rows) != report['outcomes'].get('full_exact_image_recovery', 0):
            raise ValueError(f'Report recovery count mismatch: {path}')
        for row in rows:
            if row['length'] == 1:
                continue
            if row['label'] != USTC_LABELS[name] or row['capture_sha256'] != report['capture_sha256']:
                raise ValueError('Label/capture identity mismatch')
            source_name, index = row['file'], row['row']
            if source_name not in sources or not isinstance(index, int):
                raise ValueError('Invalid NPZ source row')
            images, labels = sources[source_name]
            if not 0 <= index < len(images) or int(labels[index]) != row['label']:
                raise ValueError('NPZ label/index mismatch')
            image = images[index]
            if _sha(image.tobytes()) != row['image_sha256']:
                raise ValueError('NPZ image hash mismatch')
            occupied = np.any(image.reshape(8, 128) != 0, axis=1)
            if not np.all(occupied[:row['length']]) or np.any(occupied[row['length']:]):
                raise ValueError('Image packet slots do not match recovered length')
            _check_temporal(row)
            uid = (source_name, index)
            if uid in seen_rows or row['image_sha256'] in seen_hashes:
                raise ValueError('Duplicate image/NPZ row across candidate reports')
            seen_rows.add(uid)
            seen_hashes.add(row['image_sha256'])
            for packet_index in row['packet_indices']:
                key = (row['capture_sha256'], packet_index)
                if key in seen_packets:
                    raise ValueError('Source packet reused by candidate images')
                seen_packets.add(key)
            feature = np.zeros((8, 3), dtype=np.float32)
            feature[:row['length'], 0] = row['payload_bytes']
            feature[:row['length'], 1] = row['direction']
            feature[:row['length'], 2] = row['iat_seconds']
            records.append({'image': image.copy(), 'feature': feature, 'source': source_name,
                            'row': index, 'label': row['label'], 'class': name,
                            'length': row['length'], 'flow': row['flow_group_sha256'],
                            'image_sha256': row['image_sha256'],
                            'capture_sha256': row['capture_sha256'],
                            'packet_indices': row['packet_indices']})
    if unknown_class not in seen_reports or not any(r['class'] == unknown_class for r in records):
        raise ValueError('No unknown-holdout samples')
    known_names = sorted({r['class'] for r in records} - {unknown_class})
    if len(known_names) < 2:
        raise ValueError('At least two known classes required')
    groups = defaultdict(list)
    group_labels = {}
    for i, record in enumerate(records):
        group = record['flow']
        if group in group_labels and group_labels[group] != record['label']:
            raise ValueError('Cross-label flow group')
        group_labels[group] = record['label']
        groups[(record['class'], group)].append(i)
    assignments = {}
    for name in known_names:
        class_groups = [key for key in groups if key[0] == name]
        if len(class_groups) < 3:
            raise ValueError(f'Too few flow groups for three-way split: {name}')
        class_groups.sort(key=lambda key: _sha(f'{seed}|{name}|{key[1]}'.encode()))
        n = len(class_groups)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        for i, key in enumerate(class_groups):
            assignments[key] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
    for key in groups:
        if key[0] == unknown_class:
            assignments[key] = 'test'
    records.sort(key=lambda r: (r['label'], r['source'], r['row']))
    manifest = []
    images, features, masks, lengths, labels, split_ids = [], [], [], [], [], []
    counts = defaultdict(Counter)
    for record in records:
        split = assignments[(record['class'], record['flow'])]
        counts[record['class']][split] += 1
        images.append(record['image'])
        features.append(record['feature'])
        mask = np.zeros(8, dtype=np.uint8)
        mask[:record['length']] = 1
        masks.append(mask)
        lengths.append(record['length'])
        labels.append(record['label'])
        split_ids.append(SPLIT_ID[split])
        manifest.append({key: value for key, value in record.items() if key not in ('image', 'feature')}
                        | {'split': split})
    old_partitions = defaultdict(set)
    for record in records:
        old_partitions[record['flow']].add(record['source'])
    summary = {'schema': 'bounded-ustc-mini-cohort-v1',
               'status': 'FEASIBILITY_ONLY_NOT_THESIS_EVALUATION',
               'unknown_holdout_class': unknown_class, 'known_classes': known_names,
               'split_policy': 'SHA256(seed|class|flow), per-class 70/15/15 flow groups; unknown test only',
               'seed': seed, 'features': ['transport_payload_bytes', 'direction_from_first_sender', 'iat_seconds'],
               'feature_normalization': 'none; fit on train only before any model run',
               'original_npz_cross_partition_flow_groups': sum(len(v) > 1 for v in old_partitions.values()),
               'samples': len(records), 'class_split_counts': {k: dict(v) for k, v in sorted(counts.items())},
               'source_npz': [{'name': name, 'sha256': sha_file(dataset_dir / name)} for name in NPZ_FILES],
               'source_reports': report_identities,
               'limitations': 'Bounded PCAP prefixes and incomplete class coverage; not an A-1/A-2/A-3 result.'}
    arrays = {'image': np.stack(images), 'sequence': np.stack(features),
              'mask': np.stack(masks), 'length': np.asarray(lengths, dtype=np.uint8),
              'label': np.asarray(labels, dtype=np.int64), 'split_id': np.asarray(split_ids, dtype=np.uint8)}
    return arrays, manifest, summary


def save(output_dir, arrays, manifest, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    npz = output_dir / 'mini_paired_cohort_NOT_THESIS_EVAL.npz'
    with npz.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    manifest_path = output_dir / 'manifest.jsonl'
    with manifest_path.open('x', encoding='utf-8') as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    summary['outputs'] = {'npz_sha256': sha_file(npz), 'manifest_sha256': sha_file(manifest_path)}
    with (output_dir / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--unknown-class', default='Tinba')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data, rows, result = build(args.dataset_dir, args.reports, args.unknown_class, args.seed)
    print(json.dumps(save(args.output_dir, data, rows, result), indent=2))
