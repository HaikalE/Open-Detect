"""Summarize bounded exact-packet recoveries and audit split/flow collisions."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(paths):
    reports = []
    seen_class = set()
    all_rows = []
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get('schema') != 'exact-packet-block-recovery-v1':
            raise ValueError(f'Unexpected recovery schema: {path}')
        name = report['class']
        if name in seen_class:
            raise ValueError(f'Duplicate class report: {name}')
        seen_class.add(name)
        rows = report['recovered']
        if len(rows) != report['outcomes'].get('full_exact_image_recovery', 0):
            raise ValueError(f'Recovery count mismatch: {path}')
        multi = [row for row in rows if row['length'] > 1]
        if len(multi) != report['outcomes'].get('full_multi_packet_recovery', 0):
            raise ValueError(f'Multi-packet count mismatch: {path}')
        split = Counter(row['file'] for row in multi)
        groups = defaultdict(set)
        for row in multi:
            if not any(x > 0 for x in row['iat_seconds'][1:]):
                raise ValueError(f'Nonpositive multi-packet IAT: {path}')
            groups[row['flow_group_sha256']].add(row['file'])
        reports.append({
            'class': name,
            'report_file': path.name,
            'report_sha256': hashlib.sha256(raw).hexdigest(),
            'capture_sha256': report['capture_sha256'],
            'packets_scanned': report['packets_scanned'],
            'packet_limit': report['packet_limit'],
            'recovered_exact': len(rows),
            'recovered_multi_positive_iat': len(multi),
            'multi_by_npz_partition': dict(split),
            'multi_flow_groups': len(groups),
            'multi_cross_npz_partition_flow_groups': sum(len(files) > 1 for files in groups.values()),
            'outcomes': report['outcomes'],
        })
        all_rows.extend(multi)
    packet_users = defaultdict(set)
    image_users = defaultdict(set)
    group_users = defaultdict(set)
    for row in all_rows:
        uid = (row['file'], row['row'])
        image_users[uid].add(row['capture_sha256'])
        group_users[row['flow_group_sha256']].add(row['file'])
        for index in row['packet_indices']:
            packet_users[(row['capture_sha256'], index)].add(uid)
    return {
        'schema': 'anchored-recovery-coverage-v1',
        'scope': 'Bounded PCAP scans only; candidate cohort, NOT TRAIN READY. No random row split.',
        'classes': sorted(reports, key=lambda item: item['class']),
        'total_multi_positive_iat': len(all_rows),
        'unique_npz_rows': len(image_users),
        'multi_source_image_rows': sum(len(sources) > 1 for sources in image_users.values()),
        'source_packets_reused_by_multiple_images': sum(len(users) > 1 for users in packet_users.values()),
        'cross_npz_partition_flow_groups': sum(len(files) > 1 for files in group_users.values()),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reports)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print({key: value for key, value in result.items() if key != 'classes'})
