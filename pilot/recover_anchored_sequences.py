"""Recover exact image packet blocks and timestamps from a bounded PCAP scan.

Every accepted image slot must match one source packet from the same biflow in
capture order. Ambiguous matches are reported and excluded. This is a source
investigation tool; full-dataset identity and leakage grouping still need review.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from pilot.pairing import USTC_LABELS, sha_file
from pilot.trace_preprocessing import legacy_block


def image_index(dataset_dir):
    anchor = defaultdict(list)
    for part in ('train', 'test'):
        path = Path(dataset_dir) / f'USTC_1c_{part}.npz'
        with np.load(path, allow_pickle=False) as archive:
            images, labels = archive['data'], archive['target']
            if images.dtype != np.uint8 or images.shape[1:] != (32, 32):
                raise ValueError('Unexpected USTC NPZ image schema')
            for row, (image, label) in enumerate(zip(images, labels)):
                blocks = image.reshape(8, 128)
                occupied = np.any(blocks != 0, axis=1)
                if not occupied[0]:
                    continue
                n = int(np.count_nonzero(occupied))
                anchor[hashlib.sha256(blocks[0].tobytes()).digest()].append({
                    'file': path.name, 'row': row, 'label': int(label),
                    'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'blocks': [hashlib.sha256(block.tobytes()).digest() for block in blocks[:n]],
                    'length': n, 'contiguous': bool(np.all(occupied[:n])) and not np.any(occupied[n:])})
    return anchor


def packet_catalog(capture, packet_limit):
    from scapy.all import IP, TCP, UDP, Padding, PcapReader
    by_block = defaultdict(list)
    packet_count = 0
    with PcapReader(str(capture)) as reader:
        for index, packet in enumerate(reader):
            if index >= packet_limit:
                break
            packet_count += 1
            if IP not in packet or (TCP not in packet and UDP not in packet):
                continue
            ip = packet[IP]
            transport = packet[TCP] if TCP in packet else packet[UDP]
            sender = (ip.src, int(transport.sport))
            receiver = (ip.dst, int(transport.dport))
            key = (ip.version, 'TCP' if TCP in packet else 'UDP', tuple(sorted((sender, receiver))))
            copy = transport.copy()
            if Padding in copy:
                copy[Padding].underlayer.remove_payload()
            time = Decimal(str(packet.time))
            if not time.is_finite():
                raise ValueError('Nonfinite timestamp at packet ' + str(index))
            entry = {'index': index, 'time': time, 'flow': key, 'sender': sender,
                     'payload_bytes': len(bytes(copy.payload))}
            by_block[hashlib.sha256(legacy_block(packet)).digest()].append(entry)
    return by_block, packet_count


def recover(dataset_dir, capture, class_name, packet_limit):
    if class_name not in USTC_LABELS:
        raise ValueError('Unknown USTC class')
    anchor = image_index(dataset_dir)
    catalog, scanned = packet_catalog(capture, packet_limit)
    capture_hash = sha_file(capture)
    expected = USTC_LABELS[class_name]
    outcomes = Counter()
    recovered = []
    observed_images = set()
    for first_hash, packets in catalog.items():
        image_rows = anchor.get(first_hash, [])
        if not image_rows:
            continue
        if len(image_rows) != 1:
            outcomes['ambiguous_image_anchor'] += 1
            continue
        image = image_rows[0]
        if image['label'] != expected:
            outcomes['wrong_class_anchor'] += 1
            continue
        image_key = (image['file'], image['row'])
        observed_images.add(image_key)
        if len(packets) != 1:
            outcomes['ambiguous_source_anchor'] += 1
            continue
        if not image['contiguous']:
            outcomes['noncontiguous_image_slots'] += 1
            continue
        first = packets[0]
        chosen = [first]
        status = None
        for block_hash in image['blocks'][1:]:
            eligible = [item for item in catalog.get(block_hash, [])
                        if item['flow'] == first['flow'] and item['index'] > chosen[-1]['index']
                        and item['time'] >= chosen[-1]['time'] - Decimal('0.000002')]
            if not eligible:
                status = 'missing_next_packet'
                break
            if len(eligible) != 1:
                status = 'ambiguous_next_packet'
                break
            chosen.append(eligible[0])
        if status:
            outcomes[status] += 1
            continue
        if len(chosen) != image['length']:
            raise AssertionError('Length mismatch')
        adjusted = [chosen[0]['time']]
        for item in chosen[1:]:
            adjusted.append(max(adjusted[-1], item['time']))
        iat = [float(b-a) for a, b in zip(adjusted, adjusted[1:])]
        if len(chosen) > 1 and not any(value > 0 for value in iat):
            outcomes['no_positive_iat'] += 1
            continue
        outcomes['full_exact_image_recovery'] += 1
        outcomes['full_multi_packet_recovery'] += len(chosen) > 1
        recovered.append({'file': image['file'], 'row': image['row'], 'label': expected,
                          'image_sha256': image['image_sha256'],
                          'capture_sha256': capture_hash, 'capture_class': class_name,
                          'packet_indices': [item['index'] for item in chosen],
                          'timestamps_seconds': [str(item['time']) for item in chosen],
                          'payload_bytes': [item['payload_bytes'] for item in chosen],
                          'direction': [int(item['sender'] != first['sender']) for item in chosen],
                          'iat_seconds': [0.0, *iat],
                          'length': len(chosen),
                          'flow_group_sha256': hashlib.sha256(json.dumps([capture_hash, first['flow']]).encode()).hexdigest()})
    return {'schema': 'exact-packet-block-recovery-v1', 'capture': Path(capture).name,
            'capture_sha256': capture_hash, 'class': class_name, 'packet_limit': packet_limit,
            'packets_scanned': scanned, 'anchored_image_rows': len(observed_images),
            'outcomes': dict(outcomes), 'recovered': recovered,
            'scope': 'Bounded capture scan; every nonzero image slot matched a unique source packet in one biflow.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--class-name', required=True)
    p.add_argument('--packet-limit', type=int, default=10000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.packet_limit <= 0:
        p.error('Packet limit must be positive')
    report = recover(a.dataset_dir, a.capture, a.class_name, a.packet_limit)
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print({key: value for key, value in report.items() if key != 'recovered'})
