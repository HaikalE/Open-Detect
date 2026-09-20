"""CPU-only evidence audit. Hash matches are candidates, never pairing proof."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def image_hash(image):
    image = np.asarray(image)
    if image.size != 1024 or image.dtype != np.uint8:
        raise ValueError('Expected exactly 1024 uint8 image bytes; no silent conversion')
    return hashlib.sha256(image.tobytes(order='C')).hexdigest()


def audit_npz(path, index):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        x, y = data['data'], data['target']
        if x.ndim < 2 or np.prod(x.shape[1:]) != 1024 or x.dtype != np.uint8:
            raise ValueError(f'{path.name}: unsupported image schema {x.shape}/{x.dtype}')
        if y.ndim != 1 or len(x) != len(y) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f'{path.name}: invalid integer labels')
        for row, (image, label) in enumerate(zip(x, y)):
            key = image_hash(image)
            entry = index.setdefault(key, {'count': 0, 'examples': [], 'labels': set()})
            entry['count'] += 1
            entry['labels'].add(int(label))
            if len(entry['examples']) < 4:
                entry['examples'].append({'file': path.name, 'row': row, 'label': int(label)})
        labels, counts = np.unique(y, return_counts=True)
        return {'file': path.name, 'keys': keys, 'shape': list(x.shape),
                'dtype': str(x.dtype), 'samples': len(x),
                'class_counts': {str(k): int(v) for k, v in zip(labels, counts)},
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'explicit_flow_id_present': any(k in keys for k in ('flow_id', 'pcap_id', 'sample_id'))}


def audit_pcap(path, index, packet_limit=10000):
    from scapy.all import IP, IPv6, TCP, UDP, Padding, PcapReader
    from data.Preprocessing.utils import raw_packet_to_string

    path = Path(path)
    flows, blocks, seq = set(), [], []
    first_endpoint = previous_time = None
    valid = excluded = scanned = 0
    complete = True
    # Streaming: never rdpcap() a multi-gigabyte capture.
    with PcapReader(str(path)) as packets:
        for packet in packets:
            if scanned >= packet_limit:
                complete = False
                break
            scanned += 1
            try:
                header, payload = raw_packet_to_string(packet)
            except ValueError:
                excluded += 1
                continue
            ip = packet[IP] if IP in packet else packet[IPv6]
            transport = ip[TCP] if TCP in ip else ip[UDP]
            src, dst = (ip.src, int(transport.sport)), (ip.dst, int(transport.dport))
            flow = (ip.version, 'TCP' if TCP in ip else 'UDP', tuple(sorted((src, dst))))
            flows.add(flow)
            valid += 1
            if len(flows) > 1:
                # A capture is not a flow; refuse to fuse unrelated conversations.
                return {'file': path.name, 'status': 'MULTIFLOW_NEEDS_SESSIONIZATION',
                        'scanned_packets': scanned, 'observed_biflows_at_least': len(flows),
                        'pairing_verified': False}
            if len(blocks) < 8:
                if first_endpoint is None:
                    first_endpoint = src
                timestamp = float(packet.time)
                iat = 0.0 if previous_time is None else timestamp - previous_time
                if not np.isfinite(timestamp) or not np.isfinite(iat) or iat < 0:
                    return {'file': path.name, 'status': 'INVALID_TIMESTAMPS', 'pairing_verified': False}
                transport_copy = transport.copy()
                if Padding in transport_copy:
                    transport_copy[Padding].underlayer.remove_payload()
                seq.append([len(bytes(transport_copy.payload)), int(src != first_endpoint), iat])
                blocks.append(header + payload)
                previous_time = timestamp
    result = {'file': path.name, 'scanned_packets': scanned, 'valid_packets': valid,
              'excluded_packets': excluded, 'capture_fully_scanned': complete,
              'pairing_verified': False}
    if not blocks:
        return dict(result, status='NO_VALID_PACKETS')
    if not complete:
        return dict(result, status='SCAN_LIMIT_NO_PAIRING_CLAIM')
    image = np.frombuffer(bytes.fromhex(''.join(blocks)).ljust(1024, b'\x00'), dtype=np.uint8)
    key = image_hash(image)
    match = index.get(key, {'count': 0, 'examples': [], 'labels': set()})
    return dict(result, status='HASH_CANDIDATE_ONLY' if match['count'] else 'NO_HASH_MATCH',
                image_sha256=key, matching_rows=match['count'], examples=match['examples'],
                sequence_length=len(seq), sequence_features=['transport_payload_bytes', 'direction', 'iat_seconds'],
                first_iat_seconds=seq[0][2],
                note='A single biflow tuple does not prove a unique session. Check provenance, labels, windows and splits.')


def audit_paths(npz_paths, pcap_paths=(), packet_limit=10000):
    index, report = {}, {'schema': 'temporal-pilot-audit-v1', 'datasets': [], 'pcaps': [], 'errors': []}
    for path in npz_paths:
        try:
            report['datasets'].append(audit_npz(path, index))
        except Exception as exc:
            report['errors'].append({'file': Path(path).name, 'error': str(exc)})
    for path in pcap_paths:
        try:
            report['pcaps'].append(audit_pcap(path, index, packet_limit))
        except Exception as exc:
            report['errors'].append({'file': Path(path).name, 'error': str(exc)})
    report['unique_image_hashes'] = len(index)
    # Dataset label namespaces differ; do not compare labels across NPZ datasets.
    report['training_ready'] = False
    report['next_step'] = ('Build and verify a full paired manifest: source capture/session/window, image hash, '
                           'packet indices/times, label mapping and leakage-safe split. '
                           'A sampled hash audit alone never authorizes training.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--pcap', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_paths(sorted(args.dataset_dir.glob('*.npz')), args.pcap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
