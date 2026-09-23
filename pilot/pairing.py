"""P1 candidate construction, not proof of the original sessionization protocol."""
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

POLICY = 'biflow-idle60s-syn-seq-rst-v1-first8'
# Evidence: data/splits.py comment. Verify against labels of actual hash matches.
USTC_LABELS = dict(zip(('Gmail FTP Nsis-ay Facetime Weibo Cridex Zeus SMB BitTorrent '
                        'WorldOfWarcraft Shifu Outlook Virut Geodo MySQL Htbot Tinba Skype Miuref Neris').split(), range(20)))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def make_index(paths):
    """No pooling of label namespaces, no guessed row correspondence."""
    index = defaultdict(list)
    sources = []
    for path in paths:
        path = Path(path)
        namespace = 'USTC' if 'USTC_1c_' in path.name else 'combined' if 'combined_' in path.name else 'mal'
        with np.load(path, allow_pickle=False) as data:
            x, y = data['data'], data['target']
            if x.dtype != np.uint8 or x.shape[1:] != (32, 32) or y.shape != (len(x),):
                raise ValueError(f'Unsupported NPZ schema: {path.name}')
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError('Non-integer labels')
            for row, (image, label) in enumerate(zip(x, y)):
                index[hashlib.sha256(image.tobytes()).hexdigest()].append(
                    {'dataset': namespace, 'file': path.name, 'row': row, 'label': int(label)})
            sources.append({'file': path.name, 'sha256': sha_file(path), 'rows': len(x), 'dataset': namespace})
    return index, sources


def extract_candidates(path, capture_class, packet_limit=200000, session_limit=10000,
                       timestamp_tolerance_us=0):
    """Read in capture order; preserve exact decimal times before IAT subtraction.

    Biflow key: IP version, TCP/UDP, unordered endpoint pair. Split on idle>60s,
    a new initiating SYN sequence (not retransmissions), or first packet after RST.
    FIN alone does not split: its ACK may still belong to that session. This is an
    explicit candidate policy, not a claim to reproduce an unknown original tool.
    """
    from scapy.all import IP, IPv6, TCP, Padding, PcapReader
    from data.Preprocessing.utils import raw_packet_to_string
    if timestamp_tolerance_us < 0:
        raise ValueError('timestamp_tolerance_us must be nonnegative')
    policy = POLICY if timestamp_tolerance_us == 0 else POLICY + f'-tol{timestamp_tolerance_us}us'
    tolerance = Decimal(timestamp_tolerance_us) / Decimal(1000000)
    path = Path(path)
    capture_hash = sha_file(path)
    states, sessions, errors = {}, [], []
    scanned = excluded = timestamp_adjustments = 0
    complete = True
    with PcapReader(str(path)) as packets:
        for packet_id, packet in enumerate(packets):
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
            transport = ip[TCP] if TCP in ip else ip.payload
            # IPv6 extension headers: select actual UDP layer explicitly.
            if TCP not in ip:
                from scapy.all import UDP
                transport = ip[UDP]
            src, dst = (ip.src, int(transport.sport)), (ip.dst, int(transport.dport))
            key = (ip.version, 'TCP' if TCP in ip else 'UDP', tuple(sorted((src, dst))))
            now = Decimal(str(packet.time))
            if not now.is_finite():
                raise ValueError(f'Nonfinite capture timestamp at packet {packet_id}')
            syn = (src, int(transport.seq)) if TCP in ip and transport.flags.S and not transport.flags.A else None
            state = states.get(key)
            if state is not None and now < state['last']:
                if state['last'] - now <= tolerance:
                    # Explicit quantization tolerance. Preserve packet order;
                    # a sub-microsecond reversal contributes zero IAT.
                    now = state['last']
                    timestamp_adjustments += 1
                else:
                    raise ValueError(f'Nonmonotonic per-biflow timestamp at packet {packet_id}')
            new = state is None or now - state['last'] > 60 or state['reset'] or (
                syn is not None and syn != state['syn'])
            if new:
                if len(sessions) >= session_limit:
                    complete = False
                    break
                identity = json.dumps([capture_hash, policy, key, packet_id], separators=(',', ':'))
                state = {'id': hashlib.sha256(identity.encode()).hexdigest(), 'first': src,
                         'last': now, 'syn': syn, 'reset': False, 'blocks': [], 'seq': [],
                         'packet_ids': [], 'times': [], 'total': 0}
                sessions.append(state)
                states[key] = state
            if len(state['blocks']) < 8:
                copy = transport.copy()
                if Padding in copy:
                    copy[Padding].underlayer.remove_payload()
                delta = Decimal(0) if not state['times'] else now - Decimal(state['times'][-1])
                state['seq'].append([len(bytes(copy.payload)), int(src != state['first']), float(delta)])
                state['blocks'].append(header + payload)
                state['packet_ids'].append(packet_id)
                state['times'].append(str(now))
            state['last'] = now
            state['total'] += 1
            state['reset'] = bool(TCP in ip and transport.flags.R)
    records, images, sequences, lengths = [], [], [], []
    for state in sessions:
        n = len(state['blocks'])
        image = np.frombuffer(bytes.fromhex(''.join(state['blocks'])).ljust(1024, b'\x00'), dtype=np.uint8).reshape(32,32)
        seq = np.zeros((8,3), dtype=np.float64)
        seq[:n] = state['seq']
        if not np.isfinite(seq).all():
            raise ValueError('Nonfinite sequence values')
        records.append({'sample_id': state['id'], 'capture_sha256': capture_hash,
                        'capture_file': path.name, 'capture_class': capture_class,
                        'expected_ustc_label': USTC_LABELS.get(capture_class),
                        'label_source': 'data/splits.py comment; capture-class mapping not independently verified',
                        'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                        'packet_indices': state['packet_ids'], 'timestamps_seconds': state['times'],
                        'length': n, 'session_observed_packets': state['total'],
                        'capture_complete': complete, 'policy': policy, 'split': None,
                        'pairing_verified': False})
        images.append(image)
        sequences.append(seq)
        lengths.append(n)
    arrays = {'images': np.asarray(images, dtype=np.uint8).reshape(-1,32,32),
              'sequences': np.asarray(sequences, dtype=np.float64).reshape(-1,8,3),
              'lengths': np.asarray(lengths, dtype=np.int64)}
    arrays['mask'] = np.arange(8)[None,:] < arrays['lengths'][:,None]
    return records, arrays, {'capture_file': path.name, 'capture_sha256': capture_hash,
        'class': capture_class, 'packets_scanned': scanned, 'excluded_packets': excluded,
        'sessions': len(records), 'complete': complete, 'policy': policy,
        'timestamp_tolerance_us': timestamp_tolerance_us,
        'timestamp_adjustments': timestamp_adjustments, 'errors': errors}


def annotate_matches(records, index):
    counts = defaultdict(int)
    for row in records:
        counts[row['image_sha256']] += 1
    summary = defaultdict(int)
    for row in records:
        matches = index.get(row['image_sha256'], [])
        row['matching_rows'] = matches
        row['candidate_hash_multiplicity'] = counts[row['image_sha256']]
        expected = row['expected_ustc_label']
        consistent = expected is not None and all(
            m['label'] == expected if m['dataset'] == 'USTC' else
            m['label'] == expected + 24 if m['dataset'] == 'combined' else False
            for m in matches)
        source_rows = [m for m in matches if m['dataset'] == 'USTC']
        status = ('NO_MATCH' if not matches else 'LABEL_CONFLICT' if not consistent else
                  'AMBIGUOUS_HASH' if counts[row['image_sha256']] > 1 or len(source_rows) != 1 else
                  'UNIQUE_IN_PILOT_ONLY')
        row['match_status'] = status
        summary[status] += 1
    return dict(summary)


def temporal_coverage(records):
    result = {}
    for row in records:
        entry = result.setdefault(row['capture_class'], {
            'candidates': 0, 'length_histogram': {}, 'unique_pilot_matches': 0,
            'matched_multi_packet_candidates': 0, 'matched_with_positive_iat': 0})
        entry['candidates'] += 1
        length = str(row['length'])
        entry['length_histogram'][length] = entry['length_histogram'].get(length, 0) + 1
        if row['match_status'] == 'UNIQUE_IN_PILOT_ONLY':
            entry['unique_pilot_matches'] += 1
            entry['matched_multi_packet_candidates'] += int(row['length'] > 1)
            times = [Decimal(t) for t in row['timestamps_seconds']]
            entry['matched_with_positive_iat'] += int(any(b > a for a, b in zip(times, times[1:])))
    return result


def run_pairing(npz_paths, captures, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index, sources = make_index(npz_paths)
    records, batches, scans, errors = [], [], [], []
    for path, capture_class in captures:
        try:
            rows, arrays, scan = extract_candidates(path, capture_class)
            records.extend(rows)
            batches.append(arrays)
            scans.append(scan)
        except Exception as exc:
            errors.append({'file': Path(path).name, 'error': str(exc)})
    matches = annotate_matches(records, index)
    for i, row in enumerate(records):
        row['array_row'] = i
    manifest = output/'candidate_manifest.jsonl'
    with manifest.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    if batches:
        np.savez_compressed(output/'paired_candidates_NOT_TRAIN_READY.npz',
            **{k: np.concatenate([a[k] for a in batches]) for k in batches[0]})
    summary = {'schema': 'temporal-pairing-pilot-v1', 'policy': POLICY,
               'features': ['transport_payload_bytes', 'direction_first_sender_0', 'iat_seconds'],
               'normalization': 'none; fit on future train split only',
               'datasets': sources, 'captures': scans, 'errors': errors,
               'candidates': len(records), 'match_counts': matches,
               'temporal_coverage': temporal_coverage(records),
               'training_ready': False, 'scope': 'USTC small-capture pilot, not all scenarios',
               'remaining_gates': ['original session/window rules and label provenance',
                   'full-source collision audit (pilot uniqueness is not global uniqueness)',
                   'matched sequences with multiple packets and usable timing',
                   'leakage-safe split and baseline comparability', 'Malicious_TLS raw timestamped PCAP source'],
               'artifacts': {p.name: {'sha256': sha_file(p), 'bytes': p.stat().st_size}
                             for p in output.iterdir() if p.is_file()}}
    (output/'pairing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
