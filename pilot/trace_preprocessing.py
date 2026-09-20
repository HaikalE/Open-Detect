"""Diagnostic only: compare upstream legacy serialization on existing P1 windows.

Does not change sessionization, labels, datasets, or authorize training.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pilot.pairing import make_index, sha_file


def legacy_block(packet):
    # Mirrors upstream utils.py but copies first to avoid its IP-mutation side effect.
    p = packet.copy()
    try:
        ip = p['IP']
        ip.src = ip.dst = '0.0.0.0'
        header = bytes(ip).hex()
        try:
            payload = bytes(p['Raw']).hex()
            header = header.replace(payload, '')
        except IndexError:
            payload = ''
        return bytes.fromhex(header[:160].ljust(160, '0') + payload[:96].ljust(96, '0'))
    except IndexError:
        return bytes(128)


def trace(manifest, dataset_dir, raw_root):
    from scapy.all import PcapReader
    rows = [json.loads(line) for line in Path(manifest).open(encoding='utf-8')]
    index, _ = make_index(sorted(Path(dataset_dir).glob('*.npz')))
    groups = defaultdict(list)
    for row in rows:
        groups[row['capture_file']].append(row)
    report = {'manifest_sha256': sha_file(manifest), 'captures': [],
              'scope': 'same P1 packet selections, legacy serializer; no full-source pairing proof'}
    for filename, group in groups.items():
        paths = list(Path(raw_root).rglob(filename))
        if len(paths) != 1:
            raise ValueError('Capture path missing or ambiguous: ' + filename)
        path = paths[0]
        if sha_file(path) != group[0]['capture_sha256']:
            raise ValueError('Capture SHA changed')
        wanted = {i for row in group for i in row['packet_indices']}
        blocks = {}
        with PcapReader(str(path)) as packets:
            for i, packet in enumerate(packets):
                if i in wanted:
                    blocks[i] = legacy_block(packet)
                if len(blocks) == len(wanted):
                    break
        counts = Counter()
        for row in group:
            legacy = b''.join(blocks[i] for i in row['packet_indices']).ljust(1024,b'\x00')
            key = hashlib.sha256(legacy).hexdigest()
            counts['same_as_p1'] += key == row['image_sha256']
            matches = [m for m in index.get(key,[]) if m['dataset']=='USTC']
            valid = matches and all(m['label']==row['expected_ustc_label'] for m in matches)
            counts['legacy_matching_candidates'] += bool(valid)
            counts['legacy_matching_multi_packet'] += bool(valid and row['length']>1)
            counts['legacy_label_conflicts'] += bool(matches and not valid)
        report['captures'].append({'file': filename, 'candidates': len(group), **counts})
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = trace(a.manifest,a.dataset_dir,a.raw_root)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
