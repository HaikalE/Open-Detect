"""Consume audited exact-image group indices; never regenerate splits at runtime."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision import transforms

from data import dataset as source
from data.splits import get_splits
from provenance import sha256

GROUP_PROTOCOL = 'exact-image-group-stratified-80-10-10-v1-preview'
SCENARIOS = {**{f'A-{i+1}': ('USTC', i) for i in range(3)},
             **{f'B-{i+1}': ('mal', i) for i in range(3)},
             **{f'C-{i+1}': ('combined_USTC_mal', i) for i in range(2)}}
PARTS = ('train', 'validation', 'known_test', 'unknown_test', 'unknown_unused')


def bundle_record(directory, dataset, split, seed):
    directory = Path(directory)
    report_path = directory / 'REPORT.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['protocol'] != GROUP_PROTOCOL:
        raise ValueError('Unexpected grouped split protocol')
    matches = [r for r in report['runs'] if r['dataset'] == dataset
               and r['seed'] == seed and SCENARIOS.get(r['scenario']) == (dataset, split)]
    if len(matches) != 1:
        raise ValueError('Exactly one verified manifest required for scenario/seed')
    record = matches[0]
    path = (directory / record['indices_file']).resolve()
    if path.parent != directory.resolve() or sha256(path) != record['indices_sha256']:
        raise ValueError('Split index path/hash mismatch')
    known, unknown, kd, ud = get_splits(dataset, num_split=split)
    if (record['known_classes'] != known or record['unknown_classes'] != unknown
            or kd != dataset or ud != dataset):
        raise ValueError('Scenario class mapping differs from manifest')
    if record['source_row_order'] != list(source.DATASET_FILES[dataset]):
        raise ValueError('Source concatenation order mismatch')
    identity = {'protocol': GROUP_PROTOCOL, 'scenario': record['scenario'], 'seed': seed,
                'report_sha256': sha256(report_path), 'indices_sha256': sha256(path),
                'source_sha256': {n: report['source_sha256'][n] for n in record['source_row_order']}}
    return report, record, path, identity


def validate_indices(data, labels, indices, known, unknown):
    if set(indices) != set(PARTS):
        raise ValueError('Unexpected index arrays')
    n = len(labels)
    for name, ix in indices.items():
        if ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer) or not len(ix):
            raise ValueError('Invalid/empty indices: ' + name)
        if ix.min() < 0 or ix.max() >= n:
            raise ValueError('Out-of-bounds indices: ' + name)
        expected = known if name in PARTS[:3] else unknown
        if set(np.unique(labels[ix]).tolist()) != set(expected):
            raise ValueError('Wrong or missing class in partition: ' + name)
    combined = np.concatenate([indices[p] for p in PARTS])
    if len(combined) != n or not np.array_equal(np.sort(combined), np.arange(n)):
        raise ValueError('Index partitions do not cover each source row exactly once')
    seen = {}
    for name in PARTS:
        for i in indices[name]:
            key = hashlib.sha256(data[i].tobytes()).digest()
            previous = seen.setdefault(key, (name, int(labels[i])))
            if previous != (name, int(labels[i])):
                raise ValueError('Identical image crosses partitions or labels')


def get_grouped_splits(dataset, split, seed, directory):
    report, record, path, identity = bundle_record(directory, dataset, split, seed)
    for name, expected in identity['source_sha256'].items():
        if sha256(Path(source.data_dir) / 'dataset' / name) != expected:
            raise ValueError('Source NPZ differs from audited split: ' + name)
    data, labels = source._load_npz_arrays(dataset)
    if data.dtype != np.uint8:
        raise ValueError('Audited source must contain uint8 images')
    with np.load(path, allow_pickle=False) as archive:
        indices = {name: archive[name] for name in archive.files}
    known, unknown = record['known_classes'], record['unknown_classes']
    validate_indices(data, labels, indices, known, unknown)
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    datasets = []
    for name in PARTS[:4]:
        ix = indices[name]
        datasets.append(source.TrafficArrayDataset(
            data[ix], labels[ix], transform=aug if name == 'train' else transforms.ToTensor(),
            select_classes=unknown if name == 'unknown_test' else known,
            target_transform='open' if name == 'unknown_test' else 'reindex'))
    return (*datasets, identity)


def verify_checkpoint_split(checkpoint, identity):
    expected_protocol = identity['protocol'] if identity else None
    if identity and (checkpoint.get('protocol') != expected_protocol
                     or checkpoint.get('split_identity') != identity):
        raise ValueError('Checkpoint does not belong to this grouped split; train from scratch')

