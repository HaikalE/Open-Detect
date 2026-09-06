"""Dataset and protocol evidence; never rebalance data to imitate paper counts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np

PROTOCOL = 'five_seeded_stratified_80_10_10_repetitions'
IMPLEMENTATION_VERSION = 'replication-fixes-v3-grouped'
FILES = {
    'USTC': ('USTC_1c_train.npz', 'USTC_1c_test.npz'),
    'mal': ('mal_32_1c_train.npz', 'mal_32_1c_test.npz'),
    'combined_USTC_mal': ('combined_train_data.npz', 'combined_test_data.npz'),
}
# PDF Table II (p. 10), not a target for resampling or synthetic augmentation.
PAPER_USTC = [2000] * 19 + [79]
PAPER_MAL = [2000, 1315, 3000, 3000, 3000, 2453, 3000, 2721,
             3004, 1352, 701, 2978, 2786, 2000, 3000, 2000,
             1148, 3000, 2000, 3126, 3000, 3000, 3000, 1234]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_identity():
    root = Path(__file__).resolve().parent
    names = ['model.py', 'train.py', 'test.py', 'utils.py', 'provenance.py',
             'data/dataset.py', 'data/splits.py', 'networks/resnet.py',
             'networks/__init__.py', 'run_5fold.py', 'data/Preprocessing/utils.py',
             'data/grouped.py', 'resume_support.py', 'run_grouped.py']
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'version': IMPLEMENTATION_VERSION, 'git_commit': commit,
            'source_sha256': {name: sha256(root / name) for name in names}}


def dataset_manifest(dataset, data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'data/dataset'
    counts = {}
    files = {}
    for name in FILES[dataset]:
        path = root / name
        with np.load(path, allow_pickle=False) as z:
            data, labels = z['data'], np.asarray(z['target']).reshape(-1)
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError('Labels must be integers: ' + name)
            if data.size != len(labels) * 1024 or len(data) != len(labels):
                raise ValueError('Expected one 32x32 image per label: ' + name)
            if not np.isfinite(data).all() or data.min() < 0 or data.max() > 255:
                raise ValueError('Invalid byte values: ' + name)
            if not np.equal(data, np.floor(data)).all():
                raise ValueError('Fractional byte values: ' + name)
            ids, totals = np.unique(labels, return_counts=True)
            for i, n in zip(ids, totals):
                counts[int(i)] = counts.get(int(i), 0) + int(n)
            files[name] = {'sha256': sha256(path), 'samples': len(labels),
                           'shape': list(data.shape), 'dtype': str(data.dtype)}
    expected = PAPER_USTC if dataset == 'USTC' else PAPER_MAL
    if dataset == 'combined_USTC_mal':
        expected = PAPER_MAL + PAPER_USTC
    if set(counts) != set(range(len(expected))):
        raise ValueError('Dataset class IDs do not match the configured label space')
    differences = {str(i): {'actual': counts[i], 'paper_table_II': n}
                   for i, n in enumerate(expected) if counts[i] != n}
    if differences:
        warnings.warn('Dataset counts differ from paper Table II: ' + json.dumps(differences), stacklevel=2)
    return {'dataset': dataset, 'files': files, 'total_samples': sum(counts.values()),
            'class_counts': {str(i): n for i, n in sorted(counts.items())},
            'paper_count_differences': differences,
            'limitation': 'Hashes/counts do not verify PCAP provenance, label semantics, or flow-level leakage.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dset', choices=list(FILES), required=True)
    p.add_argument('--data_dir')
    a = p.parse_args()
    print(json.dumps(dataset_manifest(a.dset, a.data_dir), indent=2))
