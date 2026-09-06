import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from data.grouped import (GROUP_PROTOCOL, SCENARIOS, PARTS, bundle_record,
                          get_grouped_splits, validate_indices, verify_checkpoint_split)
from data.splits import get_splits
from provenance import sha256, dataset_manifest
import train
import test as evaluation


class GroupedTests(unittest.TestCase):
    def test_all_bundled_scenarios_and_seeds(self):
        folder = Path(__file__).resolve().parents[1] / 'grouped_manifests'
        for scenario, (dataset, split) in SCENARIOS.items():
            for seed in range(2022, 2027):
                _, record, _, identity = bundle_record(folder, dataset, split, seed)
                self.assertEqual(record['scenario'], scenario)
                self.assertEqual(identity['seed'], seed)

    def test_indices_reject_overlap_wrong_role_missing_row_and_duplicate_image(self):
        labels = np.array([0, 0, 0, 1, 1])
        images = np.arange(5, dtype=np.uint8)[:, None]
        indices = {k: np.array([i]) for i, k in enumerate(PARTS)}
        validate_indices(images, labels, indices, [0], [1])
        for bad in [dict(indices, validation=np.array([0])),
                    dict(indices, unknown_test=np.array([2])),
                    dict(indices, unknown_unused=np.array([], dtype=int))]:
            with self.assertRaises(ValueError):
                validate_indices(images, labels, bad, [0], [1])
        images[1] = images[0]
        with self.assertRaisesRegex(ValueError, 'Identical'):
            validate_indices(images, labels, indices, [0], [1])

    def test_old_or_other_seed_checkpoint_rejected(self):
        identity = {'protocol': GROUP_PROTOCOL, 'seed': 2022}
        for checkpoint in [{}, {'protocol': GROUP_PROTOCOL},
                           {'protocol': GROUP_PROTOCOL, 'split_identity': dict(identity, seed=2023)}]:
            with self.assertRaisesRegex(ValueError, 'train from scratch'):
                verify_checkpoint_split(checkpoint, identity)
        verify_checkpoint_split({'protocol': GROUP_PROTOCOL, 'split_identity': identity}, identity)

    def test_real_grouped_loader_train_evaluate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'dataset'; data.mkdir()
            manifests = root / 'manifests'; manifests.mkdir()
            labels = np.repeat(np.arange(20), 10)
            images = np.random.default_rng(19).integers(0, 256, (200, 32, 32), dtype=np.uint8)
            names = ['USTC_1c_train.npz', 'USTC_1c_test.npz']
            for name, sl in zip(names, [slice(0, 100), slice(100, 200)]):
                np.savez(data / name, data=images[sl], target=labels[sl])
            known, unknown, _, _ = get_splits('USTC', 0)
            indices = {'train': [], 'validation': [], 'known_test': [],
                       'unknown_test': [169], 'unknown_unused': list(range(160, 169))}
            for c in known:
                indices['train'].extend(range(c*10, c*10+8))
                indices['validation'].append(c*10+8)
                indices['known_test'].append(c*10+9)
            path = manifests / 'A-1_seed_2022.indices.npz'
            np.savez(path, **{k: np.array(v) for k, v in indices.items()})
            report = {'protocol': GROUP_PROTOCOL, 'source_sha256': {n: sha256(data/n) for n in names},
                      'runs': [{'dataset': 'USTC', 'scenario': 'A-1', 'seed': 2022,
                                'known_classes': known, 'unknown_classes': unknown,
                                'source_row_order': names, 'indices_file': path.name,
                                'indices_sha256': sha256(path)}]}
            (manifests/'REPORT.json').write_text(json.dumps(report))
            manifest_for = lambda dataset: dataset_manifest(dataset, data)
            common = ['--dset', 'USTC', '--gpu', '-1', '--num_workers', '0',
                      '--save_dir', str(root/'save'), '--split_manifest_dir', str(manifests)]
            with patch('data.dataset.data_dir', str(root)), \
                 patch.object(train, 'dataset_manifest', side_effect=manifest_for), \
                 patch.object(evaluation, 'dataset_manifest', side_effect=manifest_for), \
                 contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sets = get_grouped_splits('USTC', 0, 2022, manifests)
                self.assertEqual([len(d) for d in sets[:4]], [152, 19, 19, 1])
                self.assertEqual(sets[3][0][1], 999)
                args = train.build_parser().parse_args(common + ['--epoch', '1', '--h', '4', '--batch_size', '64'])
                train.train(args)
                ev = evaluation.build_parser().parse_args(common)
                metrics = evaluation.evaluate(ev)
                self.assertEqual(metrics['protocol'], GROUP_PROTOCOL)
                self.assertEqual(metrics['split_identity'], sets[-1])
                self.assertGreaterEqual(metrics['validation_known_acceptance'], .95)
                with path.open('ab') as stream:
                    stream.write(b'changed indices')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    get_grouped_splits('USTC', 0, 2022, manifests)

    @unittest.skipUnless(os.environ.get('OPENDETECT_QA_DATA_ROOT'), 'Full source NPZ QA is opt-in')
    def test_all_40_actual_npz_partitions(self):
        folder = Path(__file__).resolve().parents[1] / 'grouped_manifests'
        with patch('data.dataset.data_dir', os.environ['OPENDETECT_QA_DATA_ROOT']):
            for scenario, (dataset, split) in SCENARIOS.items():
                for seed in range(2022, 2027):
                    with self.subTest(scenario=scenario, seed=seed):
                        sets = get_grouped_splits(dataset, split, seed, folder)
                        self.assertEqual(sets[-1]['scenario'], scenario)


if __name__ == '__main__':
    unittest.main()
