import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pilot.build_mini_cohort import build, save
from pilot.pairing import USTC_LABELS


class MiniCohortTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        train_images, test_images = [], []
        train_labels, test_labels = [], []
        reports = {}
        val = 1
        for name, group_count in (('Geodo', 4), ('Zeus', 4), ('Tinba', 2)):
            rows = []
            for group_index in range(group_count):
                count = 2 if name == 'Geodo' and group_index == 0 else 1
                for repeat in range(count):
                    pixels = np.zeros((8, 128), dtype=np.uint8)
                    pixels[:2] = val
                    image = pixels.reshape(32, 32)
                    source = 'USTC_1c_test.npz' if repeat else 'USTC_1c_train.npz'
                    if repeat:
                        row_index = len(test_images)
                        test_images.append(image)
                        test_labels.append(USTC_LABELS[name])
                    else:
                        row_index = len(train_images)
                        train_images.append(image)
                        train_labels.append(USTC_LABELS[name])
                    rows.append({'file': source, 'row': row_index, 'label': USTC_LABELS[name],
                                 'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                 'capture_sha256': name, 'capture_class': name,
                                 'flow_group_sha256': f'{name}-flow-{group_index}',
                                 'packet_indices': [val * 2, val * 2 + 1],
                                 'timestamps_seconds': [str(val), str(val) + '.1'],
                                 'payload_bytes': [10, 20], 'direction': [0, 1],
                                 'iat_seconds': [0.0, 0.1], 'length': 2})
                    val += 1
            reports[name] = {'schema': 'exact-packet-block-recovery-v1', 'class': name,
                             'capture_sha256': name, 'packet_limit': 100,
                             'outcomes': {'full_exact_image_recovery': len(rows)}, 'recovered': rows}
        np.savez(self.dataset / 'USTC_1c_train.npz', data=np.stack(train_images),
                 target=np.asarray(train_labels))
        np.savez(self.dataset / 'USTC_1c_test.npz', data=np.stack(test_images),
                 target=np.asarray(test_labels))
        self.paths = []
        for name, report in reports.items():
            path = self.root / f'{name}.json'
            path.write_text(json.dumps(report))
            self.paths.append(path)

    def test_flow_disjoint_and_unknown_test_only(self):
        arrays, manifest, summary = build(self.dataset, self.paths)
        self.assertEqual(summary['original_npz_cross_partition_flow_groups'], 1)
        self.assertEqual(len(manifest), 11)
        self.assertEqual(arrays['sequence'].shape, (11, 8, 3))
        self.assertTrue(np.all(arrays['mask'][:, :2] == 1))
        self.assertTrue(np.all(arrays['mask'][:, 2:] == 0))
        group_splits = {}
        for row in manifest:
            group_splits.setdefault(row['flow'], set()).add(row['split'])
            if row['class'] == 'Tinba':
                self.assertEqual(row['split'], 'test')
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        for name in ('Geodo', 'Zeus'):
            self.assertEqual({row['split'] for row in manifest if row['class'] == name},
                             {'train', 'val', 'test'})
        out = self.root / 'cohort'
        save(out, arrays, manifest, summary)
        with np.load(out / 'mini_paired_cohort_NOT_THESIS_EVAL.npz') as archive:
            self.assertEqual(archive['image'].shape, (11, 32, 32))
        self.assertTrue((out / 'summary.json').exists())

    def test_hash_mismatch_rejected(self):
        report = json.loads(self.paths[0].read_text())
        report['recovered'][0]['image_sha256'] = '0' * 64
        self.paths[0].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'image hash mismatch'):
            build(self.dataset, self.paths)

    def test_unsupported_iat_rejected(self):
        report = json.loads(self.paths[0].read_text())
        report['recovered'][0]['iat_seconds'][1] = 0.2
        self.paths[0].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'IAT does not match'):
            build(self.dataset, self.paths)


if __name__ == '__main__':
    unittest.main()
