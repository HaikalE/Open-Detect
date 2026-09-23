import json
import tempfile
import unittest
from pathlib import Path

from pilot.summarize_anchor_coverage import summarize


def _report(path, name, rows):
    path.write_text(json.dumps({
        'schema': 'exact-packet-block-recovery-v1', 'class': name,
        'capture_sha256': name, 'packets_scanned': 10, 'packet_limit': 10,
        'outcomes': {'full_exact_image_recovery': len(rows),
                     'full_multi_packet_recovery': len(rows)},
        'recovered': rows,
    }))


def _row(file, row, group, packets):
    return {'file': file, 'row': row, 'flow_group_sha256': group,
            'capture_sha256': 'A', 'length': 2, 'iat_seconds': [0.0, 0.1],
            'packet_indices': packets}


class AnchorCoverageSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'a.json'

    def test_detects_flow_and_packet_reuse(self):
        _report(self.path, 'A', [_row('USTC_1c_train.npz', 0, 'flow', [1, 2]),
                                 _row('USTC_1c_test.npz', 1, 'flow', [2, 3])])
        summary = summarize([self.path])
        self.assertEqual(summary['total_multi_positive_iat'], 2)
        self.assertEqual(summary['cross_npz_partition_flow_groups'], 1)
        self.assertEqual(summary['source_packets_reused_by_multiple_images'], 1)

    def test_rejects_nonpositive_iat(self):
        row = _row('USTC_1c_train.npz', 0, 'flow', [1, 2])
        row['iat_seconds'] = [0.0, 0.0]
        _report(self.path, 'A', [row])
        with self.assertRaisesRegex(ValueError, 'Nonpositive'):
            summarize([self.path])


if __name__ == '__main__':
    unittest.main()
