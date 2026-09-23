import tempfile
import unittest
from pathlib import Path

from pilot.pairing import USTC_LABELS
from pilot.scan_ustc_coverage import select_captures


class CaptureSelectionTest(unittest.TestCase):
    def test_direct_nested_and_chunked_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in USTC_LABELS:
                target = root / 'Benign' / f'{name}.pcap'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'example')
            (root / 'Benign' / 'SMB.pcap').unlink()
            smaller = root / 'Benign' / 'SMB' / 'SMB' / 'SMB-2.pcap'
            smaller.parent.mkdir(parents=True, exist_ok=True)
            smaller.write_bytes(b'x')
            (smaller.parent / 'SMB-1.pcap').write_bytes(b'longer')
            selected = dict(select_captures(root))
            self.assertEqual(len(selected), len(USTC_LABELS))
            self.assertEqual(selected['SMB'], smaller)
            self.assertEqual(selected['Facetime'], root / 'Benign' / 'Facetime.pcap')

    def test_missing_class_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing USTC class'):
                select_captures(directory)
