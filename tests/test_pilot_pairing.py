import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, TCP, Raw, wrpcap
from data.Preprocessing.utils import read_pcap_list
from pilot.pairing import extract_candidates, annotate_matches, run_pairing


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def packet(self, t, reverse=False, flags='PA', seq=1, port=1234):
        p = Ether()/IP(src='10.0.0.2' if reverse else '10.0.0.1', dst='10.0.0.1' if reverse else '10.0.0.2')/TCP(
            sport=443 if reverse else port, dport=port if reverse else 443, flags=flags, seq=seq)/Raw(b'abc')
        p.time = t
        return p

    def extract(self, packets, **kwargs):
        path = self.root/'Facetime.pcap'
        wrpcap(str(path), packets)
        return extract_candidates(path, 'Facetime', **kwargs)

    def test_pairing_matches_original_image_and_time(self):
        rows, arrays, meta = self.extract([self.packet(1), self.packet(1.25, True)])
        reference = read_pcap_list(str(self.root/'Facetime.pcap'))[0]['data']
        np.testing.assert_array_equal(arrays['images'][0].flatten(), reference)
        np.testing.assert_allclose(arrays['sequences'][0,:2], [[3,0,0],[3,1,.25]])
        self.assertEqual(arrays['mask'][0].sum(), 2)
        self.assertEqual(rows[0]['packet_indices'], [0,1])
        self.assertTrue(meta['complete'])

    def test_interleaved_biflows(self):
        rows, arrays, _ = self.extract([self.packet(1), self.packet(1.1,port=5555), self.packet(1.2,True)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['packet_indices'], [0,2])
        self.assertAlmostEqual(arrays['sequences'][0,1,2], .2)

    def test_idle_timeout(self):
        rows, _, _ = self.extract([self.packet(1), self.packet(62)])
        self.assertEqual(len(rows), 2)

    def test_syn_retransmission_and_new_syn(self):
        rows, _, _ = self.extract([self.packet(1,flags='S'), self.packet(2,flags='S'), self.packet(3,flags='S',seq=42)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['length'], 2)

    def test_rst_boundary(self):
        rows, _, _ = self.extract([self.packet(1), self.packet(2,flags='R'), self.packet(3)])
        self.assertEqual(len(rows), 2)

    def test_negative_time_fails_capture(self):
        with self.assertRaises(ValueError):
            self.extract([self.packet(2), self.packet(1)])

    def test_microsecond_timestamp_quantization_is_explicit(self):
        path = self.root/'Facetime.pcap'
        wrpcap(str(path), [self.packet(2), self.packet(2-0.000001)])
        rows, arrays, meta = extract_candidates(path, 'Facetime', timestamp_tolerance_us=2)
        self.assertEqual(meta['timestamp_adjustments'], 1)
        self.assertEqual(rows[0]['length'], 2)
        self.assertEqual(arrays['sequences'][0,1,2], 0)
        self.assertIn('tol2us', rows[0]['policy'])

    def test_bounded_scan_marked_partial(self):
        rows, _, meta = self.extract([self.packet(1), self.packet(2)], packet_limit=1)
        self.assertFalse(meta['complete'])
        self.assertFalse(rows[0]['capture_complete'])

    def test_first_eight_and_determinism(self):
        packets = [self.packet(t) for t in range(12)]
        a, arrays, _ = self.extract(packets)
        b, _, _ = self.extract(packets)
        self.assertEqual(a[0]['length'], 8)
        self.assertEqual(a[0]['sample_id'], b[0]['sample_id'])
        self.assertEqual(a[0]['session_observed_packets'], 12)

    def test_conflicting_and_duplicate_matches(self):
        rows, _, _ = self.extract([self.packet(1)])
        key = rows[0]['image_sha256']
        self.assertEqual(annotate_matches(rows, {key:[{'dataset':'USTC','label':99}]}), {'LABEL_CONFLICT':1})
        match = {'dataset':'USTC','label':3}
        self.assertEqual(annotate_matches(rows, {key:[match,match]}), {'AMBIGUOUS_HASH':1})
        self.assertEqual(annotate_matches(rows, {key:[match]}), {'UNIQUE_IN_PILOT_ONLY':1})
        self.assertFalse(rows[0]['pairing_verified'])

    def test_artifacts_align_and_no_training_flag(self):
        rows, arrays, _ = self.extract([self.packet(1)])
        npz = self.root/'USTC_1c_test.npz'
        np.savez(npz,data=arrays['images'],target=np.array([3]))
        out = self.root/'out'
        report = run_pairing([npz], [(self.root/'Facetime.pcap','Facetime')], out)
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['temporal_coverage']['Facetime']['matched_multi_packet_candidates'], 0)
        self.assertEqual(report['match_counts'], {'UNIQUE_IN_PILOT_ONLY':1})
        manifest = json.loads((out/'candidate_manifest.jsonl').read_text())
        self.assertEqual(manifest['array_row'], 0)
        self.assertIsNone(manifest['split'])
        with self.assertRaises(FileExistsError):
            run_pairing([npz], [], out)

    def test_notebook_syntax(self):
        nb = json.loads((Path(__file__).resolve().parents[1]/'pilot/02_PAIRING_PILOT_CPU.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(l for l in cell['source'] if not l.startswith('%')),cell['id'],'exec')


if __name__ == '__main__':
    unittest.main()
