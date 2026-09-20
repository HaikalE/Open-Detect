import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

from pilot.audit import audit_npz, audit_pcap, audit_paths, image_hash
from data.Preprocessing.utils import read_pcap_list


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def packet(self, reverse=False, timestamp=1.0):
        a, b = ('10.0.0.1', '10.0.0.2') if not reverse else ('10.0.0.2', '10.0.0.1')
        packet = Ether()/IP(src=a, dst=b)/TCP(sport=443 if reverse else 1234,
                                               dport=1234 if reverse else 443)/Raw(b'abc')
        packet.time = timestamp
        return packet

    def capture(self, packets):
        path = self.root/'sample.pcap'
        wrpcap(str(path), packets)
        return path

    def test_matches_original_preprocessing_but_not_pairing_proof(self):
        path = self.capture([self.packet(), self.packet(True, 1.2)])
        image = read_pcap_list(str(path))[0]['data'].astype(np.uint8)
        npz = self.root/'data.npz'
        np.savez(npz, data=image.reshape(1,32,32), target=np.array([2]))
        index = {}
        meta = audit_npz(npz, index)
        result = audit_pcap(path, index)
        self.assertEqual(meta['keys'], ['data', 'target'])
        self.assertEqual(result['status'], 'HASH_CANDIDATE_ONLY')
        self.assertEqual(result['matching_rows'], 1)
        self.assertEqual(result['sequence_length'], 2)
        self.assertEqual(result['first_iat_seconds'], 0)
        self.assertFalse(result['pairing_verified'])

    def test_multiflow_rejected(self):
        other = self.packet()
        other[IP].dst = '10.0.0.3'
        result = audit_pcap(self.capture([self.packet(), other]), {})
        self.assertEqual(result['status'], 'MULTIFLOW_NEEDS_SESSIONIZATION')

    def test_out_of_order_timestamps(self):
        result = audit_pcap(self.capture([self.packet(timestamp=2), self.packet(timestamp=1)]), {})
        self.assertEqual(result['status'], 'INVALID_TIMESTAMPS')

    def test_scan_limit_never_claims_match(self):
        result = audit_pcap(self.capture([self.packet(), self.packet(timestamp=2)]), {}, packet_limit=1)
        self.assertEqual(result['status'], 'SCAN_LIMIT_NO_PAIRING_CLAIM')

    def test_no_match(self):
        self.assertEqual(audit_pcap(self.capture([self.packet()]), {})['status'], 'NO_HASH_MATCH')

    def test_dhcp_excluded(self):
        packet = Ether()/IP()/UDP(sport=67,dport=68)/Raw(b'x')
        self.assertEqual(audit_pcap(self.capture([packet]), {})['status'], 'NO_VALID_PACKETS')

    def test_invalid_dtype_rejected(self):
        with self.assertRaises(ValueError):
            image_hash(np.zeros((32,32), dtype=np.float32))

    def test_corrupt_npz_is_reported(self):
        result = audit_paths([self.root/'missing.npz'])
        self.assertEqual(len(result['errors']), 1)
        self.assertFalse(result['training_ready'])

    def test_duplicate_rows_not_dropped(self):
        path = self.root/'duplicate.npz'
        np.savez(path, data=np.zeros((2,32,32), dtype=np.uint8), target=np.array([0,0]))
        index = {}
        audit_npz(path, index)
        self.assertEqual(next(iter(index.values()))['count'], 2)

    def test_generated_notebook_code_compiles(self):
        path = Path(__file__).resolve().parents[1]/'pilot/01_AUDIT_DATA_CPU.ipynb'
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(line for line in cell['source'] if not line.startswith('%'))
                compile(source, cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()
