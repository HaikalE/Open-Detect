import tempfile
import unittest
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, PcapReader, TCP, Raw, wrpcap

from pilot.recover_anchored_sequences import recover
from pilot.trace_preprocessing import legacy_block


class AnchoredRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, packets, image_indices):
        capture = self.root/'Facetime.pcap'
        wrpcap(str(capture), packets)
        with PcapReader(str(capture)) as reader:
            blocks = [legacy_block(packet) for packet in reader]
        image = np.frombuffer(b''.join(blocks[i] for i in image_indices).ljust(1024,b'\x00'),dtype=np.uint8).reshape(1,32,32)
        np.savez(self.root/'USTC_1c_train.npz', data=image, target=np.array([3]))
        np.savez(self.root/'USTC_1c_test.npz', data=np.empty((0,32,32),dtype=np.uint8),target=np.empty(0,dtype=int))
        return capture

    def packet(self, time, port, reverse=False, payload=b'one'):
        p = Ether()/IP(src='10.0.0.2' if reverse else '10.0.0.1',
                       dst='10.0.0.1' if reverse else '10.0.0.2')/TCP(
            sport=443 if reverse else port,dport=port if reverse else 443)/Raw(payload)
        p.time = time
        return p

    def test_interleaved_full_image_recovers_one_flow_and_iat(self):
        capture = self.prepare([self.packet(1,1234),self.packet(1.1,5678),
                                self.packet(1.2,1234,True,b'two')],[0,2])
        report = recover(self.root,capture,'Facetime',packet_limit=10)
        self.assertEqual(report['outcomes']['full_multi_packet_recovery'],1)
        row = report['recovered'][0]
        self.assertEqual(row['packet_indices'],[0,2])
        self.assertEqual(row['direction'],[0,1])
        self.assertAlmostEqual(row['iat_seconds'][1],0.2)

    def test_ambiguous_first_packet_refused(self):
        first = self.packet(1,1234)
        duplicate = first.copy()
        duplicate.time = 1.2
        capture = self.prepare([first,duplicate],[0])
        report = recover(self.root,capture,'Facetime',packet_limit=10)
        self.assertEqual(report['outcomes']['ambiguous_source_anchor'],1)
        self.assertFalse(report['recovered'])


if __name__ == '__main__':
    unittest.main()
