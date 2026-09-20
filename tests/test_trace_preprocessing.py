import unittest
from scapy.all import Ether, IP, IPv6, TCP, Raw
from pilot.trace_preprocessing import legacy_block


class LegacyTraceTests(unittest.TestCase):
    def test_original_header_payload_layout(self):
        packet = Ether()/IP(src='10.0.0.1',dst='10.0.0.2')/TCP()/Raw(b'hello')
        result = legacy_block(packet)
        self.assertEqual(len(result),128)
        self.assertEqual(result[80:85],b'hello')
        self.assertEqual(result[85:],bytes(43))
        self.assertEqual(result[12:20],bytes(8))

    def test_does_not_mutate_source_flow_endpoints(self):
        packet = Ether()/IP(src='10.0.0.1',dst='10.0.0.2')/TCP()/Raw(b'x')
        original = bytes(packet)
        legacy_block(packet)
        self.assertEqual(bytes(packet),original)

    def test_upstream_ipv4_only_failure_padding(self):
        self.assertEqual(legacy_block(Ether()/IPv6()/TCP()),bytes(128))


if __name__ == '__main__':
    unittest.main()
