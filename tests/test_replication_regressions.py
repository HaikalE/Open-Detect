import unittest
import tempfile
from pathlib import Path
import warnings
import io
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from scapy.all import Ether, IP, IPv6, TCP, UDP, Raw, ARP, DNS, DNSQR

from networks.resnet import ResNet18Dec
from test import threshold_from_known_validation, open_world_metrics, load_model
from data.Preprocessing.utils import raw_packet_to_string, read_pcap_list
from model import OpenDetectNet
from provenance import dataset_manifest
import train
import test as evaluation
from run_5fold import validate_aggregate


class ReplicationRegressions(unittest.TestCase):
    def test_aggregation_rejects_mixed_implementations_and_inputs(self):
        record = {'decoder_version': 2, 'training_code_identity': {'git_commit': 'same'},
                  'data_manifest': {'files': 'same'}, 'unknown_data_manifest': {'files': 'same'},
                  'training_config': {'lr': .001, 'fold': 0}}
        other = deepcopy(record)
        other['training_config']['fold'] = 1
        validate_aggregate([record, other])
        for field, value in [('decoder_version', 1), ('training_code_identity', {}),
                             ('data_manifest', {}), ('unknown_data_manifest', {}),
                             ('training_config', {'lr': .1})]:
            changed = deepcopy(record)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_aggregate([record, changed])

    def test_float32_threshold_reaches_requested_acceptance(self):
        for dtype in [np.float32, np.float64]:
            scores = np.arange(3258, dtype=dtype)
            threshold = threshold_from_known_validation(scores)
            result = open_world_metrics(scores, np.array([10000], dtype=dtype), threshold)
            self.assertGreaterEqual(result['known_test_acceptance'], .95)

    def test_last_decoder_block_receives_reconstruction_gradient(self):
        torch.manual_seed(7)
        decoder = ResNet18Dec(nc=1, z_dim=4)
        z = torch.randn(2, 4)
        lateral = {f'x_l{i}': torch.randn(2, channels, side, side)
                   for i, channels, side in [(1, 64, 32), (2, 128, 16),
                                              (3, 256, 8), (4, 512, 4)]}
        output = decoder(z, lateral)
        (output - .3).square().mean().backward()
        grad = decoder.layer1[0].conv1.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_payload_ablation_cannot_leak_into_header(self):
        packet = Ether()/IP(src='1.2.3.4', dst='5.6.7.8')/TCP()/Raw(b'PAYLOAD_SECRET' * 8)
        normal_header, payload = raw_packet_to_string(packet.copy(), keep_payload=True)
        ablated_header, ablated_payload = raw_packet_to_string(packet.copy(), keep_payload=False)
        self.assertEqual(normal_header, ablated_header)
        self.assertNotEqual(payload, '00' * 48)
        self.assertEqual(ablated_payload, '00' * 48)

    def test_invalid_scores_are_rejected(self):
        for values in [[], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                threshold_from_known_validation(values)

    def test_threshold_ties_and_full_acceptance(self):
        for values in [np.ones(100, dtype=np.float32), np.arange(20, dtype=np.float32)]:
            for acceptance in [.95, 1.0]:
                threshold = threshold_from_known_validation(values, acceptance)
                metrics = open_world_metrics(values, np.array([100.]), threshold)
                self.assertGreaterEqual(metrics['known_test_acceptance'], acceptance)

    def test_confusion_counts_and_f1_are_explicit(self):
        result = open_world_metrics([0, 0, 2], [0, 2], 1.)
        self.assertEqual([result[k] for k in ['unknown_true_positive', 'known_false_positive',
                                            'unknown_false_negative', 'known_true_negative']], [1, 1, 1, 2])
        self.assertEqual(result['open_f1'], .5)
        self.assertAlmostEqual(result['binary_macro_f1'], (.5 + 2/3)/2)

    def test_packet_is_not_mutated(self):
        packet = Ether()/IP(src='1.2.3.4', dst='5.6.7.8')/TCP()/Raw(b'example')
        before = bytes(packet)
        raw_packet_to_string(packet)
        self.assertEqual(bytes(packet), before)

    def test_decoded_application_layer_and_ipv6(self):
        application = DNS(qd=DNSQR(qname='example.org'))
        for network in [IP(), IPv6()]:
            packet = network/UDP(sport=1000, dport=53)/application
            header, payload = raw_packet_to_string(packet)
            self.assertTrue(payload.startswith(bytes(application).hex()[:96]))
            self.assertEqual(len(header), 160)
            self.assertEqual(len(payload), 96)

    @patch('data.Preprocessing.utils.scapy.rdpcap')
    def test_excluded_packets_do_not_displace_valid_packets(self, rdpcap):
        valid = IP()/TCP()/Raw(b'valid')
        rdpcap.return_value = [Ether()/ARP(), IP()/UDP(sport=68, dport=67)] + [valid] * 8
        result = read_pcap_list('unused')
        header, payload = raw_packet_to_string(valid)
        expected = np.frombuffer(bytes.fromhex((header + payload) * 8), dtype=np.uint8)
        np.testing.assert_array_equal(result[0]['data'], expected)
        rdpcap.return_value = [Ether()/ARP()]
        self.assertEqual(read_pcap_list('unused'), [])

    def test_legacy_checkpoint_preserves_decoder_path(self):
        model = OpenDetectNet(channel=1, latent_dim=4, n_classes=2, decoder_version=1).eval()
        x = torch.rand(2, 1, 32, 32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pt'
            torch.save({'model_state_dict': model.state_dict(), 'model_config': {
                'arch': 'resnet18', 'channel': 1, 'latent_dim': 4,
                'n_classes': 2, 'temp_inter': 1.}}, path)
            loaded, _ = load_model(path, torch.device('cpu'))
            loaded.eval()
            self.assertEqual(loaded.decoder_version, 1)
            with torch.no_grad():
                torch.testing.assert_close(model(x)[3], loaded(x)[3], rtol=0, atol=0)

    def test_existing_training_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = train.build_parser().parse_args(['--save_dir', tmp])
            path = Path(train.checkpoint_path(args))
            path.write_bytes(b'existing result')
            with self.assertRaises(FileExistsError):
                train.train(args)
            self.assertEqual(path.read_bytes(), b'existing result')

    def test_actual_training_evaluation_and_data_identity(self):
        # Real ResNet, actual stratified dataset path and checkpoint loader on tiny data.
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data_root = root / 'data'; data_root.mkdir()
            rng = np.random.default_rng(7)
            labels = np.repeat(np.arange(20), 10)
            images = rng.integers(0, 256, (200, 32, 32), dtype=np.uint8)
            for suffix, part in [('train', slice(0, 100)), ('test', slice(100, 200))]:
                np.savez(data_root / ('USTC_1c_' + suffix + '.npz'), data=images[part], target=labels[part])
            manifest_for = lambda dataset: dataset_manifest(dataset, data_root)
            args = train.build_parser().parse_args(['--dset', 'USTC', '--save_dir', str(root / 'save'),
                '--gpu', '-1', '--num_workers', '0', '--epoch', '1', '--batch_size', '64', '--h', '4'])
            # dataset.py joins its module data_dir with 'dataset'.
            data_root.rename(root / 'dataset'); data_root = root / 'dataset'
            with patch('data.dataset.data_dir', str(root)), \
                 patch.object(train, 'dataset_manifest', side_effect=manifest_for), \
                 patch.object(evaluation, 'dataset_manifest', side_effect=manifest_for), \
                 redirect_stdout(io.StringIO()), \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                checkpoint_path = train.train(args)
                ev = evaluation.build_parser().parse_args(['--dset', 'USTC', '--save_dir', str(root / 'save'),
                    '--gpu', '-1', '--num_workers', '0', '--scores_out', str(root / 'scores.npz')])
                metrics = evaluation.evaluate(ev)
                self.assertEqual(metrics['decoder_version'], 2)
                self.assertGreaterEqual(metrics['validation_known_acceptance'], .95)
                self.assertTrue((root / 'scores.npz').is_file())
                ev.fold = 1; ev.model_path = checkpoint_path
                with self.assertRaisesRegex(ValueError, 'fold mismatch'):
                    evaluation.evaluate(ev)
                ev.fold = 0
                # Different bytes, same shape and labels: must refuse mismatched evaluation data.
                np.savez(data_root / 'USTC_1c_train.npz', data=np.zeros_like(images[:100]), target=labels[:100])
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    evaluation.evaluate(ev)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
