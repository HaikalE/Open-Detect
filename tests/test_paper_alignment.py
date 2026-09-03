import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import stratified_split_indices
from data.Preprocessing.utils import (
    HEADER_BYTES_PER_PACKET,
    IMAGE_BYTES,
    PACKETS_PER_FLOW,
    PAYLOAD_BYTES_PER_PACKET,
    read_pcap_list,
)
from model import OpenDetectNet
from test import open_world_metrics, threshold_from_known_validation
from utils import reset_prototype


class LossOnlyOpenDetect(OpenDetectNet):
    def __init__(self, kl_divergences, reconstruction):
        nn.Module.__init__(self)
        self.n_classes = kl_divergences.shape[1]
        self.temp_inter = 1.0
        self.kl_divergences = kl_divergences
        self.reconstruction = reconstruction

    def forward(self, x):
        batch_size = len(x)
        latent = torch.zeros(batch_size, 1)
        distance = torch.zeros_like(self.kl_divergences)
        return latent, distance, self.kl_divergences, self.reconstruction


class IdentityEncoder(nn.Module):
    def forward(self, x):
        mean = x.view(len(x), -1)
        return mean, torch.zeros_like(mean), {}


class PrototypeOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityEncoder()
        self.prototypes = nn.Parameter(torch.zeros(2, 2))


class PaperAlignmentTests(unittest.TestCase):
    @patch('data.Preprocessing.utils.raw_packet_to_string')
    @patch('data.Preprocessing.utils.scapy.rdpcap')
    def test_preprocessing_uses_eight_packets_and_128_bytes_each(
        self,
        mock_rdpcap,
        mock_packet_to_string,
    ):
        mock_rdpcap.return_value = list(range(10))
        mock_packet_to_string.return_value = (
            'aa' * HEADER_BYTES_PER_PACKET,
            'bb' * PAYLOAD_BYTES_PER_PACKET,
        )

        result = read_pcap_list('dummy.pcap', if_augment=False)

        self.assertEqual(PACKETS_PER_FLOW, 8)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['data'].shape, (IMAGE_BYTES,))
        self.assertEqual(mock_packet_to_string.call_count, PACKETS_PER_FLOW)

    @patch('data.Preprocessing.utils.raw_packet_to_string')
    @patch('data.Preprocessing.utils.scapy.rdpcap')
    def test_preprocessing_augmentation_returns_valid_sliding_windows(
        self,
        mock_rdpcap,
        mock_packet_to_string,
    ):
        mock_rdpcap.return_value = list(range(10))
        mock_packet_to_string.return_value = (
            'aa' * HEADER_BYTES_PER_PACKET,
            'bb' * PAYLOAD_BYTES_PER_PACKET,
        )

        result = read_pcap_list('dummy.pcap', if_augment=True)

        self.assertEqual(len(result), 3)
        self.assertTrue(all(item['data'].shape == (IMAGE_BYTES,) for item in result))

    def test_split_is_stratified_disjoint_and_8_1_1(self):
        targets = np.repeat(np.arange(3), 100)
        train_indices, validation_indices, test_indices = stratified_split_indices(targets, seed=7)

        self.assertEqual((len(train_indices), len(validation_indices), len(test_indices)), (240, 30, 30))
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertFalse(set(train_indices) & set(test_indices))
        self.assertFalse(set(validation_indices) & set(test_indices))
        self.assertEqual(len(set(train_indices) | set(validation_indices) | set(test_indices)), 300)
        for class_id in range(3):
            self.assertEqual(np.sum(targets[train_indices] == class_id), 80)
            self.assertEqual(np.sum(targets[validation_indices] == class_id), 10)
            self.assertEqual(np.sum(targets[test_indices] == class_id), 10)

    def test_threshold_accepts_95_percent_of_known_validation(self):
        validation_scores = np.arange(100, dtype=np.float64)
        threshold = threshold_from_known_validation(validation_scores, known_acceptance=0.95)
        self.assertEqual(np.mean(validation_scores < threshold), 0.95)

    def test_open_decision_uses_strict_paper_rule(self):
        metrics = open_world_metrics(
            known_scores=np.array([0.1, 0.2]),
            unknown_scores=np.array([0.3, 0.4]),
            threshold=0.3,
        )
        self.assertEqual(metrics['known_test_acceptance'], 1.0)
        self.assertEqual(metrics['unknown_test_rejection'], 1.0)

    def test_kl_matches_gaussian_prototype_equation(self):
        model = object.__new__(OpenDetectNet)
        nn.Module.__init__(model)
        mean = torch.tensor([[1.0, 2.0]])
        logvar = torch.log(torch.tensor([[2.0, 0.5]]))
        prototypes = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        actual = model.kl_div_to_prototypes(mean, logvar, prototypes)
        expected = 0.5 * (
            torch.sum((mean[:, None, :] - prototypes[None, :, :]) ** 2, dim=2)
            + torch.sum(logvar.exp() - logvar - 1, dim=1, keepdim=True)
        )
        torch.testing.assert_close(actual, expected)

    def test_loss_has_only_the_two_paper_constraints(self):
        x = torch.zeros(2, 1, 2, 2)
        reconstruction = torch.ones_like(x)
        kl_divergences = torch.tensor([[0.2, 1.2], [1.5, 0.5]])
        labels = torch.tensor([0, 1])
        model = LossOnlyOpenDetect(kl_divergences, reconstruction)

        _, _, predictions, losses = model.loss(x, labels)

        self.assertEqual(set(losses), {'rec', 'kld', 'dis'})
        torch.testing.assert_close(predictions, torch.tensor([0, 1]))
        torch.testing.assert_close(losses['rec'], F.mse_loss(reconstruction, x))
        torch.testing.assert_close(losses['kld'], torch.tensor(0.35))
        torch.testing.assert_close(losses['dis'], F.cross_entropy(-kl_divergences, labels))

    def test_prototype_reset_updates_parameter_in_place(self):
        features = torch.tensor([
            [[[1.0, 3.0]]],
            [[[3.0, 5.0]]],
            [[[10.0, 14.0]]],
            [[[14.0, 18.0]]],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
        model = PrototypeOnlyModel()
        original_parameter = model.prototypes

        reset_prototype(model, loader)

        self.assertIs(model.prototypes, original_parameter)
        torch.testing.assert_close(
            model.prototypes,
            torch.tensor([[2.0, 4.0], [12.0, 16.0]]),
        )


if __name__ == '__main__':
    unittest.main()
