import unittest

import numpy as np
import torch

from model import OpenDetectNet
from model_fusion import OpenDetectFusionNet
from networks.temporal import TemporalBiGRU
from pilot.smoke_mini_fusion import normalize_sequence


class TemporalFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_padding_does_not_change_bigru_output(self):
        torch.manual_seed(1)
        net = TemporalBiGRU(3).eval()
        sequence = torch.randn(2, 8, 3)
        lengths = torch.tensor([2, 5])
        changed = sequence.clone()
        changed[0, 2:] = 1000
        changed[1, 5:] = -1000
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(sequence, lengths), net(changed, lengths)))

    def test_fusion_shapes_gradients_and_original_model(self):
        torch.manual_seed(2)
        image = torch.rand(2, 1, 32, 32)
        sequence = torch.rand(2, 8, 3)
        length = torch.tensor([2, 4])
        labels = torch.tensor([0, 1])
        fusion = OpenDetectFusionNet(channel=1, n_classes=2, sequence_features=3)
        fusion.train()
        _, recon, pred, terms = fusion.loss(image, sequence, length, labels)
        self.assertEqual(recon.shape, image.shape)
        self.assertEqual(pred.shape, labels.shape)
        sum(terms.values()).backward()
        self.assertIsNotNone(fusion.encoder.temporal.gru.weight_ih_l0.grad)
        self.assertIsNotNone(fusion.encoder.image_encoder.conv1.weight.grad)
        self.assertIsNotNone(fusion.prototypes.grad)
        baseline = OpenDetectNet(channel=1, n_classes=2)
        baseline.eval()
        with torch.no_grad():
            self.assertEqual(baseline(image)[-1].shape, image.shape)

    def test_two_feature_ablation_ignores_iat_column(self):
        torch.manual_seed(3)
        model = TemporalBiGRU(2).eval()
        sequence = torch.rand(2, 8, 3)
        changed = sequence.clone()
        changed[:, :, 2] = 999
        lengths = torch.tensor([3, 8])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(sequence, lengths), model(changed, lengths)))

    def test_normalizer_fits_only_train_and_zeros_padding(self):
        values = np.array([[[1, 0, 0], [3, 1, 0.5], [0, 0, 0]],
                           [[10000, 0, 1000], [0, 0, 0], [0, 0, 0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        split_id = np.array([0, 2], dtype=np.uint8)
        transformed, stats = normalize_sequence(values, mask, split_id)
        self.assertAlmostEqual(stats['log1p_payload_bytes']['mean'],
                               float(np.log1p([1, 3]).mean()))
        self.assertTrue(np.all(transformed[mask == 0] == 0))


if __name__ == '__main__':
    unittest.main()
