"""CPU fault-injection tests of the actual adapted training loop on tiny data."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import train as adapted
import train as original

import resume_support as recovery

class SmallDataset(Dataset):
    def __init__(self, augment):
        self.augment = augment

    def __len__(self):
        return 12

    def __getitem__(self, index):
        x = torch.full((1, 2, 2), float(index % 2))
        if self.augment:
            x += 0.01 * (torch.rand_like(x) + np.random.random() + random.random())
        return x, index % 2


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        mu = self.linear(x.flatten(1))
        return mu, torch.zeros_like(mu), {}


class SmallModel(nn.Module):
    latest = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.decoder_version = 2
        self.encoder = SmallEncoder()
        self.decoder = nn.Linear(2, 4)
        self.prototypes = nn.Parameter(torch.randn(2, 2))
        SmallModel.latest = self

    def loss(self, x, y):
        mu, _, _ = self.encoder(x)
        z = mu + torch.randn_like(mu) * 0.01 if self.training else mu
        distances = (z[:, None] - self.prototypes[None]).square().sum(2)
        reconstruction = self.decoder(z).reshape_as(x)
        losses = {
            'rec': (reconstruction - x).square().mean(),
            'kld': distances[torch.arange(len(y)), y].mean(),
            'dis': nn.functional.cross_entropy(-distances, y),
        }
        return z, reconstruction, distances.argmin(1), losses


def tiny_splits(*args, **kwargs):
    return SmallDataset(True), SmallDataset(False), SmallDataset(False)


def args_for(folder, epochs=6, workers=0, gpu=-1):
    return adapted.build_parser().parse_args([
        '--gpu', str(gpu), '--epoch', str(epochs), '--num_workers', str(workers),
        '--resume', '--batch_size', '4', '--save_dir', str(folder), '--dset', 'mal', '--split', '0'])


def run(module, args, interrupt_epoch=None):
    random.seed(9182)
    args.recovery_identity = None
    actual_train = module.train_model
    def interrupt_after_training(model, args, loader, epoch, optimizer):
        actual_train(model, args, loader, epoch, optimizer)
        if epoch == interrupt_epoch:
            raise InterruptedError('simulated runtime killed during uncommitted epoch')
    with patch.object(module, 'OpenDetectNet', SmallModel), \
         patch.object(module, 'get_dataset_splits', tiny_splits), \
         patch.object(module, 'get_splits', return_value=([0, 1], [2], 'mal', 'mal')), \
         patch.object(module, 'dataset_manifest', return_value={'files': {'mock': 'fixed'}}), \
         patch.object(module, 'train_model', interrupt_after_training), \
         contextlib.redirect_stdout(io.StringIO()):
        module.train(args)
    return {name: tensor.detach().clone() for name, tensor in SmallModel.latest.state_dict().items()}


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_restart(self, epochs, interrupt_epoch, workers=0, corrupt_latest=False, gpu=-1):
        with tempfile.TemporaryDirectory(prefix='resume-qa-') as temp:
            root = Path(temp)
            base_args = args_for(root / 'original', epochs, workers, gpu)
            expected = run(original, base_args)
            full_args = args_for(root / 'full', epochs, workers, gpu)
            uninterrupted = run(adapted, full_args)
            resumed_args = args_for(root / 'resumed', epochs, workers, gpu)
            with self.assertRaises(InterruptedError):
                run(adapted, resumed_args, interrupt_epoch)
            folder = recovery.resume_directory(resumed_args)
            records, _ = recovery.valid_records(folder)
            self.assertEqual(records[0][0]['completed_epochs'], interrupt_epoch)
            if corrupt_latest:
                records[0][1].write_bytes(b'fault injection: torn write')
            actual = run(adapted, resumed_args)
            for name in expected:
                torch.testing.assert_close(expected[name], uninterrupted[name], rtol=0, atol=0)
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
            records, _ = recovery.valid_records(folder)
            state = recovery.load_owned_checkpoint(records[0][1])
            full_records, _ = recovery.valid_records(recovery.resume_directory(full_args))
            full_state = recovery.load_owned_checkpoint(full_records[0][1])
            self.assertEqual(state['completed_epochs'], epochs)
            self.assertEqual(state['scheduler'], full_state['scheduler'])
            for k, v in state['optimizer']['state'].items():
                for field, tensor in v.items():
                    torch.testing.assert_close(tensor, full_state['optimizer']['state'][k][field], rtol=0, atol=0)
            for key in ('torch_cpu', 'loader_generator'):
                self.assertTrue(torch.equal(state['rng'][key], full_state['rng'][key]))
            self.assertEqual(state['rng']['python'], full_state['rng']['python'])
            np.testing.assert_equal(state['rng']['numpy'], full_state['rng']['numpy'])
            best_orig = recovery.load_owned_checkpoint(original.checkpoint_path(base_args))
            best_resumed = recovery.load_owned_checkpoint(adapted.checkpoint_path(resumed_args))
            for name, tensor in best_orig['model_state_dict'].items():
                torch.testing.assert_close(tensor, best_resumed['model_state_dict'][name], rtol=0, atol=0)
            # Restart after epoch N was saved but launcher marker not written.
            self.assertEqual(run(adapted, resumed_args).keys(), actual.keys())
            # Wrong hyperparameters must stop instead of silently restarting.
            resumed_args.lr = 0.002
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                run(adapted, resumed_args)

    def test_restart_all_states(self):
        self.check_restart(6, 3)

    def test_latest_corrupt_falls_back(self):
        self.check_restart(6, 3, corrupt_latest=True)

    def test_scheduler_and_prototype_reset_boundary(self):
        self.check_restart(53, 51)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_restart(self):
        self.check_restart(5, 3, gpu=0)

    def test_worker_rng_restart(self):
        self.check_restart(3, 1, workers=2)

    def test_incomplete_commit_keeps_previous_slot(self):
        with tempfile.TemporaryDirectory(prefix='resume-write-qa-') as temp:
            args = args_for(Path(temp), 4)
            with self.assertRaises(InterruptedError):
                run(adapted, args, 2)
            folder = recovery.resume_directory(args)
            real_publish = recovery.publish_json
            def fail_manifest(path, value):
                if str(path).endswith('.pt.json'):
                    raise OSError('simulated disk write failure')
                return real_publish(path, value)
            with patch.object(recovery, 'publish_json', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    run(adapted, args)
            records, _ = recovery.valid_records(folder)
            self.assertEqual(records[0][0]['completed_epochs'], 2)
            run(adapted, args)
            for record, path in recovery.valid_records(folder)[0]:
                path.write_bytes(b'corrupt all slots')
            with self.assertRaisesRegex(RuntimeError, 'No valid'):
                run(adapted, args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
