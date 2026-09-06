"""Synthetic fixtures exercise reporting only; never represented as thesis results."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import ToTensor

import report_grouped as report
import test as evaluator
from data.dataset import TrafficArrayDataset
from model import OpenDetectNet


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_identity = {'git_commit': report.TRAIN_COMMIT, 'source_sha256': {'test.py': 'fixture'}}
        self.identity = {'protocol': 'fixture-grouped', 'seed': 2022, 'scenario': 'A-1'}
        self.grouped = types.SimpleNamespace(
            SCENARIOS={'A-1': ('USTC', 0)}, GROUP_PROTOCOL='fixture-grouped',
            bundle_record=lambda *args: (None, None, None, self.identity))

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        for folder in ('state', 'results', 'save_model'):
            (self.root/folder).mkdir(exist_ok=True)
        config = {'scenario': 'A-1', 'dataset': 'USTC', 'split': 0, 'epoch': 100,
                  'folds': 5, 'base_seed': 2022, 'known_acceptance': .95,
                  'code': self.source_identity, 'splits': [self.identity]*5, 'protocol': 'fixture-grouped'}
        report.save_json(self.root/'GROUPED_CONFIG.json', config)
        name = 'USTC_split_0_fold_0'
        scores = {'validation_scores': np.linspace(.1, .9, 40),
                  'known_scores': np.linspace(.1, 1, 20), 'unknown_scores': np.linspace(.5, 2, 12),
                  'known_labels': np.arange(20) % 2, 'known_predictions': np.arange(20) % 2}
        scores['threshold'] = evaluator.threshold_from_known_validation(scores['validation_scores'])
        self.checkpoint = self.root/'save_model'/f'{name}.pt'
        torch.save({'fixture': True}, self.checkpoint)
        self.score_path = self.root/'results'/f'{name}.scores.npz'
        np.savez_compressed(self.score_path, **scores)
        metrics = evaluator.open_world_metrics(scores['known_scores'], scores['unknown_scores'], scores['threshold'])
        metrics.update(evaluator.closed_world_metrics(scores['known_labels'], scores['known_predictions']))
        metrics.update(fold=0, split_seed=2022, dataset='USTC', scenario_split=0,
                       split_identity=self.identity, protocol='fixture-grouped', threshold=scores['threshold'],
                       checkpoint_sha256=report.digest(self.checkpoint), training_code_identity=self.source_identity,
                       evaluation_code_identity=self.source_identity, known_test_samples=20,
                       unknown_test_samples=12, known_validation_samples=40)
        self.result_path = self.root/'results'/f'{name}.json'
        report.save_json(self.result_path, metrics)
        self.marker_path = self.root/'state'/f'{name}.completed.json'
        self.marker = {'config': config, 'checkpoint_sha256': report.digest(self.checkpoint),
                       'result_sha256': report.digest(self.result_path), 'scores_sha256': report.digest(self.score_path)}
        report.save_json(self.marker_path, self.marker)

    def verify(self):
        return report.verify_run(self.root, 'A-1', 0, evaluator, self.grouped, self.source_identity, self.root)

    def test_pending_does_not_require_files(self):
        self.assertIsNone(self.verify())

    def test_verified_read_only(self):
        self.fixture()
        before = {str(p): report.digest(p) for p in self.root.rglob('*') if p.is_file()}
        run = self.verify()
        self.assertEqual(run['split_seed'], 2022)
        self.assertEqual(before, {str(p): report.digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_corrupt_checkpoint_rejected(self):
        self.fixture()
        self.checkpoint.write_bytes(b'corrupt fixture')
        with self.assertRaisesRegex(ValueError, 'hash mismatch: checkpoint'):
            self.verify()

    def test_wrong_run_rejected_even_with_matching_hash(self):
        self.fixture()
        m = report.read_json(self.result_path)
        m['split_seed'] = 2024
        report.save_json(self.result_path, m)
        self.marker['result_sha256'] = report.digest(self.result_path)
        report.save_json(self.marker_path, self.marker)
        with self.assertRaisesRegex(ValueError, 'another run'):
            self.verify()

    def test_wrong_metric_rejected_even_with_matching_hash(self):
        self.fixture()
        m = report.read_json(self.result_path)
        m['open_f1'] = 1.
        report.save_json(self.result_path, m)
        self.marker['result_sha256'] = report.digest(self.result_path)
        report.save_json(self.marker_path, self.marker)
        with self.assertRaisesRegex(ValueError, 'Saved metric'):
            self.verify()

    def test_test_tuned_threshold_rejected(self):
        self.fixture()
        with np.load(self.score_path) as z:
            scores = {k: z[k].copy() for k in z.files}
        scores['threshold'] = 1.5
        np.savez_compressed(self.score_path, **scores)
        self.marker['scores_sha256'] = report.digest(self.score_path)
        report.save_json(self.marker_path, self.marker)
        with self.assertRaisesRegex(ValueError, 'known-validation-only'):
            self.verify()

    def test_partial_summary_has_no_final_mean(self):
        self.fixture()
        run = self.verify()
        self.assertIsNone(report.aggregate([run])[0]['open_accuracy_mean_percent'])
        complete = []
        for i in range(5):
            row = copy.deepcopy(run)
            row['split_seed'] = 2022+i
            row['metrics']['open_f1'] = .5 + i*.1
            complete.append(row)
        summary = report.aggregate(complete)[0]
        self.assertAlmostEqual(summary['open_f1_mean_percent'], 70)
        self.assertAlmostEqual(summary['open_f1_sd_percent'], np.std([50,60,70,80,90], ddof=1))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.aggregate([run, run])

    def test_plot_report_smoke(self):
        self.fixture()
        run = self.verify()
        out = self.root/'demo_SYNTHETIC_NOT_RESULTS'
        (out/'A-1').mkdir(parents=True)
        report.score_plots(run, out/'A-1')
        report.render_report(out, report.aggregate([run]), [{'scenario': 'A-1', 'status': 'synthetic test'}])
        self.assertGreater((out/'A-1/scores_roc_confusion.png').stat().st_size, 1000)
        self.assertIn('tanpa ablasi', (out/'REPORT.html').read_text(encoding='utf-8'))

    def test_real_model_diagnostics_without_training(self):
        torch.set_num_threads(2)
        torch.manual_seed(2022)
        model = OpenDetectNet('resnet18', 1, 8, 2, init=False).eval()
        rng = np.random.default_rng(2022)
        data = rng.integers(0, 256, (12,32,32), dtype=np.uint8)
        known = TrafficArrayDataset(data[:8], np.arange(8)%2, ToTensor(), [0,1], 'reindex')
        unknown = TrafficArrayDataset(data[8:], np.full(4, 2), ToTensor(), [2], 'open')
        checkpoint = self.root/'random_UNTRAINED_FIXTURE.pt'
        torch.save({'model_state_dict': model.state_dict(), 'model_config': {
            'arch':'resnet18', 'channel':1, 'latent_dim':8, 'n_classes':2, 'temp_inter':1., 'decoder_version':2},
            'fold':0, 'split_seed':2022, 'dataset':'USTC', 'scenario_split':0,
            'protocol': 'fixture-grouped', 'split_identity':self.identity,
            'code_identity':self.source_identity}, checkpoint)
        mock_grouped = types.SimpleNamespace(get_grouped_splits=lambda *args: (None,None,known,unknown,self.identity),
                                            verify_checkpoint_split=lambda c,i: None)
        run = {'scenario':'A-1','fold':0,'split_seed':2022, 'metrics':{'dataset':'USTC','scenario_split':0,'threshold':1.},
               'paths':{'checkpoint':checkpoint}, 'marker':{'checkpoint_sha256': report.digest(checkpoint)},
               'config':{'code': self.source_identity}}
        out = self.root/'random_model_diagnostics'
        out.mkdir()
        before = report.digest(checkpoint)
        report.model_plots(run, out, evaluator, mock_grouped, self.root, torch.device('cpu'))
        self.assertEqual(before, report.digest(checkpoint))
        self.assertTrue((out/'latent_tsne.png').exists())
        timing = report.read_json(out/'inference_timing.json')
        self.assertIsNone(timing['feature_extraction_ms'])
        self.assertGreater(timing['mean_ms'], 0)
        with np.load(out/'pixel_gradient.npz') as z:
            self.assertTrue(np.isfinite(z['absolute_gradient']).all())
            self.assertGreater(z['absolute_gradient'].max(), 0)


if __name__ == '__main__':
    unittest.main()
