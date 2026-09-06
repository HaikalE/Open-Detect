from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_grouped as runner
from resume_support import VERSION, publish_json
from provenance import sha256


class RunnerTests(unittest.TestCase):
    def args(self, output):
        return Namespace(scenario='B-1', output=str(output), gpu=-1, num_workers=0,
                         session_hours=3., split_manifest_dir=str(runner.ROOT/'grouped_manifests'))

    def fake_run(self, command, logfile, env):
        output = Path(self.current.output)
        fold = int(command[command.index('--fold') + 1])
        config = json.loads((output/'GROUPED_CONFIG.json').read_text())
        name = f'mal_split_0_fold_{fold}'
        if command[1].endswith('train.py'):
            (output/'save_model'/f'{name}.pt').write_bytes(b'own mock checkpoint')
            folder = output/'resume_state'/name; folder.mkdir(exist_ok=True)
            slot = folder/'last.pt'; slot.write_bytes(b'own mock resume')
            publish_json(folder/'last.pt.json', {'format': VERSION, 'completed_epochs': self.epochs,
                                                'sha256': sha256(slot)})
        else:
            scores = output/'results'/f'{name}.scores.npz'; scores.write_bytes(b'mock scores')
            result = {key: .5 for key in runner.SUMMARY_METRICS}
            result.update(fold=fold, split_seed=2022+fold, protocol=runner.GROUP_PROTOCOL,
                          split_identity=config['splits'][fold], decoder_version=2,
                          training_code_identity=config['code'], data_manifest=config['data'],
                          unknown_data_manifest=config['data'], training_config=config)
            publish_json(output/'results'/f'{name}.json', result)

    def execute(self, args, epochs=100):
        self.current, self.epochs = args, epochs
        with patch.object(runner, 'dataset_manifest', return_value={'files': {'mock': 'same'}}), \
             patch.object(runner, 'run_logged', side_effect=self.fake_run) as calls, \
             redirect_stdout(io.StringIO()):
            result = runner.run(args)
        return result, calls.call_count

    def test_five_seeds_skip_completed_and_reject_changed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/'output')
            result, calls = self.execute(args)
            self.assertEqual([r['split_seed'] for r in result], list(range(2022, 2027)))
            self.assertEqual(calls, 10)
            self.assertEqual(self.execute(args)[1], 0)
            self.assertTrue((Path(args.output)/'summary.json').is_file())
            (Path(args.output)/'save_model/mal_split_0_fold_0.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                self.execute(args)

    def test_partial_epoch_is_not_complete_and_can_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)/'output')
            results, calls = self.execute(args, epochs=3)
            self.assertEqual((len(results), calls), (0, 1))
            status = json.loads((Path(args.output)/'SESSION_STATUS.json').read_text())
            self.assertEqual(status['status'], 'paused')
            self.assertFalse((Path(args.output)/'summary.json').exists())
            self.assertEqual(len(self.execute(args)[0]), 5)

    def test_refuse_old_output_and_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'old'; output.mkdir(); (output/'old.pt').write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'empty output'):
                self.execute(self.args(output))
            args = self.args(Path(tmp)/'new')
            self.execute(args, epochs=3)
            args.num_workers = 1
            with self.assertRaisesRegex(ValueError, 'different code/data/config'):
                self.execute(args)


if __name__ == '__main__':
    unittest.main()
