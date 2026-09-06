"""Fresh grouped experiment, resumable at epoch boundaries, isolated output ownership."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from data.grouped import SCENARIOS, GROUP_PROTOCOL, bundle_record
from provenance import code_identity, dataset_manifest, sha256
from resume_support import publish_json, valid_records
from run_5fold import SUMMARY_METRICS, validate_aggregate

ROOT = Path(__file__).resolve().parent


def run_logged(command, logfile, env):
    print('COMMAND:', subprocess.list2cmdline(command), flush=True)
    with logfile.open('a', encoding='utf-8') as log:
        log.write('\nSTART ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def verify_completed(marker, config, checkpoint, result, scores, identity):
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record['config'] != config:
        raise ValueError('Completed run config mismatch')
    for key, path in [('checkpoint', checkpoint), ('result', result), ('scores', scores)]:
        if sha256(path) != record[key + '_sha256']:
            raise ValueError('Completed artifact changed: ' + str(path))
    metrics = json.loads(result.read_text(encoding='utf-8'))
    if metrics['split_identity'] != identity or metrics['protocol'] != GROUP_PROTOCOL:
        raise ValueError('Completed result split mismatch')
    return metrics


def write_summary(output, results, config):
    status = {'scenario': config['scenario'], 'completed_folds': len(results), 'total_folds': 5,
              'status': 'complete' if len(results) == 5 else 'paused', 'protocol': GROUP_PROTOCOL}
    publish_json(output / 'SESSION_STATUS.json', status)
    if results:
        keys = ['fold', 'split_seed', *SUMMARY_METRICS]
        temporary = output / 'per_run.csv.writing'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in results)
        os.replace(temporary, output / 'per_run.csv')
    if len(results) == 5:
        validate_aggregate(results)
        if [r['split_seed'] for r in results] != list(range(2022, 2027)):
            raise ValueError('Expected all five seeds, without selection')
        summary = {k: {'mean': float(np.mean([r[k] for r in results])),
                       'std': float(np.std([r[k] for r in results], ddof=1))} for k in SUMMARY_METRICS}
        publish_json(output / 'summary.json', {'config': config, 'per_fold': results,
                                               'summary': summary, 'protocol': GROUP_PROTOCOL})
    print(json.dumps(status, indent=2), flush=True)


def run(args):
    dataset, split = SCENARIOS[args.scenario]
    manifests = Path(args.split_manifest_dir).resolve()
    identities = [bundle_record(manifests, dataset, split, 2022 + f)[3] for f in range(5)]
    config = {'scenario': args.scenario, 'dataset': dataset, 'split': split,
              'protocol': GROUP_PROTOCOL, 'folds': 5, 'base_seed': 2022,
              'epoch': 100, 'batch_size': 128, 'eval_batch_size': 256,
              'lr': .001, 'h': 128, 'c': 1, 'lamda': .005, 'temp_inter': 1.,
              'known_acceptance': .95, 'num_workers': args.num_workers,
              'code': code_identity(), 'data': dataset_manifest(dataset), 'splits': identities}
    output = Path(args.output).resolve()
    config_path = output / 'GROUPED_CONFIG.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Output belongs to different code/data/config; files preserved')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Fresh grouped run requires empty output directory, not old experiment output')
        output.mkdir(parents=True, exist_ok=True)
        publish_json(config_path, config)
    for name in ('save_model', 'results', 'logs', 'state', 'resume_state'):
        (output / name).mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OPENDETECT_RESUME_ROOT=str(output / 'resume_state'),
               OPENDETECT_EXPERIMENT_SIGNATURE=hashlib.sha256(
                   json.dumps(config, sort_keys=True).encode()).hexdigest(),
               OPENDETECT_DEADLINE=str(time.time() + args.session_hours * 3600))
    results = []
    for fold, identity in enumerate(identities):
        name = f'{dataset}_split_{split}_fold_{fold}'
        checkpoint = output / 'save_model' / (name + '.pt')
        result = output / 'results' / (name + '.json')
        scores = output / 'results' / (name + '.scores.npz')
        marker = output / 'state' / (name + '.completed.json')
        if marker.exists():
            results.append(verify_completed(marker, config, checkpoint, result, scores, identity))
            print('SKIP verified:', name, flush=True)
            continue
        if time.time() >= float(env['OPENDETECT_DEADLINE']):
            break
        common = ['--dset', dataset, '--split', str(split), '--fold', str(fold), '--seed', '2022',
                  '--gpu', str(args.gpu), '--num_workers', str(args.num_workers),
                  '--save_dir', str(output / 'save_model'), '--split_manifest_dir', str(manifests)]
        run_logged([sys.executable, str(ROOT / 'train.py'), *common, '--resume', '--epoch', '100',
                    '--batch_size', '128', '--lr', '.001', '--h', '128', '--c', '1',
                    '--temp_inter', '1.0', '--lamda', '.005'], output / 'logs' / (name + '.train.log'), env)
        records, _ = valid_records(output / 'resume_state' / name)
        if not records or records[0][0]['completed_epochs'] != 100:
            break  # graceful time-budget pause is not training completion
        run_logged([sys.executable, str(ROOT / 'test.py'), *common, '--batch_size', '256',
                    '--known_acceptance', '.95', '--metrics_out', str(result), '--scores_out', str(scores)],
                   output / 'logs' / (name + '.test.log'), env)
        metrics = json.loads(result.read_text(encoding='utf-8'))
        if (metrics['split_identity'] != identity or metrics['fold'] != fold
                or metrics['split_seed'] != 2022 + fold):
            raise ValueError('Evaluation returned wrong run identity')
        publish_json(marker, {'config': config, 'checkpoint_sha256': sha256(checkpoint),
                              'result_sha256': sha256(result), 'scores_sha256': sha256(scores)})
        results.append(verify_completed(marker, config, checkpoint, result, scores, identity))
        write_summary(output, results, config)
    write_summary(output, results, config)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=list(SCENARIOS), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split_manifest_dir', default=str(ROOT / 'grouped_manifests'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--session_hours', type=float, default=3.0)
    arguments = parser.parse_args()
    if arguments.session_hours <= 0 or arguments.num_workers < 0:
        parser.error('session_hours must be positive; num_workers nonnegative')
    run(arguments)
