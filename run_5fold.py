import argparse
import json
import os
import subprocess
import sys

import numpy as np
from provenance import PROTOCOL, code_identity


SUMMARY_METRICS = [
    'closed_accuracy',
    'closed_precision',
    'closed_recall',
    'closed_f1',
    'auroc',
    'open_accuracy',
    'open_precision',
    'open_recall',
    'open_f1',
    'binary_macro_f1',
    'binary_weighted_f1',
    'known_test_acceptance',
    'unknown_test_rejection',
    'threshold',
]


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run five paper-aligned, stratified 8:1:1 Open-Detect experiments'
    )
    parser.add_argument('--dset', default='mal', choices=['mal', 'USTC', 'combined_USTC_mal'])
    parser.add_argument('--split', type=int, default=0, help='unknown-class scenario index')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; use -1 for CPU')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--temp_inter', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=0.005)
    parser.add_argument('--known_acceptance', type=float, default=0.95)
    parser.add_argument('--save_dir', default='./save_model_v2')
    parser.add_argument('--results_dir', default='./results_v2')
    parser.add_argument('--skip_train', action='store_true')
    return parser


def validate_aggregate(fold_results):
    if not fold_results:
        raise ValueError('Cannot aggregate empty results')
    if len({r['decoder_version'] for r in fold_results}) != 1:
        raise ValueError('Cannot aggregate different decoder versions')
    for field in ['training_code_identity', 'data_manifest', 'unknown_data_manifest']:
        if any(result[field] != fold_results[0][field] for result in fold_results):
            raise ValueError('Cannot aggregate different ' + field)
    scientific_args = ['arch', 'h', 'c', 'epoch', 'batch_size', 'lr', 'lamda', 'temp_inter', 'temp_intra']
    configs = [{key: (r.get('training_config') or {}).get(key) for key in scientific_args}
               for r in fold_results]
    if any(config != configs[0] for config in configs):
        raise ValueError('Cannot aggregate different training hyperparameters')


def run(args):
    if args.folds < 2:
        raise ValueError('folds must be at least 2 to calculate mean and standard deviation')

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(args.save_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Fail before starting a costly run if any training checkpoint would be overwritten.
    if not args.skip_train:
        for fold in range(args.folds):
            path = os.path.join(save_dir, '{}_split_{}_fold_{}.pt'.format(args.dset, args.split, fold))
            if os.path.exists(path):
                raise FileExistsError('Existing checkpoint; choose a new save_dir: ' + path)

    fold_results = []
    for fold in range(args.folds):
        common = [
            '--dset', args.dset,
            '--split', str(args.split),
            '--fold', str(fold),
            '--seed', str(args.seed),
            '--gpu', str(args.gpu),
            '--num_workers', str(args.num_workers),
            '--save_dir', save_dir,
        ]
        if not args.skip_train:
            train_command = [
                sys.executable,
                os.path.join(repo_dir, 'train.py'),
                *common,
                '--epoch', str(args.epoch),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--h', str(args.h),
                '--c', str(args.c),
                '--temp_inter', str(args.temp_inter),
                '--lamda', str(args.lamda),
            ]
            subprocess.run(train_command, check=True, cwd=repo_dir)

        fold_output = os.path.join(
            results_dir,
            '{}_split_{}_fold_{}.json'.format(args.dset, args.split, fold),
        )
        test_command = [
            sys.executable,
            os.path.join(repo_dir, 'test.py'),
            *common,
            '--batch_size', str(args.eval_batch_size),
            '--known_acceptance', str(args.known_acceptance),
            '--metrics_out', fold_output,
            '--scores_out', os.path.splitext(fold_output)[0] + '.scores.npz',
        ]
        subprocess.run(test_command, check=True, cwd=repo_dir)
        with open(fold_output, encoding='utf-8') as input_file:
            fold_results.append(json.load(input_file))

    validate_aggregate(fold_results)
    summary = {
        metric: {
            'mean': float(np.mean([result[metric] for result in fold_results])),
            'std': float(np.std([result[metric] for result in fold_results], ddof=1)),
        }
        for metric in SUMMARY_METRICS
    }
    output = {
        'protocol': PROTOCOL,
        'code_identity': code_identity(),
        'interpretation': 'Seeded 80:10:10 repetitions, not disjoint classical 5-fold CV. All seeds retained.',
        'dataset': args.dset,
        'scenario_split': args.split,
        'base_seed': args.seed,
        'folds': args.folds,
        'known_acceptance_target': args.known_acceptance,
        'per_fold': fold_results,
        'summary': summary,
    }
    summary_path = os.path.join(
        results_dir,
        '{}_split_{}_summary.json'.format(args.dset, args.split),
    )
    with open(summary_path, 'w', encoding='utf-8') as output_file:
        json.dump(output, output_file, indent=2, sort_keys=True)
        output_file.write('\n')

    print('\nFive-run result (mean +/- sample standard deviation)')
    for metric in SUMMARY_METRICS:
        values = summary[metric]
        print('{}: {:.6f} +/- {:.6f}'.format(metric, values['mean'], values['std']))
    print('Summary: {}'.format(summary_path))
    return output


if __name__ == '__main__':
    run(build_parser().parse_args())
