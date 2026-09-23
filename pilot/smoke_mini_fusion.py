"""One-batch CPU integration smoke; output is not a model comparison metric."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import OpenDetectNet
from model_fusion import OpenDetectFusionNet
from pilot.pairing import USTC_LABELS, sha_file


def normalize_sequence(sequence, mask, split_id):
    sequence = sequence.astype(np.float32, copy=True)
    if not np.isfinite(sequence).all() or np.any(sequence[:, :, 0] < 0) or np.any(sequence[:, :, 2] < 0):
        raise ValueError('Invalid source sequence features')
    sequence[:, :, 0] = np.log1p(sequence[:, :, 0])
    sequence[:, :, 2] = np.log1p(sequence[:, :, 2])
    active_train = (split_id == 0)[:, None] & (mask != 0)
    if not np.any(active_train):
        raise ValueError('No train packets for normalization')
    stats = {}
    for col, key in ((0, 'log1p_payload_bytes'), (2, 'log1p_iat_seconds')):
        train_values = sequence[:, :, col][active_train]
        mean = float(train_values.mean())
        std = max(float(train_values.std()), 1e-6)
        sequence[:, :, col] = (sequence[:, :, col] - mean) / std
        stats[key] = {'mean': mean, 'std': std}
    sequence[mask == 0] = 0
    return sequence, stats


def smoke(cohort_dir, batch_size=4, seed=2026):
    cohort_dir = Path(cohort_dir)
    summary = json.loads((cohort_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'FEASIBILITY_ONLY_NOT_THESIS_EVALUATION':
        raise ValueError('Not a bounded feasibility cohort')
    npz = cohort_dir / 'mini_paired_cohort_NOT_THESIS_EVAL.npz'
    if sha_file(npz) != summary['outputs']['npz_sha256']:
        raise ValueError('Cohort NPZ checksum mismatch')
    if sha_file(cohort_dir / 'manifest.jsonl') != summary['outputs']['manifest_sha256']:
        raise ValueError('Cohort manifest checksum mismatch')
    with np.load(npz, allow_pickle=False) as archive:
        images = archive['image'][:]
        sequence = archive['sequence'][:]
        mask = archive['mask'][:]
        length = archive['length'][:]
        label = archive['label'][:]
        split_id = archive['split_id'][:]
    if len(images) != summary['samples'] or mask.shape != sequence.shape[:2]:
        raise ValueError('Cohort shape mismatch')
    if not np.array_equal(mask.sum(axis=1), length):
        raise ValueError('Invalid packet mask')
    known = {USTC_LABELS[name]: i for i, name in enumerate(summary['known_classes'])}
    unknown = USTC_LABELS[summary['unknown_holdout_class']]
    if np.any((label == unknown) & (split_id != 2)):
        raise ValueError('Unknown class leaked into train/validation')
    train_rows = np.where(split_id == 0)[0]
    if any(int(label[i]) not in known for i in train_rows):
        raise ValueError('Non-known train class')
    transformed, stats = normalize_sequence(sequence, mask, split_id)
    batch = train_rows[:batch_size]
    if len(batch) < 2:
        raise ValueError('Smoke batch needs at least two samples')
    image = torch.from_numpy(images[batch].astype(np.float32)[:, None] / 255.0)
    packet = torch.from_numpy(transformed[batch])
    lengths = torch.from_numpy(length[batch].astype(np.int64))
    target = torch.tensor([known[int(label[i])] for i in batch], dtype=torch.int64)
    torch.set_num_threads(2)
    run = {}
    for arm, n_features in (('B0_image_only', 0), ('E1_length_direction', 2), ('E2_plus_iat', 3)):
        torch.manual_seed(seed)
        model = (OpenDetectNet(channel=1, n_classes=len(known)) if n_features == 0 else
                 OpenDetectFusionNet(channel=1, n_classes=len(known), sequence_features=n_features))
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        if n_features == 0:
            _, _, _, losses = model.loss(image, target)
        else:
            _, _, _, losses = model.loss(image, packet, lengths, target)
        total = 0.5 * (losses['rec'] + losses['kld']) + 0.5 * losses['dis']
        if not torch.isfinite(total):
            raise FloatingPointError(f'Nonfinite {arm} loss')
        total.backward()
        if n_features and model.encoder.temporal.gru.weight_ih_l0.grad is None:
            raise AssertionError(f'No temporal gradient in {arm}')
        optimizer.step()
        run[arm] = {'one_batch_loss': float(total.detach()),
                    'parameters': sum(p.numel() for p in model.parameters())}
    return {'schema': 'mini-fusion-one-batch-smoke-v1', 'status': 'ENGINEERING_ONLY_NOT_EVALUATION',
            'cohort_npz_sha256': summary['outputs']['npz_sha256'], 'batch_size': len(batch),
            'seed': seed, 'normalization_fit': 'known train packets only',
            'normalizer': stats, 'arms': run}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.cohort_dir, args.batch_size, args.seed)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
