"""Short, single-seed feasibility training on the bounded flow-disjoint cohort."""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import OpenDetectNet
from model_fusion import OpenDetectFusionNet
from pilot.pairing import USTC_LABELS, sha_file
from pilot.smoke_mini_fusion import normalize_sequence
from utils import weight_init


class MiniDataset(Dataset):
    def __init__(self, images, sequence, lengths, labels, indices, reindex):
        self.images = images
        self.sequence = sequence
        self.lengths = lengths
        self.labels = labels
        self.indices = np.asarray(indices)
        self.reindex = reindex

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        i = int(self.indices[position])
        label = self.reindex.get(int(self.labels[i]), -1)
        return (torch.from_numpy(self.images[i].astype(np.float32)[None] / 255.0),
                torch.from_numpy(self.sequence[i]), int(self.lengths[i]), label)


def verify_manifest(path, summary):
    groups = defaultdict(set)
    seen = set()
    unknown_label = USTC_LABELS[summary['unknown_holdout_class']]
    known_labels = {USTC_LABELS[name] for name in summary['known_classes']}
    with Path(path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != summary['samples']:
        raise ValueError('Manifest/cohort sample count mismatch')
    for row in rows:
        key = (row['source'], row['row'])
        if key in seen:
            raise ValueError('Duplicate NPZ row in manifest')
        seen.add(key)
        groups[row['flow']].add(row['split'])
        if row['class'] == summary['unknown_holdout_class']:
            if row['split'] != 'test' or row['label'] != unknown_label:
                raise ValueError('Unknown holdout leaked out of test')
        elif row['label'] not in known_labels:
            raise ValueError('Unexpected known class label')
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('A flow group crosses splits')
    return rows


def _weighted_f1(labels, predictions):
    total = len(labels)
    result = 0.0
    for class_id in np.unique(labels):
        tp = int(np.sum((labels == class_id) & (predictions == class_id)))
        fp = int(np.sum((labels != class_id) & (predictions == class_id)))
        fn = int(np.sum((labels == class_id) & (predictions != class_id)))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        result += float(np.sum(labels == class_id)) / total * f1
    return result


def _auc_binary(labels, scores):
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('AUROC requires known and unknown test samples')
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def _evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores, all_preds = [], [], []
    with torch.no_grad():
        for image, sequence, lengths, labels in loader:
            image = image.to(device)
            sequence = sequence.to(device)
            lengths = lengths.to(device)
            if isinstance(model, OpenDetectFusionNet):
                _, _, kl, _ = model(image, sequence, lengths)
            else:
                _, _, kl, _ = model(image)
            scores, predictions = kl.min(dim=1)
            all_labels.append(labels.numpy())
            all_scores.append(scores.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())
    return (np.concatenate(all_labels), np.concatenate(all_scores).astype(np.float64),
            np.concatenate(all_preds))


def _threshold(validation_scores, known_acceptance=0.95):
    if not np.isfinite(validation_scores).all() or not len(validation_scores):
        raise ValueError('Validation scores must be finite and nonempty')
    rank = max(1, math.ceil(known_acceptance * len(validation_scores)))
    selected = np.partition(validation_scores, rank - 1)[rank - 1]
    return float(np.nextafter(selected, np.inf))


def _run_arm(arm, model, loaders, device, max_epochs, patience, learning_rate, lamda, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_accuracy, best_loss, best_epoch, stale = None, -1.0, float('inf'), -1, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        seen, loss_sum = 0, 0.0
        generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(loaders['train'].dataset, batch_size=loaders['batch_size'],
                                  shuffle=True, generator=generator, num_workers=0)
        for image, sequence, lengths, labels in train_loader:
            image, sequence = image.to(device), sequence.to(device)
            lengths, labels = lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, OpenDetectFusionNet):
                _, _, _, terms = model.loss(image, sequence, lengths, labels)
            else:
                _, _, _, terms = model.loss(image, labels)
            total = lamda * (terms['rec'] + terms['kld']) + (1 - lamda) * terms['dis']
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite {arm} loss at epoch {epoch + 1}')
            total.backward()
            optimizer.step()
            loss_sum += float(total.detach()) * len(labels)
            seen += len(labels)
        val_loss_sum, val_correct, val_seen = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for image, sequence, lengths, labels in loaders['val']:
                image, sequence = image.to(device), sequence.to(device)
                lengths, labels = lengths.to(device), labels.to(device)
                if isinstance(model, OpenDetectFusionNet):
                    _, _, predictions, terms = model.loss(image, sequence, lengths, labels)
                else:
                    _, _, predictions, terms = model.loss(image, labels)
                total = lamda * (terms['rec'] + terms['kld']) + (1 - lamda) * terms['dis']
                val_loss_sum += float(total) * len(labels)
                val_correct += int((predictions == labels).sum())
                val_seen += len(labels)
        val_loss = val_loss_sum / val_seen
        val_accuracy = val_correct / val_seen
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / seen,
                        'val_loss': val_loss, 'val_accuracy': val_accuracy})
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch, stale = val_accuracy, val_loss, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f'No finite validation checkpoint for {arm}')
    model.load_state_dict(best_state)
    del best_state
    _, validation_scores, _ = _evaluate(model, loaders['val'], device)
    known_labels, known_scores, known_predictions = _evaluate(model, loaders['known_test'], device)
    _, unknown_scores, _ = _evaluate(model, loaders['unknown_test'], device)
    if not np.isfinite(np.concatenate([validation_scores, known_scores, unknown_scores])).all():
        raise FloatingPointError(f'Nonfinite evaluation scores for {arm}')
    threshold = _threshold(validation_scores)
    open_labels = np.r_[np.zeros(len(known_scores), dtype=np.int64),
                        np.ones(len(unknown_scores), dtype=np.int64)]
    open_scores = np.r_[known_scores, unknown_scores]
    open_predictions = (open_scores >= threshold).astype(np.int64)
    result = {'best_epoch': best_epoch, 'epochs_run': len(history),
              'best_validation_accuracy': best_accuracy, 'best_validation_loss': best_loss,
              'threshold_from_known_validation_95pct': threshold,
              'known_validation_acceptance': float(np.mean(validation_scores < threshold)),
              'known_test_accuracy': float(np.mean(known_predictions == known_labels)),
              'known_test_weighted_f1': _weighted_f1(known_labels, known_predictions),
              'unknown_auroc': _auc_binary(open_labels, open_scores),
              'unknown_test_rejection': float(np.mean(unknown_scores >= threshold)),
              'known_test_acceptance': float(np.mean(known_scores < threshold)),
              'open_binary_accuracy': float(np.mean(open_predictions == open_labels)),
              'open_binary_f1_unknown_positive': _weighted_f1(open_labels, open_predictions),
              'known_test_samples': int(len(known_scores)), 'unknown_test_samples': int(len(unknown_scores)),
              'training_history': history, 'elapsed_seconds': time.perf_counter() - start_time,
              'checkpoint_policy': 'best state kept in RAM only; no model/checkpoint written'}
    return result


def run(cohort_dir, output, epochs=8, patience=2, batch_size=128, seed=2022,
        learning_rate=1e-3, lamda=0.005, device_name='auto'):
    if epochs < 1 or patience < 1 or batch_size < 2 or learning_rate <= 0 or not 0 <= lamda <= 1:
        raise ValueError('Invalid training configuration')
    cohort_dir = Path(cohort_dir)
    summary = json.loads((cohort_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'FEASIBILITY_ONLY_NOT_THESIS_EVALUATION':
        raise ValueError('Unsupported cohort scope')
    npz = cohort_dir / 'mini_paired_cohort_NOT_THESIS_EVAL.npz'
    manifest_path = cohort_dir / 'manifest.jsonl'
    if sha_file(npz) != summary['outputs']['npz_sha256'] or sha_file(manifest_path) != summary['outputs']['manifest_sha256']:
        raise ValueError('Cohort checksum mismatch')
    manifest = verify_manifest(manifest_path, summary)
    with np.load(npz, allow_pickle=False) as data:
        images, sequence, mask = data['image'][:], data['sequence'][:], data['mask'][:]
        lengths, labels, split_id = data['length'][:], data['label'][:], data['split_id'][:]
    if not np.array_equal(mask.sum(1), lengths) or len(images) != len(manifest):
        raise ValueError('Cohort sequence metadata mismatch')
    sequence, norm_stats = normalize_sequence(sequence, mask, split_id)
    label_map = {USTC_LABELS[name]: i for i, name in enumerate(summary['known_classes'])}
    unknown_label = USTC_LABELS[summary['unknown_holdout_class']]
    indices = {
        'train': np.flatnonzero((split_id == 0) & np.isin(labels, list(label_map))),
        'val': np.flatnonzero((split_id == 1) & np.isin(labels, list(label_map))),
        'known_test': np.flatnonzero((split_id == 2) & np.isin(labels, list(label_map))),
        'unknown_test': np.flatnonzero((split_id == 2) & (labels == unknown_label)),
    }
    if any(not len(v) for v in indices.values()):
        raise ValueError('An evaluation partition is empty')
    datasets = {key: MiniDataset(images, sequence, lengths, labels, ids, label_map if key != 'unknown_test' else {})
                for key, ids in indices.items()}
    loaders = {key: DataLoader(value, batch_size=batch_size, shuffle=False, num_workers=0)
               for key, value in datasets.items()}
    loaders['batch_size'] = batch_size
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    arms, shared_initial = {}, None
    for name, n_features in (('B0_image_only', 0), ('E1_length_direction', 2), ('E2_plus_iat', 3)):
        torch.manual_seed(seed)
        model = (OpenDetectNet(channel=1, n_classes=len(label_map)) if n_features == 0 else
                 OpenDetectFusionNet(channel=1, n_classes=len(label_map), sequence_features=n_features))
        model.apply(weight_init)
        if name == 'B0_image_only':
            shared_initial = {
                'image_encoder': {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()
                                  if k not in ('mu.weight', 'mu.bias', 'logvar.weight', 'logvar.bias')},
                'decoder': {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()},
                'prototypes': model.prototypes.detach().cpu().clone(),
            }
        else:
            model.encoder.image_encoder.load_state_dict(shared_initial['image_encoder'], strict=False)
            model.decoder.load_state_dict(shared_initial['decoder'])
            with torch.no_grad():
                model.prototypes.copy_(shared_initial['prototypes'].to(model.prototypes.device))
        model.to(device)
        arms[name] = _run_arm(name, model, loaders, device, epochs, patience,
                              learning_rate, lamda, seed)
        arms[name]['parameters'] = sum(p.numel() for p in model.parameters())
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result = {'schema': 'mini-open-set-training-v1',
              'status': 'FEASIBILITY_ONLY_NOT_THESIS_EVALUATION',
              'cohort_npz_sha256': summary['outputs']['npz_sha256'],
              'manifest_sha256': summary['outputs']['manifest_sha256'],
              'known_classes': summary['known_classes'],
              'unknown_holdout_class': summary['unknown_holdout_class'],
              'samples': {key: int(len(ids)) for key, ids in indices.items()},
              'device': str(device), 'seed': seed, 'max_epochs': epochs, 'patience': patience,
              'batch_size': batch_size, 'learning_rate': learning_rate, 'lamda': lamda,
              'initialization': 'same image trunk, decoder, and class prototypes for B0/E1/E2',
              'sequence_features': ['log1p_payload_bytes', 'direction', 'log1p_iat_seconds'],
              'normalizer_fit': 'known training packets only', 'normalizer': norm_stats,
              'known_acceptance_target': 0.95,
              'threshold_source': 'known validation minimum-KL scores only',
              'selection': 'minimum validation composite OpenDetect loss; no unknown validation',
              'arms': arms,
              'limitations': 'One seed, five USTC classes, bounded source prefixes; exploratory only.'}
    output = Path(output)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--lamda', type=float, default=0.005)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    print(json.dumps(run(args.cohort_dir, args.output, args.epochs, args.patience,
                         args.batch_size, args.seed, args.learning_rate, args.lamda, args.device), indent=2))
