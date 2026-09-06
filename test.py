import argparse
import json
import math
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data.dataset import get_dataset_splits
from data.splits import get_splits
from model import OpenDetectNet
from utils import setup_seed
from provenance import dataset_manifest, code_identity, sha256, PROTOCOL
import warnings
from data.grouped import get_grouped_splits, verify_checkpoint_split


def finite_scores(values):
    # NumPy may otherwise cast a Python float64 threshold back to float32,
    # losing the nextafter step and rejecting the boundary validation sample.
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError('Scores must be non-empty and finite')
    return scores


def get_output(model, data_loader, device=None):
    """Return labels, minimum KL score, and nearest-prototype predictions."""
    if device is None:
        device = next(model.parameters()).device

    labels = []
    scores = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            _, _, kl_divergences, _ = model(images)
            batch_scores, batch_predictions = torch.min(kl_divergences, dim=1)
            labels.append(batch_labels.cpu().numpy())
            scores.append(batch_scores.cpu().numpy())
            predictions.append(batch_predictions.cpu().numpy())

    if not labels:
        raise ValueError('Cannot evaluate an empty dataset')
    return (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(predictions),
    )


def threshold_from_known_validation(known_validation_scores, known_acceptance=0.95):
    """Choose a threshold that accepts at least 95% of known validation samples.

    Equations 21-22 use a strict ``score < threshold`` known decision, so the
    returned value is the next representable float above the selected score.
    Unknown samples are deliberately not used when choosing the threshold.
    """
    scores = finite_scores(known_validation_scores)
    if not len(scores):
        raise ValueError('known_validation_scores must not be empty')
    if not 0 < known_acceptance <= 1:
        raise ValueError('known_acceptance must be in the interval (0, 1]')

    rank = max(1, math.ceil(known_acceptance * len(scores)))
    selected_score = np.partition(scores, rank - 1)[rank - 1]
    threshold = float(np.nextafter(selected_score, np.inf))
    if not np.isfinite(threshold):
        raise ValueError('No finite threshold above the selected score')
    return threshold


def closed_world_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0,
    )
    return {
        'closed_accuracy': float(accuracy_score(labels, predictions)),
        'closed_precision': float(precision),
        'closed_recall': float(recall),
        'closed_f1': float(f1),
    }


def open_world_metrics(known_scores, unknown_scores, threshold):
    """Evaluate known=0 versus unknown=1 using the fixed validation threshold."""
    known_scores, unknown_scores = finite_scores(known_scores), finite_scores(unknown_scores)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    y_true = np.concatenate([
        np.zeros(len(known_scores), dtype=np.int64),
        np.ones(len(unknown_scores), dtype=np.int64),
    ])
    y_score = np.concatenate([known_scores, unknown_scores])
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'open_accuracy': float(accuracy_score(y_true, y_pred)),
        'open_precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'open_recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'open_f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'binary_macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'binary_weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'unknown_true_positive': int(np.sum((y_true == 1) & (y_pred == 1))),
        'known_false_positive': int(np.sum((y_true == 0) & (y_pred == 1))),
        'unknown_false_negative': int(np.sum((y_true == 1) & (y_pred == 0))),
        'known_true_negative': int(np.sum((y_true == 0) & (y_pred == 0))),
        'known_test_acceptance': float(np.mean(known_scores < threshold)),
        'unknown_test_rejection': float(np.mean(unknown_scores >= threshold)),
    }


def evaluate_openset(
    model,
    validation_loader,
    known_test_loader,
    unknown_test_loader,
    known_acceptance=0.95,
    scores_out=None,
):
    device = next(model.parameters()).device
    _, validation_scores, _ = get_output(model, validation_loader, device)
    known_labels, known_scores, known_predictions = get_output(model, known_test_loader, device)
    _, unknown_scores, _ = get_output(model, unknown_test_loader, device)
    validation_scores = finite_scores(validation_scores)
    known_scores, unknown_scores = finite_scores(known_scores), finite_scores(unknown_scores)

    threshold = threshold_from_known_validation(validation_scores, known_acceptance)
    metrics = closed_world_metrics(known_labels, known_predictions)
    metrics.update(open_world_metrics(known_scores, unknown_scores, threshold))
    metrics.update({
        'metric_schema': 'replication-v2; open_f1=binary_unknown_positive; closed_f1=weighted_known',
        'threshold': threshold,
        'validation_known_acceptance': float(np.mean(validation_scores < threshold)),
        'known_validation_samples': int(len(validation_scores)),
        'known_test_samples': int(len(known_scores)),
        'unknown_test_samples': int(len(unknown_scores)),
    })
    # Retain scores for later diagnostics without retraining or tuning on test labels.
    if scores_out:
        os.makedirs(os.path.dirname(os.path.abspath(scores_out)), exist_ok=True)
        np.savez_compressed(scores_out, validation_scores=validation_scores,
                            known_scores=known_scores, unknown_scores=unknown_scores,
                            known_labels=known_labels, known_predictions=known_predictions,
                            threshold=np.float64(threshold))
    return metrics


def checkpoint_path(args):
    if args.model_path:
        return args.model_path
    return os.path.join(
        args.save_dir,
        '{}_split_{}_fold_{}.pt'.format(args.dset, args.split, args.fold),
    )


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            'Unsupported checkpoint format: {}. Retrain it with the updated train.py.'.format(path)
        )

    config = checkpoint['model_config']
    model = OpenDetectNet(
        config['arch'],
        config['channel'],
        config['latent_dim'],
        config['n_classes'],
        config['temp_inter'],
        config.get('temp_intra', 1.0),
        init=False,
        decoder_version=config.get('decoder_version', 1),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate paper-aligned Open-Detect')
    parser.add_argument('--dset', default='mal', choices=['mal', 'USTC', 'combined_USTC_mal'])
    parser.add_argument('--split', type=int, default=0, help='unknown-class scenario index')
    parser.add_argument('--fold', type=int, default=0, help='0-based repeated split index')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--known_acceptance', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; use -1 for CPU')
    parser.add_argument('--save_dir', default='./save_model_v2')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--metrics_out', default=None)
    parser.add_argument('--scores_out', default=None, help='Optional NPZ of validation/test scores for diagnostics')
    parser.add_argument('--split_manifest_dir', default=None)
    return parser


def evaluate(args):
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    split_seed = args.seed + args.fold
    setup_seed(split_seed)

    known_classes, unknown_classes, known_dataset, unknown_dataset = get_splits(
        args.dset,
        num_split=args.split,
    )
    split_identity = None
    if getattr(args, 'split_manifest_dir', None):
        _, validation_set, known_test_set, unknown_test_set, split_identity = get_grouped_splits(
            args.dset, args.split, split_seed, args.split_manifest_dir)
    else:
        _, validation_set, known_test_set = get_dataset_splits(
            known_dataset, select_classes=known_classes, target_transform='reindex', seed=split_seed)
        _, _, unknown_test_set = get_dataset_splits(
            unknown_dataset, select_classes=unknown_classes, target_transform='open', seed=split_seed)

    loader_args = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'drop_last': False,
    }
    validation_loader = DataLoader(validation_set, **loader_args)
    known_test_loader = DataLoader(known_test_set, **loader_args)
    unknown_test_loader = DataLoader(unknown_test_set, **loader_args)

    path = checkpoint_path(args)
    model, checkpoint = load_model(path, device)
    if checkpoint and checkpoint.get('known_classes') != known_classes:
        raise ValueError('Checkpoint known classes do not match the selected scenario')
    expected = {'dataset': args.dset, 'scenario_split': args.split,
                'fold': args.fold, 'split_seed': split_seed}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError('Checkpoint {} mismatch: expected {}, got {}'.format(key, value, checkpoint.get(key)))
    protocol = split_identity['protocol'] if split_identity else PROTOCOL
    verify_checkpoint_split(checkpoint, split_identity)
    if checkpoint.get('protocol', PROTOCOL) != protocol:
        raise ValueError('Checkpoint split protocol differs from evaluation')
    manifest = dataset_manifest(known_dataset)
    if checkpoint.get('data_manifest'):
        if checkpoint['data_manifest']['files'] != manifest['files']:
            raise ValueError('Dataset fingerprint differs from training checkpoint')
    else:
        warnings.warn('Legacy checkpoint has no dataset fingerprint; identity is not verified')
    unknown_manifest = manifest if unknown_dataset == known_dataset else dataset_manifest(unknown_dataset)

    metrics = evaluate_openset(
        model,
        validation_loader,
        known_test_loader,
        unknown_test_loader,
        known_acceptance=args.known_acceptance,
        scores_out=getattr(args, 'scores_out', None),
    )
    metrics.update({
        'dataset': args.dset,
        'scenario_split': args.split,
        'fold': args.fold,
        'split_seed': split_seed,
        'checkpoint': path,
        'checkpoint_sha256': sha256(path),
        'best_epoch': checkpoint.get('best_epoch'),
        'training_config': checkpoint.get('training_config'),
        'decoder_version': model.decoder_version,
        'protocol': protocol,
        'split_identity': split_identity,
        'data_manifest': manifest,
        'unknown_data_manifest': unknown_manifest,
        'evaluation_code_identity': code_identity(),
        'training_code_identity': checkpoint.get('code_identity'),
    })

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, 'w', encoding='utf-8') as output_file:
            json.dump(metrics, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
    return metrics


if __name__ == '__main__':
    evaluate(build_parser().parse_args())
