import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataset_splits
from data.splits import get_splits
from model import OpenDetectNet, train_model, validate_model
from utils import reset_prototype, setup_seed, weight_init
from provenance import dataset_manifest, code_identity, PROTOCOL
from data.grouped import get_grouped_splits
from resume_support import restore_training, save_training, publish_torch


def build_parser():
    parser = argparse.ArgumentParser(description='Train paper-aligned Open-Detect')
    parser.add_argument('--dset', default='mal', choices=['mal', 'USTC', 'combined_USTC_mal'])
    parser.add_argument('--split', type=int, default=0, help='unknown-class scenario index')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--h', type=int, default=128, help='latent dimension')
    parser.add_argument('--c', type=int, default=1, help='image channels')
    parser.add_argument('--temp_inter', type=float, default=1.0, help='1/gamma in Eq. 18')
    parser.add_argument('--temp_intra', type=float, default=1.0, help='kept for checkpoint compatibility')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; use -1 for CPU')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--lamda', type=float, default=0.005, help='lambda in Eq. 20')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--fold', type=int, default=0, help='0-based repeated split index')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', default='./save_model_v2')
    parser.add_argument('--split_manifest_dir', default=None)
    parser.add_argument('--resume', action='store_true', help='Restore only matching epoch state')
    return parser


def checkpoint_path(args):
    return os.path.join(
        args.save_dir,
        '{}_split_{}_fold_{}.pt'.format(args.dset, args.split, args.fold),
    )


def train(args):
    output_path = checkpoint_path(args)
    if os.path.exists(output_path) and not getattr(args, 'resume', False):
        raise FileExistsError('Checkpoint exists; use a separate output directory: ' + output_path)
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    split_seed = args.seed + args.fold
    setup_seed(split_seed)
    os.makedirs(args.save_dir, exist_ok=True)

    known_classes, _, known_dataset, _ = get_splits(args.dset, num_split=args.split)
    manifest = dataset_manifest(known_dataset)
    identity = code_identity()
    args.num_classes = len(known_classes)
    split_identity = None
    if getattr(args, 'split_manifest_dir', None):
        train_set, val_set, _, _, split_identity = get_grouped_splits(
            args.dset, args.split, split_seed, args.split_manifest_dir)
    else:
        train_set, val_set, _ = get_dataset_splits(
            known_dataset, select_classes=known_classes,
            target_transform='reindex', seed=split_seed)
    protocol = split_identity['protocol'] if split_identity else PROTOCOL
    args.recovery_identity = {'data': manifest, 'code': identity, 'split': split_identity}

    generator = torch.Generator()
    generator.manual_seed(split_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = OpenDetectNet(
        args.arch,
        args.c,
        args.h,
        args.num_classes,
        args.temp_inter,
        args.temp_intra,
    ).to(device)
    model.apply(weight_init)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    print(args)
    print('Device: {} | split seed: {} | train/val: {}/{}'.format(
        device, split_seed, len(train_set), len(val_set)
    ))

    best = float('-inf')
    start_epoch = 0
    if getattr(args, 'resume', False):
        start_epoch, best = restore_training(args, model, optimizer, scheduler, generator, output_path)
    deadline = float(os.environ.get('OPENDETECT_DEADLINE', 'inf'))
    for epoch in range(start_epoch, args.epoch):
        train_model(model, args, train_loader, epoch, optimizer)
        if epoch in [50, 80]:
            reset_prototype(model, train_loader)
        val_acc = validate_model(model, args, val_loader, epoch)
        scheduler.step()

        if val_acc > best:
            publish_torch(
                output_path,
                {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'arch': args.arch,
                        'channel': args.c,
                        'latent_dim': args.h,
                        'n_classes': args.num_classes,
                        'temp_inter': args.temp_inter,
                        'temp_intra': args.temp_intra,
                        'decoder_version': model.decoder_version,
                    },
                    'known_classes': known_classes,
                    'dataset': args.dset,
                    'scenario_split': args.split,
                    'fold': args.fold,
                    'split_seed': split_seed,
                    'validation_accuracy': val_acc,
                    'best_epoch': epoch + 1,
                    'protocol': protocol,
                    'split_identity': split_identity,
                    'data_manifest': manifest,
                    'code_identity': identity,
                    'training_config': vars(args).copy(),
                },
            )
            best = val_acc

        if getattr(args, 'resume', False):
            save_training(args, model, optimizer, scheduler, generator, best, output_path, epoch)
            if time.time() >= deadline and epoch + 1 < args.epoch:
                print('PAUSED safely at epoch boundary; Run all again to resume.', flush=True)
                return None

    print('Finished. Best validation accuracy: {:.6f}'.format(best))
    print('Checkpoint: {}'.format(output_path))
    return output_path


if __name__ == '__main__':
    train(build_parser().parse_args())
