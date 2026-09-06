"""Epoch-boundary recovery only; no model, loss, dataset, or metric changes.

Two alternating slots are selected by committed epoch, NOT by filename/mtime.
The other committed slot survives an interrupted overwrite. Drive/FUSE is not
transactional: checksums detect torn writes, but cannot guarantee cloud durability.
"""
import hashlib
import errno
import json
import os
from pathlib import Path
import random
import warnings

import numpy as np
import torch

VERSION = 2
SLOTS = ('last.pt', 'last.backup.pt')


def flush_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
        warnings.warn('Filesystem does not support fsync; relying on readback hash, not cloud durability.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        flush_file(stream)
    os.replace(temporary, path)
    if json.loads(path.read_text(encoding='utf-8')) != data:
        raise IOError('JSON readback failed: ' + str(path))


def publish_torch(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with open(temporary, 'wb') as stream:
        torch.save(data, stream)
        flush_file(stream)
    checksum = digest(temporary)
    os.replace(temporary, path)
    if digest(path) != checksum:
        raise IOError('Checkpoint readback failed: ' + str(path))
    return checksum


def load_owned_checkpoint(path):
    # Only own, hash-verified resume bundles: NumPy/Python RNG need pickle objects.
    return torch.load(path, map_location='cpu', weights_only=False)


def resume_directory(args):
    root = Path(os.environ.get('OPENDETECT_RESUME_ROOT', str(Path(args.save_dir) / 'resume_state')))
    path = root / f'{args.dset}_split_{args.split}_fold_{args.fold}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def signature(args):
    training_args = dict(vars(args))
    # Locations may move between runtimes. Actual code/data/index hashes remain strict.
    for name in ('save_dir', 'split_manifest_dir', 'gpu'):
        training_args.pop(name, None)
    return {
        'format': VERSION,
        'training_args': training_args,
        'experiment_signature': os.environ.get('OPENDETECT_EXPERIMENT_SIGNATURE', ''),
        'adapter_sha256': digest(__file__),
        'torch_version': str(torch.__version__),
        'numpy_version': str(np.__version__),
    }


def capture_rng(generator):
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'loader_generator': generator.get_state(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    generator.set_state(state['loader_generator'])
    cuda_state = state['torch_cuda']
    if cuda_state:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError('CUDA device count changed; do not silently resume on a different topology.')
        torch.cuda.set_rng_state_all(cuda_state)
    torch.backends.cudnn.deterministic = state['cudnn_deterministic']
    torch.backends.cudnn.benchmark = state['cudnn_benchmark']


def valid_records(folder):
    found = []
    saw_checkpoint = False
    for name in SLOTS:
        path = folder / name
        manifest = folder / (name + '.json')
        saw_checkpoint |= path.exists() or manifest.exists()
        try:
            record = json.loads(manifest.read_text(encoding='utf-8'))
            if record['format'] != VERSION or digest(path) != record['sha256']:
                raise ValueError('format or hash mismatch')
            if not isinstance(record['completed_epochs'], int) or record['completed_epochs'] < 1:
                raise ValueError('invalid epoch')
            found.append((record, path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            if path.exists() or manifest.exists():
                warnings.warn(f'Ignoring incomplete/corrupt resume slot {name}: {error}')
    return sorted(found, key=lambda pair: pair[0]['completed_epochs'], reverse=True), saw_checkpoint


def restore_training(args, model, optimizer, scheduler, generator, best_path):
    folder = resume_directory(args)
    records, saw_checkpoint = valid_records(folder)
    if not records:
        if saw_checkpoint:
            raise RuntimeError('No valid committed resume checkpoint. Files preserved; inspect Drive before restarting.')
        if Path(best_path).exists():
            raise RuntimeError('Best checkpoint without matching resume state; refusing to overwrite or resume old training.')
        print('RESUME: no saved epoch yet; start epoch 1.', flush=True)
        return 0, float('-inf')
    expected = signature(args)
    for record, path in records:
        # Configuration mismatches are not corruption: never fall back silently.
        if record['signature'] != expected:
            raise RuntimeError('Resume config/data/code/environment mismatch. Use the original environment or a new output folder.')
        try:
            state = load_owned_checkpoint(path)
        except Exception as error:
            warnings.warn(f'Cannot deserialize {path.name}; trying older slot: {error}')
            continue
        if state['signature'] != expected or state['completed_epochs'] != record['completed_epochs']:
            raise RuntimeError('Resume manifest and payload disagree.')
        completed = state['completed_epochs']
        if not 0 < completed <= args.epoch:
            raise RuntimeError('Saved epoch outside requested training range.')
        current_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        if current_gpu != state['device_name']:
            warnings.warn('GPU model changed; epoch recovery is supported, but bitwise-equivalent results are not guaranteed.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        # Best checkpoint may have been overwritten during an uncommitted epoch.
        # Restore the best snapshot belonging to the committed training state.
        publish_torch(best_path, state['best_checkpoint'])
        restore_rng(state['rng'], generator)
        print(f'RESUME: {completed}/{args.epoch} epoch tersimpan; '
              f'lanjut epoch {completed + 1 if completed < args.epoch else "SELESAI"} '
              f'dari {path.name}.', flush=True)
        return completed, state['best_accuracy']
    raise RuntimeError('All resume payloads failed to load; no automatic restart performed.')


def save_training(args, model, optimizer, scheduler, generator, best, best_path, epoch):
    folder = resume_directory(args)
    completed = epoch + 1
    expected = signature(args)
    state = {
        'signature': expected, 'completed_epochs': completed,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'rng': capture_rng(generator),
        'best_accuracy': best, 'best_checkpoint': load_owned_checkpoint(best_path),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    slot = folder / SLOTS[epoch % 2]
    checksum = publish_torch(slot, state)
    record = {'format': VERSION, 'completed_epochs': completed,
              'signature': expected, 'sha256': checksum}
    # Publish the manifest LAST: partial writes cannot be accepted as committed.
    publish_json(folder / (slot.name + '.json'), record)
    publish_json(folder / 'progress.json', {
        'completed_epochs': completed, 'total_epochs': args.epoch,
        'next_epoch': completed + 1 if completed < args.epoch else None,
        'slot': slot.name, 'checkpoint_sha256': checksum,
        'status': 'training_complete' if completed == args.epoch else 'checkpoint_committed',
    })
    print(f'CHECKPOINT COMMITTED: {completed}/{args.epoch}; {slot.name}', flush=True)
