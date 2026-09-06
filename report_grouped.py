"""Read-only reporting for completed grouped experiments. No training or ablation."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

TRAIN_COMMIT = 'c729403f7fd4d1207522e3ca793cafb5f8cb8eb0'
# P01 Open-Detect, DOI 10.1109/TIFS.2025.3612141, Tables V-VII, pp.11-12.
# Percent units, mean and printed standard deviation; reference, not measured here.
PAPER = {
    'A-1': (98.20, .65, 98.24, .51, 'V'),
    'A-2': (89.22, .67, 89.10, .35, 'V'),
    'A-3': (95.51, .35, 91.17, .45, 'V'),
    'B-1': (90.20, .19, 90.71, .15, 'VI'),
    'B-2': (90.34, .24, 89.62, .21, 'VI'),
    'B-3': (85.94, .31, 85.97, .29, 'VI'),
    'C-1': (93.31, .34, 93.56, .38, 'VII'),
    'C-2': (97.45, .29, 98.65, .17, 'VII'),
}
METRICS = ['open_accuracy', 'open_f1', 'open_precision', 'open_recall',
           'binary_macro_f1', 'binary_weighted_f1', 'auroc', 'closed_accuracy',
           'closed_precision', 'closed_recall', 'closed_f1',
           'known_test_acceptance', 'unknown_test_rejection']
LIMITS = '''Evaluasi tambahan tanpa training dan tanpa ablasi. Hanya marker completed
yang lolos pemeriksaan identitas dan hash yang dibaca. Semua lima seed 2022-2026
dilaporkan; mean dan sample SD (ddof=1) hanya dibuat setelah lima seed lengkap.
Protokol ini exact-image-group-stratified 80:10:10 berulang, bukan lima test-fold
saling lepas. Tidak membuktikan bebas overlap flow/PCAP. Jumlah data sumber juga
tidak identik dengan Tabel II paper. open_f1 adalah binary F1 dengan unknown
sebagai kelas positif; jenis averaging F1 paper belum terkonfirmasi. Nilai paper
hanya referensi, bukan target threshold dan bukan bukti replikasi identik.
closed_* mengukur klasifikasi kelas known dalam skenario, bukan eksperimen
closed-world seluruh dataset Tabel IV. Tidak ada pemilihan threshold dari test.
Grafik memakai fold 0/seed 2022 yang ditetapkan sebelumnya, bukan seed terbaik.
t-SNE adalah pilihan implementasi visualisasi ini; paper tidak menetapkan
algoritme proyeksi secara eksplisit dalam bagian yang diperiksa. Pemisahan visual
bukan metrik keberhasilan. Heatmap menunjukkan sensitivitas pixel, bukan bukti
kausal maupun identifikasi field TLS. Waktu ekstraksi PCAP tidak tersedia dari
NPZ. Waktu inferensi yang diukur di sini tidak langsung sebanding dengan V100
paper; perangkat, scope pengukuran, dan pengaturan timing dicatat terpisah.'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def bind_source(source):
    """Import scientific operations only from the unmodified training checkout."""
    source = Path(source).resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != TRAIN_COMMIT:
        raise ValueError('Wrong training source commit: ' + head)
    # Verify tracked bytes, including files hidden by a previous sparse checkout.
    names = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '--name-only',
                                     'HEAD'], text=True).splitlines()
    for name in names:
        if not (name.endswith('.py') or name.startswith('grouped_manifests/')):
            continue
        expected = subprocess.check_output(['git', '-C', str(source), 'show', 'HEAD:' + name])
        if digest(source / name) != hashlib.sha256(expected).hexdigest():
            raise ValueError('Training source modified: ' + name)
    sys.path.insert(0, str(source))
    import test as evaluator
    from data import grouped
    from provenance import code_identity
    if Path(evaluator.__file__).resolve().parent != source:
        raise ValueError('Scientific evaluator imported from wrong directory')
    return evaluator, grouped, code_identity()


def verify_run(folder, scenario, fold, evaluator, grouped, source_identity, manifests):
    dataset, split = grouped.SCENARIOS[scenario]
    name = f'{dataset}_split_{split}_fold_{fold}'
    marker_path = folder / 'state' / (name + '.completed.json')
    if not marker_path.exists():
        return None  # A running/paused fold is not a final result.
    config = read_json(folder / 'GROUPED_CONFIG.json')
    marker = read_json(marker_path)
    identity = grouped.bundle_record(manifests, dataset, split, 2022 + fold)[3]
    if marker['config'] != config or config['code'] != source_identity:
        raise ValueError('Config/source identity mismatch')
    if (config['scenario'], config['dataset'], config['split'], config['epoch'],
            config['folds'], config['base_seed'], config['known_acceptance']) != (
            scenario, dataset, split, 100, 5, 2022, .95):
        raise ValueError('Not the expected completed grouped experiment')
    if config['splits'][fold] != identity or config['protocol'] != grouped.GROUP_PROTOCOL:
        raise ValueError('Manifest identity mismatch')
    paths = {'result': folder / 'results' / (name + '.json'),
             'scores': folder / 'results' / (name + '.scores.npz'),
             'checkpoint': folder / 'save_model' / (name + '.pt')}
    # Check all three, including checkpoint, before declaring verified.
    for kind, path in paths.items():
        if digest(path) != marker[kind + '_sha256']:
            raise ValueError('Artifact hash mismatch: ' + kind)
    result = read_json(paths['result'])
    if (result['fold'], result['split_seed'], result['dataset'], result['scenario_split']) != (
            fold, 2022 + fold, dataset, split):
        raise ValueError('Result belongs to another run')
    if (result['split_identity'] != identity or result['protocol'] != grouped.GROUP_PROTOCOL
            or result['checkpoint_sha256'] != marker['checkpoint_sha256']
            or result['training_code_identity'] != source_identity
            or result['evaluation_code_identity'] != source_identity):
        raise ValueError('Result provenance mismatch')
    with np.load(paths['scores'], allow_pickle=False) as archive:
        scores = {k: archive[k].copy() for k in archive.files}
    for k in ('validation_scores', 'known_scores', 'unknown_scores'):
        evaluator.finite_scores(scores[k])
    threshold = evaluator.threshold_from_known_validation(scores['validation_scores'], .95)
    if threshold != float(scores['threshold']) or threshold != result['threshold']:
        raise ValueError('Threshold differs from known-validation-only calibration')
    if not (len(scores['known_labels']) == len(scores['known_predictions']) == len(scores['known_scores'])):
        raise ValueError('Known score/label length mismatch')
    recomputed = evaluator.open_world_metrics(scores['known_scores'], scores['unknown_scores'], threshold)
    recomputed.update(evaluator.closed_world_metrics(scores['known_labels'], scores['known_predictions']))
    for key, value in recomputed.items():
        if not np.isclose(value, result[key], rtol=1e-9, atol=1e-12):
            raise ValueError('Saved metric differs from saved scores: ' + key)
    for key, array in [('known_test_samples', 'known_scores'), ('unknown_test_samples', 'unknown_scores'),
                       ('known_validation_samples', 'validation_scores')]:
        if result[key] != len(scores[array]):
            raise ValueError('Sample count mismatch: ' + key)
    return {'scenario': scenario, 'fold': fold, 'split_seed': 2022 + fold,
            'metrics': result, 'scores': scores, 'paths': paths, 'marker': marker,
            'identity': identity, 'config': config}


def aggregate(runs):
    rows = []
    for scenario, reference in PAPER.items():
        subset = [r for r in runs if r['scenario'] == scenario]
        seeds = sorted(r['split_seed'] for r in subset)
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seeds in summary')
        full = seeds == list(range(2022, 2027))
        row = {'scenario': scenario, 'n_completed': len(subset), 'status': 'complete' if full else 'partial',
               'paper_table': reference[4], 'paper_accuracy_percent': reference[0],
               'paper_accuracy_sd': reference[1], 'paper_f1_percent': reference[2],
               'paper_f1_sd': reference[3], 'comparability': 'different grouped protocol; paper F1 averaging unconfirmed'}
        for key in METRICS:
            values = [r['metrics'][key] * 100 for r in subset]
            row[key + '_mean_percent'] = float(np.mean(values)) if full else None
            row[key + '_sd_percent'] = float(np.std(values, ddof=1)) if full else None
        rows.append(row)
    return rows


def score_plots(run, destination):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    s, m = run['scores'], run['metrics']
    known, unknown = s['known_scores'], s['unknown_scores']
    t = float(s['threshold'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # Common bins; each class density integrates to one (not class prevalence).
    norm = t > 0
    a, b = (known / t, unknown / t) if norm else (known, unknown)
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=60)
    axes[0].hist(a, bins=bins, density=True, alpha=.55, color='#be3636', label='Known')
    axes[0].hist(b, bins=bins, density=True, alpha=.55, color='#247cc0', label='Unknown')
    axes[0].axvline(1 if norm else t, color='black', linestyle='--', label='Fixed validation threshold')
    axes[0].set(xlabel='Minimum KL / threshold' if norm else 'Minimum KL (threshold <= 0)', ylabel='Density', title='Score distribution (Fig.4-style)')
    axes[0].legend(fontsize=8)
    labels = np.r_[np.zeros(len(known)), np.ones(len(unknown))]
    fpr, tpr, _ = roc_curve(labels, np.r_[known, unknown])
    axes[1].plot(fpr, tpr, label=f"AUROC = {m['auroc']:.4f}")
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].scatter([1 - m['known_test_acceptance']], [m['unknown_test_rejection']], color='red', label='Validation-selected operating point')
    axes[1].set(xlabel='Known false-positive rate', ylabel='Unknown true-positive rate', title='ROC (additional diagnostic)')
    axes[1].legend(fontsize=7)
    matrix = np.array([[m['known_true_negative'], m['known_false_positive']],
                       [m['unknown_false_negative'], m['unknown_true_positive']]])
    axes[2].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(matrix[i, j]), ha='center', va='center',
                         color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Known', 'Unknown'],
                yticklabels=['Known', 'Unknown'], xlabel='Predicted', ylabel='True', title='Binary confusion counts')
    fig.suptitle(f"{run['scenario']} | fold 0, seed 2022 | grouped protocol; no test tuning")
    fig.tight_layout()
    fig.savefig(destination / 'scores_roc_confusion.png', dpi=160)
    plt.close(fig)


def select_examples(known, unknown, max_per_class=40):
    rng = np.random.default_rng(2022)
    chosen = []
    for part, dataset in [('known_test', known), ('unknown_test', unknown)]:
        for label in sorted(np.unique(dataset.targets).tolist()):
            indices = np.flatnonzero(dataset.targets == label)
            indices = np.sort(rng.choice(indices, min(max_per_class, len(indices)), replace=False))
            chosen.extend((part, int(i), int(label)) for i in indices)
    return chosen


def model_plots(run, destination, evaluator, grouped, manifests, device):
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    m = run['metrics']
    _, _, known, unknown, identity = grouped.get_grouped_splits(
        m['dataset'], m['scenario_split'], run['split_seed'], manifests)
    model, checkpoint = evaluator.load_model(str(run['paths']['checkpoint']), device)
    grouped.verify_checkpoint_split(checkpoint, identity)
    if (checkpoint['fold'], checkpoint['split_seed'], checkpoint['dataset'], checkpoint['scenario_split']) != (
            run['fold'], run['split_seed'], m['dataset'], m['scenario_split']):
        raise ValueError('Checkpoint run identity mismatch')
    if checkpoint['code_identity'] != run['config']['code']:
        raise ValueError('Checkpoint source identity mismatch')
    model.eval()
    selected = select_examples(known, unknown)
    datasets = {'known_test': known, 'unknown_test': unknown}
    x = torch.stack([datasets[p][i][0] for p, i, _ in selected])
    labels = np.array([datasets[p][i][1] for p, i, _ in selected])
    latent = []
    with torch.no_grad():
        for batch in x.split(64):
            # Eval latent z is exactly encoder mean; no sampling or augmentation.
            latent.append(model.encoder(batch.to(device))[0].cpu().numpy())
    latent = np.concatenate(latent)
    projection = TSNE(n_components=2, random_state=2022, perplexity=min(30, len(latent)-1),
                      init='pca', learning_rate='auto', n_iter=1000).fit_transform(latent)
    np.savez_compressed(destination / 'latent_projection.npz', latent=latent, projection=projection,
                        labels=labels, subset_indices=np.array([i for _, i, _ in selected]),
                        original_labels=np.array([y for _, _, y in selected]),
                        partitions=np.array([p for p, _, _ in selected]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for y in np.unique(labels):
        pts = projection[labels == y]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=.65,
                   color='red' if y == 999 else plt.cm.turbo(float(y) / max(1, model.n_classes-1)),
                   label='Unknown' if y == 999 else f'Known {y}')
        if y != 999:
            ax.text(*np.median(pts, axis=0), str(y), fontsize=8)
    ax.set(title=f"{run['scenario']} | seed 2022 | t-SNE of encoder mean (Fig.5-style)",
           xlabel='t-SNE 1 (arbitrary)', ylabel='t-SNE 2 (arbitrary)')
    ax.text(.01, -.12, 'Fixed balanced subsample, at most 40/original class. Unknown classes collapsed in red.', transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / 'latent_tsne.png', dpi=160)
    plt.close(fig)
    # Fixed first selected known sample, no search for a visually persuasive example.
    sample = x[:1].to(device).detach().requires_grad_(True)
    logits = -model(sample)[2] / model.temp_inter
    target = int(labels[0])
    probability = torch.softmax(logits, dim=1)[0, target]
    gradient = torch.autograd.grad(probability, sample)[0].abs()[0, 0].detach().cpu().numpy()
    heat = (gradient-gradient.min()) / max(float(np.ptp(gradient)), 1e-30)
    raw = sample[0, 0].detach().cpu().numpy()
    np.savez_compressed(destination / 'pixel_gradient.npz', input=raw, absolute_gradient=gradient, normalized=heat)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(raw, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(heat, cmap='hot', vmin=0, vmax=1)
    axes[2].imshow(raw, cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(heat, cmap='hot', vmin=0, vmax=1, alpha=.6)
    for ax, title in zip(axes, ['Input bytes as pixels', 'Absolute probability gradient', 'Overlay (no TLS field mapping)']):
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle(f"{run['scenario']} | true known class {target}; original ID {selected[0][2]} | Fig.6-style")
    fig.tight_layout()
    fig.savefig(destination / 'pixel_heatmap.png', dpi=160)
    plt.close(fig)
    timing = inference_timing(model, x[:min(50, len(x))], device, m['threshold'])
    timing.update({'scenario': run['scenario'], 'seed': run['split_seed'],
                   'feature_extraction_ms': None, 'total_pcap_to_decision_ms': None,
                   'paper_V100_reference_ms': {'FET': .20, 'CPT': .93, 'total': 1.13},
                   'comparability': 'Not hardware/scope matched to Table VIII; NPZ cannot measure PCAP extraction'})
    save_json(destination / 'inference_timing.json', timing)
    save_json(destination / 'diagnostic_metadata.json', {
        'checkpoint_sha256': run['marker']['checkpoint_sha256'], 'split_identity': identity,
        'seed': 2022, 'max_examples_per_original_class': 40, 'projection': 'sklearn t-SNE, PCA init, learning_rate auto, 1000 iterations',
        'heatmap_selection': selected[0], 'heatmap_target': target, 'known_class_mapping': known.transform_dict,
        'sample_unit': 'row in verified known_test or unknown_test partition; see latent_projection.npz',
        'gradient': 'absolute d softmax(-KL/temp_inter)[true_known_class] / d input, per-image minmax',
        'limitations': 'No semantic TLS field positions inferred; no ablation; t-SNE is an implementation choice'})
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def inference_timing(model, examples, device, threshold):
    import torch
    cached = examples.to(device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def predict(sample):
        # Same full forward as test.py, including decoder, then prototype decision.
        score, prediction = model(sample)[2].min(dim=1)
        return torch.where(score.double() < threshold, prediction, 999)
    durations = []
    with torch.no_grad():
        for i in range(10):
            predict(cached[i % len(cached):i % len(cached)+1])
        sync()
        for sample in cached.split(1):
            sync()
            start = time.perf_counter()
            predict(sample)
            sync()
            durations.append((time.perf_counter()-start)*1000)
    return {'batch_size': 1, 'warmup': 10, 'n_timed': len(durations),
            'mean_ms': float(np.mean(durations)), 'median_ms': float(np.median(durations)),
            'p95_ms': float(np.percentile(durations, 95)), 'measurements_ms': durations,
            'scope': 'cached device input; full eval forward incl decoder + KL argmin + threshold; excludes I/O, transforms, host transfer',
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cpu': platform.processor(), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'python': sys.version, 'platform': platform.platform()}


def render_report(output, summary, status):
    import pandas as pd
    sections = ['<h1>OpenDetect grouped evaluation — tanpa ablasi</h1>',
                '<p><a href="https://doi.org/10.1109/TIFS.2025.3612141">P01 paper</a> · Tables V–VII, Fig.4–6, Table VIII (scope-limited)</p>',
                '<p>' + html.escape(LIMITS).replace('\n', ' ') + '</p>',
                '<h2>Status 40 run</h2>', pd.DataFrame(status).to_html(index=False, escape=True),
                '<h2>Ringkasan (persen; mean/SD kosong bila belum 5/5)</h2>',
                pd.DataFrame(summary)[['scenario', 'n_completed', 'paper_accuracy_percent',
                    'open_accuracy_mean_percent', 'open_accuracy_sd_percent', 'paper_f1_percent',
                    'open_f1_mean_percent', 'open_f1_sd_percent']].to_html(index=False, na_rep='pending', escape=True),
                '<p>CSV lengkap: status.csv, per_run.csv, summary_vs_paper.csv. Paper F1 averaging belum terkonfirmasi; tidak menghitung klaim selisih apples-to-apples.</p>']
    for scenario in PAPER:
        images = sorted((output / scenario).glob('*.png'))
        if images:
            sections.append('<h2>' + scenario + ' — fold 0 / seed 2022</h2>')
        for path in images:
            sections.append(f'<figure><img src="{path.relative_to(output).as_posix()}" style="max-width:100%"><figcaption>{html.escape(path.stem)}</figcaption></figure>')
    timings = []
    for scenario in PAPER:
        path = output/scenario/'inference_timing.json'
        timing = read_json(path) if path.exists() else {}
        timings.append({'scenario': scenario, 'device': timing.get('gpu') or timing.get('device', 'pending'),
                        'inference_mean_ms': timing.get('mean_ms'), 'inference_median_ms': timing.get('median_ms'),
                        'inference_p95_ms': timing.get('p95_ms'), 'n_timed': timing.get('n_timed'),
                        'PCAP_extraction': 'not measured: NPZ input', 'paper_V100_CPT_reference_ms': .93,
                        'comparability': 'hardware/scope not matched; see timing JSON'})
    write_csv(output/'inference_timing_summary.csv', timings, list(timings[0]))
    sections.extend(['<h2>Waktu inferensi (bukan waktu PCAP ke keputusan)</h2>',
                     '<p>Batch 1, input sudah di device, full forward termasuk decoder, 10 warmup; tidak termasuk I/O, transform atau transfer CPU ke GPU. Angka paper hanya referensi V100.</p>',
                     pd.DataFrame(timings).to_html(index=False, na_rep='pending', escape=True)])
    sections.append('<h2>Diagnostik model</h2><p>Lihat diagnostics_status.csv. Ketiadaan checkpoint/data tidak diganti hasil sintetis. Timing dan metadata berada di subfolder skenario.</p>')
    page = '<!doctype html><html lang="id"><meta charset="utf-8"><title>Grouped evaluation</title><style>body{font:15px system-ui;margin:32px;max-width:1500px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccc}h2{margin-top:32px}</style>' + '\n'.join(sections) + '</html>'
    (output / 'REPORT.html').write_text(page, encoding='utf-8')
    (output / 'LIMITATIONS.txt').write_text(LIMITS, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    os.environ['MPLBACKEND'] = 'Agg'
    source, root, output = map(lambda p: Path(p).resolve(), (args.source_repo, args.input_root, args.output))
    if output == root or root in output.parents or output in root.parents or source == output or source in output.parents:
        raise ValueError('Report must be outside training input/source directories')
    output.mkdir(parents=True, exist_ok=False)  # No report overwrite either.
    evaluator, grouped, identity = bind_source(source)
    manifests = source / 'grouped_manifests'
    runs, statuses, diagnostic_status = [], [], []
    for scenario in PAPER:
        for fold in range(5):
            status = {'scenario': scenario, 'fold': fold, 'seed': 2022 + fold}
            try:
                run = verify_run(root/scenario, scenario, fold, evaluator, grouped, identity, manifests)
                status.update(status='verified' if run else 'pending', detail='')
                if run:
                    runs.append(run)
            except Exception as error:
                status.update(status='invalid', detail=str(error))
            statuses.append(status)
            print(scenario, fold, status['status'], status['detail'], flush=True)
    summary = aggregate(runs)
    write_csv(output/'status.csv', statuses, ['scenario', 'fold', 'seed', 'status', 'detail'])
    write_csv(output/'per_run.csv', [{**r['metrics'], 'scenario': r['scenario']} for r in runs],
              ['scenario', 'fold', 'split_seed', 'best_epoch', *METRICS, 'threshold', 'known_test_samples', 'unknown_test_samples'])
    write_csv(output/'summary_vs_paper.csv', summary, list(summary[0]))
    save_json(output/'summary.json', summary)
    save_json(output/'report_provenance.json', {'training_source': identity, 'reporter_sha256': digest(__file__),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'input_root': str(root),
        'verified_artifacts': [{'scenario': r['scenario'], 'fold': r['fold'],
             **{k: r['marker'][k] for k in ('checkpoint_sha256', 'result_sha256', 'scores_sha256')}} for r in runs]})
    for scenario in PAPER:
        run = next((r for r in runs if r['scenario'] == scenario and r['fold'] == 0), None)
        row = {'scenario': scenario, 'seed': 2022, 'status': 'pending fold 0'}
        if run:
            dest = output/scenario
            dest.mkdir()
            score_plots(run, dest)
            row['status'] = 'scores only; diagnostics disabled'
            if args.diagnostics:
                try:
                    import torch
                    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                                          'cpu' if args.device == 'auto' else args.device)
                    model_plots(run, dest, evaluator, grouped, manifests, device)
                    row['status'] = 'complete'
                except Exception as error:
                    row['status'] = 'diagnostics failed: ' + str(error)
        diagnostic_status.append(row)
        print('Diagnostics:', row, flush=True)
    write_csv(output/'diagnostics_status.csv', diagnostic_status, ['scenario', 'seed', 'status'])
    render_report(output, summary, statuses)
    print('REPORT:', output/'REPORT.html', flush=True)
    # Publish useful partial report but never silently label invalid artifacts successful.
    if any(s['status'] == 'invalid' for s in statuses):
        raise SystemExit('Report written, but invalid artifacts require inspection; see status.csv')
    if any('failed:' in s['status'] for s in diagnostic_status):
        raise SystemExit('Tables written, but diagnostics incomplete; see diagnostics_status.csv')


if __name__ == '__main__':
    main()
