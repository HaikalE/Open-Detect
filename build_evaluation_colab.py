"""Generate an evaluation-only Colab pinned to independent reporter/training commits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parent
TRAIN_COMMIT = 'c729403f7fd4d1207522e3ca793cafb5f8cb8eb0'


def cell(kind, source, index):
    source = textwrap.dedent(source).strip() + '\n'
    if kind == 'code':
        ast.parse(source)
    item = {'cell_type': kind, 'metadata': {}, 'id': f'evaluation-{index}', 'source': source.splitlines(True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


SETUP = '''
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, shutil, subprocess, uuid
from google.colab import drive
drive.mount('/content/drive')
PROJECT = Path('/content/drive/MyDrive/THESIS IMPLEMENTASI')
PACKAGE = PROJECT / 'OpenDetect_GROUPED_2026-09-06'
INPUT = PACKAGE / 'outputs'
DATA = PROJECT / 'data/dataset'
REPORT = PACKAGE / 'evaluation_reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
# These are NEW local checkouts, never the running training notebook's directory.
def checkout(branch, commit, destination):
    if not destination.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', branch,
                        'https://github.com/HaikalE/Open-Detect.git', str(destination)], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        assert not subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip(), 'Dirty checkout preserved'
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', commit], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    return destination
SOURCE = checkout('codex/grouped-split-colab', TRAIN_COMMIT, Path('/content/OpenDetect_eval_training_source'))
REPORTER = checkout('codex/grouped-evaluation', EVAL_COMMIT, Path('/content/OpenDetect_eval_reporter'))
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
assert sha256(REPORTER/'report_grouped.py') == REPORTER_SHA256, 'Reporter changed'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
UV = [sys.executable, '-m', 'uv']
VENV = Path('/content/opendetect_eval_python310')
if not (VENV/'bin/python').exists():
    subprocess.run(UV + ['venv', '--python', '3.10', str(VENV)], check=True)
PYTHON = str(VENV/'bin/python')
subprocess.run(UV + ['pip', 'install', '--python', PYTHON, '-r', str(SOURCE/'requirements.txt')], check=True)
subprocess.run([PYTHON, '-c', "import torch; print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"], env=ENV, check=True)
print('Input (read only):', INPUT)
print('New report:', REPORT)
'''

DATA = '''
# NPZ only needed for feature plots/heatmaps/timing; score tables use saved results.
if RUN_MODEL_DIAGNOSTICS:
    manifest = json.loads((SOURCE/'grouped_manifests/REPORT.json').read_text())
    needed = set()
    for rec in manifest['runs']:
        if rec['seed'] != 2022:
            continue
        scenario = rec['scenario']
        # Never start training or wait for a running fold here.
        marker_dir = INPUT/scenario/'state'
        if marker_dir.exists() and any(marker_dir.glob('*_fold_0.completed.json')):
            needed.update(rec['source_row_order'])
    for name in sorted(needed):
        src, dst = DATA/name, SOURCE/'data/dataset'/name
        expected = manifest['source_sha256'][name]
        if not src.exists():
            print('MISSING NPZ (tables still run; diagnostics will report missing data):', src)
            continue
        assert sha256(src) == expected, 'NPZ Drive hash mismatch: ' + name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tmp = dst.with_suffix('.npz.copying')
            shutil.copyfile(src, tmp)
            assert sha256(tmp) == expected
            os.replace(tmp, dst)
        assert sha256(dst) == expected, 'Local NPZ differs; preserved: ' + name
print('No Drive training outputs were modified.')
'''

RUN = '''
assert sha256(REPORTER/'report_grouped.py') == REPORTER_SHA256
command = [PYTHON, str(REPORTER/'report_grouped.py'), '--source-repo', str(SOURCE),
           '--input-root', str(INPUT), '--output', str(REPORT), '--device', 'auto']
if RUN_MODEL_DIAGNOSTICS:
    command += ['--diagnostics']
process = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='', flush=True)
REPORT_EXIT_CODE = process.wait()
print('Exit code:', REPORT_EXIT_CODE)
if REPORT_EXIT_CODE:
    print('Ada hasil invalid/diagnostik belum lengkap. Periksa status.csv dan diagnostics_status.csv; jangan anggap semua berhasil.')
print('Laporan:', REPORT)
'''

DISPLAY = '''
from IPython.display import display, HTML, Image
import pandas as pd, zipfile
if (REPORT/'summary_vs_paper.csv').exists():
    display(pd.read_csv(REPORT/'summary_vs_paper.csv')[['scenario','n_completed','status','paper_accuracy_percent','open_accuracy_mean_percent','open_accuracy_sd_percent','paper_f1_percent','open_f1_mean_percent','open_f1_sd_percent']])
    display(pd.read_csv(REPORT/'diagnostics_status.csv'))
    for scenario in ['A-1','A-2','A-3','B-1','B-2','B-3','C-1','C-2']:
        for figure in sorted((REPORT/scenario).glob('*.png')):
            print(scenario, figure.name)
            display(Image(filename=str(figure), width=1100))
    archive = REPORT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in REPORT.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(REPORT))
    print('ZIP tersimpan di Drive:', archive)
    print('Unduh dan ekstrak ZIP, kemudian buka REPORT.html. Run all berikutnya membuat laporan baru.')
else:
    print('Belum ada laporan; periksa error sel sebelumnya. Tidak ada training yang dijalankan.')
'''


def build(commit, destination):
    blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':report_grouped.py'])
    sources = [
        ('markdown', '''
        # OpenDetect — Evaluasi grouped tanpa ablasi

        Buka notebook ini **setelah setidaknya satu fold grouped selesai**. Tidak melatih ulang,
        menghentikan runtime lain, mengubah threshold, atau menulis ke folder output training.
        Hasil lama non-grouped tidak dicampur. Semua 8 skenario × 5 seed diperiksa;
        run belum selesai menjadi pending. Mean/SD hanya muncul setelah 5/5 seed lengkap.

        CPU cukup untuk tabel/grafik skor. GPU opsional mempercepat visualisasi model;
        jangan hentikan training hanya untuk menyediakan GPU evaluasi.
        Set `RUN_MODEL_DIAGNOSTICS=False` bila hanya ingin tabel dan grafik skor.

        Referensi P01: Tabel V–VII (angka utama), Fig.4 (skor), Fig.5 (fitur),
        Fig.6 (gradien pixel), Tabel VIII (timing dengan batasan). **Tidak ada ablasi**.
        Closed metrics adalah kelas known per skenario, bukan replikasi penuh Tabel IV.
        F1 di kode = binary unknown-positive; averaging F1 paper belum terkonfirmasi.
        Grouped split dan jumlah data berbeda dari paper; angka referensi bukan target tuning.
        PCAP extraction time tidak dapat diukur dari NPZ. t-SNE pilihan implementasi ini;
        heatmap tidak otomatis mengidentifikasi field TLS.'''),
        ('code', f"TRAIN_COMMIT = {TRAIN_COMMIT!r}\nEVAL_COMMIT = {commit!r}\nREPORTER_SHA256 = {hashlib.sha256(blob).hexdigest()!r}\nRUN_MODEL_DIAGNOSTICS = True"),
        ('markdown', '## 1. Mount Drive dan siapkan evaluator terpisah\n\nTidak memerlukan perubahan notebook training.'),
        ('code', SETUP),
        ('markdown', '## 2. Siapkan NPZ untuk diagnostik model\n\nHanya dataset skenario dengan fold 0 selesai. File Drive dibaca, tidak diubah.'),
        ('code', DATA),
        ('markdown', '## 3. Evaluasi artefak dan buat laporan\n\nHash checkpoint/result/scores dan metrik diperiksa. Diagnostik contoh ditetapkan fold 0/seed 2022, bukan seed terbaik.'),
        ('code', RUN),
        ('markdown', '## 4. Lihat dan unduh laporan\n\nOutput terpisah: `OpenDetect_GROUPED_2026-09-06/evaluation_reports/<waktu-unik>/`.\n\nLihat status sebelum mengutip angka. Tidak ada hasil eksperimen yang diisi dengan data sintetis.'),
        ('code', DISPLAY),
    ]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'colab': {'name': 'OpenDetect_GROUPED_EVALUATION_NO_ABLATION.ipynb'}},
        'cells': [cell(kind, text, i) for i, (kind, text) in enumerate(sources)]}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/'OpenDetect_GROUPED_EVALUATION_NO_ABLATION.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(path)
    return notebook


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    build(a.commit, a.output)
