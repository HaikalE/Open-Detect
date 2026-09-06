"""Generate small, commit-pinned Colab launchers. Does not start cloud training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parent
FOLDER = 'OpenDetect_GROUPED_2026-09-06'
SCENARIOS = [
    'A-1', 'A-2', 'A-3', 'B-1', 'B-2', 'B-3', 'C-1', 'C-2']


def cell(kind, source, number):
    source = textwrap.dedent(source).strip() + '\n'
    if kind == 'code':
        ast.parse(source)
    result = {'cell_type': kind, 'id': f'grouped-{number}', 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


SETUP = '''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
PROJECT = Path('/content/drive/MyDrive/THESIS IMPLEMENTASI')
PACKAGE = PROJECT / FOLDER
PACKAGE.mkdir(parents=True, exist_ok=True)
OUTPUT = PACKAGE / 'outputs' / SCENARIO
# Deliberately do not create files inside OUTPUT: runner verifies ownership first.
REPO = Path('/content') / ('OpenDetect_grouped_' + SCENARIO.replace('-', '_'))
if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                    'codex/grouped-split-colab', 'https://github.com/HaikalE/Open-Detect.git', str(REPO)], check=True)
assert (REPO / '.git').is_dir(), 'Not a Git checkout; nothing overwritten'
current = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
if current != COMMIT:
    assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip(), 'Dirty checkout; preserved'
    subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', COMMIT], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', COMMIT], check=True)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
def verify_source():
    for name, expected in SOURCE_HASHES.items():
        assert sha256(REPO/name) == expected, 'Source mismatch: ' + name
verify_source()
print('Pinned source verified:', COMMIT)
'''

ENVIRONMENT = '''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
UV = [sys.executable, '-m', 'uv']
VENV = Path('/content/opendetect_grouped_python310')
if not (VENV/'bin/python').exists():
    subprocess.run(UV + ['venv', '--python', '3.10', str(VENV)], check=True)
PYTHON = str(VENV/'bin/python')
subprocess.run(UV + ['pip', 'install', '--python', PYTHON, '-r', str(REPO/'requirements.txt')], check=True)
subprocess.run([PYTHON, '-c', "import torch; assert torch.cuda.is_available(), 'Pilih runtime GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"], check=True)
qa_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
subprocess.run([PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=REPO, env=qa_env, check=True)
(PACKAGE / ('environment_' + SCENARIO + '.txt')).write_text(
    subprocess.check_output([PYTHON, '-V'], text=True) +
    subprocess.check_output(UV + ['pip', 'freeze', '--python', PYTHON], text=True) +
    subprocess.check_output(['nvidia-smi'], text=True), encoding='utf-8')
'''

DATA = '''
report = json.loads((REPO/'grouped_manifests/REPORT.json').read_text())
record = next(r for r in report['runs'] if r['scenario'] == SCENARIO and r['seed'] == 2022)
for name in record['source_row_order']:
    source = PROJECT / 'data/dataset' / name
    assert source.is_file(), 'NPZ Drive belum ditemukan: ' + str(source)
    expected = report['source_sha256'][name]
    assert sha256(source) == expected, 'NPZ tidak cocok dengan indeks: ' + name
    destination = REPO/'data/dataset'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix('.npz.copying')
        shutil.copyfile(source, temporary)
        assert sha256(temporary) == expected
        os.replace(temporary, destination)
    assert sha256(destination) == expected, 'NPZ lokal berbeda; file dipertahankan: ' + name
print('NPZ asli cocok; 5 indeks grouped tersedia untuk', SCENARIO)
'''

RUN = '''
verify_source()
env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
command = [PYTHON, str(REPO/'run_grouped.py'), '--scenario', SCENARIO,
           '--output', str(OUTPUT), '--session_hours', str(SESSION_HOURS), '--num_workers', '2', '--gpu', '0']
print('Output terpisah:', OUTPUT)
print('COMMAND:', subprocess.list2cmdline(command))
# Runner writes train/test logs and checkpoints to Drive every completed epoch.
process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
except BaseException:
    if process.poll() is None:
        process.terminate()
        process.wait()
    raise
status = json.loads((OUTPUT/'SESSION_STATUS.json').read_text())
print(json.dumps(status, indent=2))
if status['status'] == 'paused':
    print('Pause terencana. Run all lagi untuk melanjutkan epoch/fold yang belum selesai.')
else:
    print('Lima pengulangan selesai. Buka summary.json dan per_run.csv di folder output.')
'''


def build(commit, output):
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit], text=True).splitlines()
    names = [n for n in names if n.endswith('.py') or n == 'requirements.txt' or n.startswith('grouped_manifests/')]
    hashes = {n: hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{n}'])).hexdigest() for n in names}
    output.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        intro = f'''
        # OpenDetect {scenario} — GROUPED, lima pengulangan

        **Eksperimen baru; bukan melanjutkan hasil split baris lama.** A-1/A-2/A-3 memakai USTC,
        B-1/B-2/B-3 memakai Malicious TLS; C-1/C-2 memakai gabungan keduanya.
        Tidak ada C-3 pada konfigurasi delapan skenario ini.

        Indeks SHA-256 gambar identik sudah dipasang ke train, validasi, known-test dan unknown-test.
        NPZ asli tidak diganti. Lima seed 2022–2026 adalah repeated grouped holdout 80:10:10
        berdasarkan grup, bukan lima test-fold saling lepas; rasio baris dapat sedikit berbeda.
        Unknown tidak dipakai training atau kalibrasi threshold; unknown_unused tidak digunakan.

        Basis kode: replication-fixes (decoder versi 2 dan koreksi evaluasi), bukan branch lama paper-alignment.
        Commit dikunci: `{commit}`. Model/loss/augmentasi tidak diubah lagi oleh integrasi grouped ini.
        Pembagian ini adalah kontrol overlap gambar, bukan bukti PCAP/flow bebas leakage atau jaminan akurasi naik.
        `preview` di nama protokol adalah ID paket indeks yang dipertahankan demi keterlacakan;
        pada notebook ini indeks sudah benar-benar dipakai trainer/evaluator.

        **Cara menjalankan:** Runtime → Change runtime type → GPU, kemudian Run all; izinkan mount Drive.
        Maksimum kerja terencana 3 jam per pemanggilan, berhenti setelah epoch lengkap. 100 epoch per seed tetap.
        Jika status `paused`, Run all lagi; run selesai diverifikasi lalu dilewati.
        Jika runtime putus, lanjut dari checkpoint grouped terakhir yang masih valid di Drive.
        Jika tidak pernah ada checkpoint, mulai epoch 1. Checkpoint lama/beda data/kode/indeks ditolak.
        Dua slot menyimpan model, optimizer, scheduler, RNG, best checkpoint. Jangan jalankan skenario yang sama
        pada dua runtime. Drive/FUSE tidak menjamin sinkronisasi instan; hash mendeteksi file rusak,
        tetapi kedua slot dapat hilang bila cloud belum menyimpan. Lintas GPU tidak dijamin bitwise sama.
        Server/Linux CLI tidak wajib. Notebook ini memakai Drive mount seperti workflow Colab sebelumnya.

        **Hasil:** `{FOLDER}/outputs/{scenario}/`: checkpoint di `save_model/`, resume di `resume_state/`,
        metrik/skor per seed di `results/`, log di `logs/`, progres `SESSION_STATUS.json`, `per_run.csv`,
        `summary.json` setelah kelima seed selesai. Slot recovery dipertahankan; sediakan beberapa GB per skenario.
        '''
        sections = [('markdown', intro), ('code', f'SCENARIO = {scenario!r}\nCOMMIT = {commit!r}\nFOLDER = {FOLDER!r}\nSESSION_HOURS = 3.0\nSOURCE_HASHES = {hashes!r}'),
                    ('markdown', '## 1. Mount Drive dan verifikasi kode'), ('code', SETUP),
                    ('markdown', '## 2. Python 3.10, dependensi, dan tes CPU'), ('code', ENVIRONMENT),
                    ('markdown', '## 3. Verifikasi dan salin NPZ tanpa perubahan'), ('code', DATA),
                    ('markdown', '## 4. Training/evaluasi lima seed — resume otomatis'), ('code', RUN)]
        notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
            'accelerator': 'GPU', 'colab': {'name': f'OpenDetect_{scenario}_GROUPED.ipynb'},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'source_commit': commit},
            'cells': [cell(k, s, i) for i, (k, s) in enumerate(sections)]}
        (output/f'OpenDetect_{scenario}_GROUPED.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps({'commit': commit, 'sha256': hashes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.commit, args.output)
