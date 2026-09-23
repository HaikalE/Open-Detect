"""Generate the short, bounded B0/E1/E2 open-set training pilot Colab."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = '43584f4'
OUTPUT_NAME = '04_MINI_OPENSET_TRAIN_CPU_V2.ipynb'


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [
        cell('markdown', '''# 04 — Mini open-set training pilot

Jalankan di runtime CPU atau GPU. B0/E1/E2 akan dilatih pada cohort mini yang
sama, maksimum 8 epoch dengan early stopping. Threshold unknown hanya memakai
validasi known; Tinba tetap di test. Hasil ini untuk cek kelayakan pada lima
kelas, **bukan evaluasi skenario tesis**. Model terbaik hanya ditahan di RAM;
tidak ada checkpoint yang disimpan. Cell terakhir mengunggah JSON metrik kecil.
'''),
        cell('code', '''from google.colab import auth
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from pathlib import Path
import hashlib, io, json, subprocess, sys, tempfile, uuid

creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
drive = build('drive', 'v3', credentials=creds, cache_discovery=False)
WORK = Path(tempfile.mkdtemp(prefix='mini_openset_train_'))
COHORT = WORK / 'cohort'
COHORT.mkdir()
P2_FOLDER_ID = '1vQyL66nIjIazvU3jtiCXaX8pdNEa34pJ'
print('Workspace:', WORK)
'''),
        cell('code', '''FILES = {
    'summary.json': '1BtZfv8N5lEXPuoXTYtzhdqGN07s5a5_m',
    'manifest.jsonl': '1VuPqylxOP5diJlPepYDE4-V90aI7PbfO',
    'mini_paired_cohort_NOT_THESIS_EVAL.npz': '1cQswB7PMYsBjyWLUb01bQU_wmSwiIJtI',
}
for name, file_id in FILES.items():
    meta = drive.files().get(fileId=file_id, fields='id,name,size,parents', supportsAllDrives=True).execute()
    assert meta['name'] == name and P2_FOLDER_ID in meta['parents']
    assert int(meta['size']) < 1024 * 1024
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    with (COHORT / name).open('wb') as output:
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(name, meta['size'], 'bytes')

summary = json.loads((COHORT/'summary.json').read_text())
def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
assert sha256(COHORT/'mini_paired_cohort_NOT_THESIS_EVAL.npz') == summary['outputs']['npz_sha256']
assert sha256(COHORT/'manifest.jsonl') == summary['outputs']['manifest_sha256']
assert summary['status'] == 'FEASIBILITY_ONLY_NOT_THESIS_EVALUATION'
print('Input checksum verification passed.')
'''),
        cell('code', f'''REPO = WORK / 'Open-Detect'
subprocess.run(['git', 'clone', '--filter=blob:none', '--single-branch',
                '--branch', 'codex/temporal-fusion',
                'https://github.com/HaikalE/Open-Detect.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', '{PINNED_COMMIT}'], check=True)
head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short=7', 'HEAD'], text=True).strip()
assert head == '{PINNED_COMMIT}', head
RESULT = WORK / 'mini_open_set_training.json'
cmd = [sys.executable, '-m', 'pilot.train_mini_open_set', '--cohort-dir', str(COHORT),
       '--output', str(RESULT), '--epochs', '8', '--patience', '2', '--batch-size', '128',
       '--seed', '2022', '--learning-rate', '0.001', '--lamda', '0.005', '--device', 'auto']
subprocess.run(cmd, cwd=REPO, check=True)
report = json.loads(RESULT.read_text())
print('Pilot completed:', report['device'], report['samples'])
'''),
        cell('code', '''# Upload one compact metrics file, never weights/checkpoints.
suffix = uuid.uuid4().hex[:8]
drive_name = f"mini_open_set_training_{suffix}.json"
metadata = {'name': drive_name, 'parents': [P2_FOLDER_ID]}
media = MediaFileUpload(str(RESULT), mimetype='application/json', resumable=False)
uploaded = drive.files().create(body=metadata, media_body=media, fields='id,name,webViewLink',
                                supportsAllDrives=True).execute()
print('Saved metrics:', uploaded['name'], uploaded.get('webViewLink'))
for arm, result in report['arms'].items():
    print(arm, {key: result[key] for key in ('best_epoch', 'known_test_accuracy',
          'known_test_weighted_f1', 'unknown_auroc', 'unknown_test_rejection',
          'known_test_acceptance', 'elapsed_seconds')})
print('Do not present this bounded five-class pilot as thesis evaluation.')
'''),
    ]
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'colab': {'name': OUTPUT_NAME}},
            'cells': cells}


if __name__ == '__main__':
    output = ROOT / 'pilot' / OUTPUT_NAME
    with output.open('x', encoding='utf-8') as handle:
        json.dump(build(), handle, indent=1, ensure_ascii=False)
    print(output)
