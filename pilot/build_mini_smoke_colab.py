"""Generate a small, reproducible CPU Colab for the bounded mini-fusion smoke."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = '1fa2730'
OUTPUT_NAME = '03_MINI_FUSION_SMOKE_CPU_V2.ipynb'


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [
        cell('markdown', '''# 03 — Mini fusion smoke (CPU; bukan evaluasi tesis)

Jalankan empat cell kode berurutan di runtime **CPU**. Notebook ini membaca
cohort mini berpasangan dari folder Drive P2, memverifikasi checksum, lalu
menjalankan satu langkah optimisasi B0 image-only, E1 panjang+arah, dan E2
+IAT. **One-batch loss bukan metrik perbandingan akurasi**. Tidak melatih model
sampai konvergen dan tidak mengunggah/menimpa artefak Drive.
'''),
        cell('code', '''from google.colab import auth
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path
import hashlib, io, json, subprocess, sys, tempfile

creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
drive = build('drive', 'v3', credentials=creds, cache_discovery=False)
WORK = Path(tempfile.mkdtemp(prefix='mini_fusion_cpu_'))
COHORT = WORK / 'cohort'
COHORT.mkdir()
print('Workspace:', WORK)
'''),
        cell('code', '''FILES = {
    'summary.json': '1BtZfv8N5lEXPuoXTYtzhdqGN07s5a5_m',
    'manifest.jsonl': '1VuPqylxOP5diJlPepYDE4-V90aI7PbfO',
    'mini_paired_cohort_NOT_THESIS_EVAL.npz': '1cQswB7PMYsBjyWLUb01bQU_wmSwiIJtI',
}
P2_FOLDER_ID = '1vQyL66nIjIazvU3jtiCXaX8pdNEa34pJ'
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
print('Checksum OK; 564 candidates, not a thesis evaluation.')
'''),
        cell('code', f'''REPO = WORK / 'Open-Detect'
subprocess.run(['git', 'clone', '--filter=blob:none', '--single-branch',
                '--branch', 'codex/temporal-fusion',
                'https://github.com/HaikalE/Open-Detect.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', '{PINNED_COMMIT}'], check=True)
head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--short=7', 'HEAD'], text=True).strip()
assert head == '{PINNED_COMMIT}', head
print('Python:', sys.version)
for pattern in ('test_temporal_fusion.py', 'test_mini_cohort.py'):
    cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v']
    result = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    print('TEST:', pattern, 'exit:', result.returncode)
    print(result.stdout)
    print(result.stderr)
    if result.returncode:
        raise RuntimeError('Test gagal; lihat error asli tepat di atas, bukan hanya CalledProcessError')
OUT = WORK / 'mini_fusion_smoke_cpu.json'
cmd = [sys.executable, '-m', 'pilot.smoke_mini_fusion', '--cohort-dir', str(COHORT),
       '--output', str(OUT)]
result = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
print(result.stdout)
print(result.stderr)
if result.returncode:
    raise RuntimeError('Smoke gagal; lihat error asli tepat di atas')
print('Smoke selesai; bukan training atau skor tesis:', OUT)
'''),
        cell('code', '''result = json.loads(OUT.read_text())
assert result['status'] == 'ENGINEERING_ONLY_NOT_EVALUATION'
print('Arms:', list(result['arms']))
print('Checksum cohort:', result['cohort_npz_sha256'])
print('Jangan bandingkan one-batch loss sebagai performa model.')
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
