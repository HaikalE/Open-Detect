"""Generate a standalone, source-pinned CPU audit notebook."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = '1eRt_MeJkoVvFCMTfuHssELQ2RgqQK0Ii'


def cell(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


def build():
    sources = {name: (ROOT / name).read_text(encoding='utf-8') for name in
               ('data/Preprocessing/utils.py', 'pilot/audit.py', 'pilot/__init__.py')}
    digests = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    cells = [cell('markdown', '''# 01 — Audit data temporal fusion (CPU)
**Fase P0, belum training fusion.** Tidak perlu GPU; jalankan berurutan.
Membaca folder sumber Drive A berdasarkan ID, bukan isi akun login secara umum.
Login harus mempunyai izin ke sumber itu. Tidak mengubah dataset/checkpoint lama.
Unduhan maksimum: NPZ 400 MiB + enam PCAP masing-masing 8 MiB.
Pemeriksaan PCAP bersifat sampel; kecocokan hash bukan bukti pairing lengkap.
Upload laporan hanya di cell terakhir. `training_ready=false` adalah gerbang ilmiah,
bukan tanda hasil replikasi gagal. Rujukan dan tahap lanjut ada di README/PLAN folder ini.
'''), cell('code', '''%pip -q install scapy==2.5.0
import sys, json, hashlib, io, uuid
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from google.colab import auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
auth.authenticate_user()
credentials, _ = google.auth.default()
drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
WORK = Path('/content/temporal_pilot_p0') / RUN_ID
WORK.mkdir(parents=True, exist_ok=False)
print('CPU audit run:', RUN_ID)
'''), cell('code', f'''# Snapshot source; no moving GitHub branch dependency.
SOURCES = {sources!r}
SOURCE_SHA256 = {digests!r}
for name, source in SOURCES.items():
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256[name]
    path = WORK / 'src' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
sys.path.insert(0, str(WORK / 'src'))
from pilot.audit import audit_paths
PILOT_FOLDER = {DESTINATION!r}
DATA_FOLDER = '1vCeFiN3ng82p9_p0tfnvwzjCe_pdeMie'
RAW_FOLDER = '1-fYdLsJfBVV5rTx922y25aRQA0k_fSB0'
for folder in (PILOT_FOLDER, DATA_FOLDER, RAW_FOLDER):
    meta = drive.files().get(fileId=folder, fields='id,name,mimeType', supportsAllDrives=True).execute()
    assert meta['mimeType'] == 'application/vnd.google-apps.folder'
    print('Akses OK:', meta['name'], meta['id'])
'''), cell('code', '''# Limited Drive enumeration; skipped/failed entries remain visible in report.
FOLDER_MIME = 'application/vnd.google-apps.folder'
def children(folder, page_token=None):
    return drive.files().list(q=f"'{folder}' in parents and trashed=false",
        fields='nextPageToken,files(id,name,mimeType,size,md5Checksum)',
        pageSize=100, pageToken=page_token, orderBy='name',
        supportsAllDrives=True, includeItemsFromAllDrives=True).execute()

def inventory(root, max_entries=3000, max_folders=100):
    queue, rows, errors = deque([(root, '')]), [], []
    visited = set()
    partial = False
    while queue:
        if len(visited) >= max_folders or len(rows) >= max_entries:
            partial = True
            break
        folder, prefix = queue.popleft()
        if folder in visited:
            continue
        visited.add(folder)
        token = None
        try:
            while True:
                response = children(folder, token)
                for item in response.get('files', []):
                    if len(rows) >= max_entries:
                        partial = True
                        break
                    item['relative_path'] = prefix + item['name']
                    rows.append(item)
                    if item['mimeType'] == FOLDER_MIME:
                        queue.append((item['id'], item['relative_path'] + '/'))
                token = response.get('nextPageToken')
                if not token or partial:
                    break
        except Exception as exc:
            errors.append({'folder_id': folder, 'error': str(exc)})
    return {'items': rows, 'partial': partial or bool(errors), 'errors': errors,
            'folders_scanned': len(visited)}

raw_inventory = inventory(RAW_FOLDER)
print('Raw entries:', len(raw_inventory['items']), 'partial:', raw_inventory['partial'])
print('Inventory errors:', raw_inventory['errors'])
print('Small PCAP candidates:', sum(
    x['name'].lower().endswith(('.pcap', '.pcapng')) and int(x.get('size', 0)) > 0
    and int(x['size']) <= 8 * 1024**2 for x in raw_inventory['items']))
'''), cell('code', '''# Download the six exact baseline NPZs from their verified folder.
EXPECTED = {'USTC_1c_train.npz', 'USTC_1c_test.npz', 'mal_32_1c_train.npz',
            'mal_32_1c_test.npz', 'combined_train_data.npz', 'combined_test_data.npz'}
data_inventory = inventory(DATA_FOLDER, max_entries=100, max_folders=1)
items = [x for x in data_inventory['items'] if x['name'] in EXPECTED]
assert len(items) == 6 and {x['name'] for x in items} == EXPECTED, 'Missing/duplicate NPZ: inspect folder, do not substitute silently.'
assert all(x.get('size') for x in items), 'Unknown file sizes'
assert sum(int(x['size']) for x in items) <= 400 * 1024**2, 'NPZ download budget exceeded'

def download(item, directory, limit):
    size = int(item.get('size', 0))
    if not 0 < size <= limit:
        raise ValueError('Missing size or download limit exceeded: ' + item['name'])
    # Drive ID as prefix prevents filename collisions and path traversal.
    path = directory / (item['id'] + '_' + Path(item['name']).name)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = drive.files().get_media(fileId=item['id'], supportsAllDrives=True)
    digest = hashlib.md5()
    with path.open('wb') as handle:
        loader = MediaIoBaseDownload(handle, request, chunksize=1024**2)
        done = False
        while not done:
            _, done = loader.next_chunk(num_retries=3)
            if handle.tell() > size or handle.tell() > limit:
                raise ValueError('File grew during transfer; stop and rerun audit')
    assert path.stat().st_size == size, 'Incomplete download'
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    if item.get('md5Checksum'):
        assert digest.hexdigest() == item['md5Checksum'], 'Drive checksum mismatch'
    return path

npz_paths, download_errors = [], []
for item in sorted(items, key=lambda x: x['name']):
    try:
        npz_paths.append(download(item, WORK / 'npz', 400 * 1024**2))
        print('NPZ OK:', item['name'])
    except Exception as exc:
        download_errors.append({'file': item['name'], 'error': str(exc)})
'''), cell('code', '''# Round-robin by source dataset; not a representative/random statistical sample.
groups = {}
for item in raw_inventory['items']:
    if item['name'].lower().endswith(('.pcap', '.pcapng')) and 0 < int(item.get('size', 0)) <= 8 * 1024**2:
        source = item['relative_path'].split('/')[0]
        groups.setdefault(source, []).append(item)
queues = [deque(sorted(items, key=lambda x: (int(x['size']), x['relative_path'])))
          for _, items in sorted(groups.items())]
selected = []
while queues and len(selected) < 6:
    for queue in queues:
        if queue and len(selected) < 6:
            selected.append(queue.popleft())
    queues = [queue for queue in queues if queue]
pcap_paths = []
for item in selected:
    try:
        pcap_paths.append(download(item, WORK / 'pcap', 8 * 1024**2))
    except Exception as exc:
        download_errors.append({'file': item['name'], 'error': str(exc)})
report = audit_paths(npz_paths, pcap_paths, packet_limit=10000)
report.update(run_id=RUN_ID, source_sha256=SOURCE_SHA256,
    source_folder_ids={'dataset': DATA_FOLDER, 'raw': RAW_FOLDER},
    raw_inventory=raw_inventory, dataset_inventory=data_inventory,
    selected_pcaps=selected, download_errors=download_errors,
    missing_dataset_count=6-len(npz_paths))
REPORT_PATH = WORK / ('pilot_audit_' + RUN_ID + '.json')
REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
for item in report['datasets']:
    print('NPZ:', item['file'], '| rows:', item['samples'], '| keys:', item['keys'])
for item in report['pcaps']:
    print('PCAP:', item['file'], '|', item['status'])
print('Errors:', report['errors'] + download_errors)
print('Missing datasets:', report['missing_dataset_count'])
print('Full pairing verified:', report['training_ready'])
if not selected:
    print('No small PCAP selected in bounded scan. Check inventory/archives/permissions; this does NOT prove raw data absent.')
print('Local report:', REPORT_PATH)
print('NEXT: verify complete pairing / sessionization; no GPU training yet.')
'''), cell('markdown', '''## Simpan laporan ke folder pilot Drive A
Cell berikut hanya mengunggah laporan JSON baru, bukan checkpoint/PCAP/dataset.
Run ulang cell ini menggunakan file laporan yang sama tidak membuat duplikat.
Jika izin menulis ditolak, laporan lokal masih tersedia dan bisa diunduh manual.
'''), cell('code', '''name = REPORT_PATH.name
existing = drive.files().list(q=f"'{PILOT_FOLDER}' in parents and trashed=false and name='{name}'",
    fields='files(id,name,webViewLink)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get('files', [])
if existing:
    print('Laporan run ini sudah ada; tidak ditimpa:', existing)
else:
    uploaded = drive.files().create(body={'name': name, 'parents': [PILOT_FOLDER]},
        media_body=MediaFileUpload(str(REPORT_PATH), mimetype='application/json', resumable=True),
        fields='id,name,webViewLink', supportsAllDrives=True).execute()
    print('Tersimpan di folder PILOT_TEMPORAL_FUSION:', uploaded)
''')]
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': '01_AUDIT_DATA_CPU.ipynb'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}}, 'cells': cells}
    for i, item in enumerate(cells):
        item['id'] = f'pilot-p0-{i:02}'
    target = ROOT / 'pilot/01_AUDIT_DATA_CPU.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
