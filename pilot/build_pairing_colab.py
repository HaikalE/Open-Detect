"""Build P1 standalone Colab from the tested P0 Drive helpers and new extractor."""
import ast
import hashlib
import json
from pathlib import Path

from build_colab import cell

ROOT = Path(__file__).resolve().parents[1]


def build():
    original = json.loads((ROOT/'pilot/01_AUDIT_DATA_CPU.ipynb').read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type'] == 'code')
    parsed = ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith('%')))
    clean = '\n'.join(line for line in code.splitlines() if not line.startswith('%'))
    helpers = '\n\n'.join(ast.get_source_segment(clean, node) for node in parsed.body
                           if isinstance(node, ast.FunctionDef) and node.name == 'download')
    sources = {name: (ROOT/name).read_text(encoding='utf-8') for name in
               ('pilot/__init__.py', 'pilot/pairing.py', 'data/Preprocessing/utils.py')}
    digests = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}
    setup = ''.join(original['cells'][1]['source']).replace('scapy==2.5.0', 'scapy==2.5.0 rarfile==4.2').replace('temporal_pilot_p0', 'temporal_pilot_p1').replace('CPU audit run:', 'CPU pairing pilot:')
    cells = [cell('markdown', '''# 02 — Sessionization & pairing pilot (CPU)
Jalankan berurutan, runtime CPU. **Tidak perlu mengulang notebook 01.**
Sumber: laporan P0 yang sudah berhasil, keenam NPZ, empat PCAP USTC kecil,
dan arsip Malicious_TLS (hanya daftar isi, tidak diekstrak/dijalankan).
Unduhan dibatasi 400 MiB NPZ + 32 MiB PCAP + 20 MiB RAR.
Membentuk citra dan [payload length, arah, IAT] dari 8 paket sesi yang SAMA,
lalu mencari kecocokan hash terhadap NPZ lama. Aturan sesi kandidat: biflow,
idle >60 detik, SYN baru (sequence berbeda), dan sesudah RST. Ini bukan klaim
bahwa pembagian sesi asli telah direproduksi. Hanya first-8, bukan sliding windows.
Output masih kandidat, **bukan dataset siap training**, tanpa split atau label tebakan.
Jika tidak cocok, cek preprocessing/sessionization/provenance; jangan paksakan mapping.
'''), cell('code', setup), cell('code', f'''SOURCES = {sources!r}
SOURCE_SHA256 = {digests!r}
for name, source in SOURCES.items():
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256[name]
    path = WORK / 'src' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
sys.path.insert(0, str(WORK / 'src'))
from pilot.pairing import run_pairing, sha_file
PILOT_FOLDER = '1eRt_MeJkoVvFCMTfuHssELQ2RgqQK0Ii'
P0_REPORT_ID = '1I2Tzy_vaKYm9-8Kh1j3IMm5DC4kKppJJ'
{helpers}

meta = drive.files().get(fileId=P0_REPORT_ID, fields='id,name,size,md5Checksum', supportsAllDrives=True).execute()
report_path = download(meta, WORK/'source_report', 2*1024**2)
p0 = json.loads(report_path.read_text())
assert p0['schema'] == 'temporal-pilot-audit-v1'
assert p0['missing_dataset_count'] == 0 and not p0['errors'] and not p0['download_errors']
print('P0 source run:', p0['run_id'])
'''), cell('code', '''# Exact files discovered by P0; stale/replaced contents fail MD5 rather than silently change inputs.
expected = {'USTC_1c_train.npz', 'USTC_1c_test.npz', 'mal_32_1c_train.npz',
            'mal_32_1c_test.npz', 'combined_train_data.npz', 'combined_test_data.npz'}
datasets = [x for x in p0['dataset_inventory']['items'] if x['name'] in expected]
assert len(datasets) == 6 and {x['name'] for x in datasets} == expected
assert sum(int(x['size']) for x in datasets) <= 400*1024**2
npz_paths = []
for item in datasets:
    npz_paths.append(download(item, WORK/'npz', 400*1024**2))
    print('OK:', item['name'])
raw = p0['raw_inventory']['items']
capture_items = sorted([x for x in raw if x['relative_path'].startswith('USTC-TFC2016/')
    and x['name'].endswith('.pcap') and 0 < int(x.get('size',0)) <= 8*1024**2],
    key=lambda x: (int(x['size']), x['relative_path']))[:4]
assert capture_items, 'No small USTC PCAP in P0 report'
captures = []
for item in capture_items:
    path = download(item, WORK/'pcap', 8*1024**2)
    captures.append((path, Path(item['name']).stem))
    print('PCAP:', item['relative_path'])
'''), cell('code', '''# Read archive directory only. Never extract/execute archive contents.
import rarfile
archive_items = [x for x in raw if x['relative_path'] == 'Malicious_TLS/malicious_TLS.rar']
archive_report = {'status': 'NOT_FOUND_IN_P0', 'entries': [], 'raw_pcap_present': None}
if len(archive_items) == 1:
    archive_path = download(archive_items[0], WORK/'archive', 20*1024**2)
    try:
        with rarfile.RarFile(archive_path) as archive:
            entries = [{'name': i.filename, 'bytes': i.file_size, 'is_dir': i.isdir()}
                       for i in archive.infolist()]
        archive_report = {'status': 'LISTED_ONLY', 'archive_sha256': sha_file(archive_path),
            'entries': entries, 'raw_pcap_present': any(
                not i['is_dir'] and i['name'].lower().endswith(('.pcap','.pcapng')) for i in entries)}
    except Exception as exc:
        archive_report = {'status': 'UNREADABLE', 'error': str(exc), 'raw_pcap_present': None}
print(json.dumps(archive_report, indent=2))
if archive_report.get('raw_pcap_present') is False:
    print('B/C: no directly listed PCAP in this archive. Need original timestamped PCAP; CSV row order is not pairing.')
'''), cell('code', '''# No training, threshold fitting, or split assignment in this stage.
OUT = WORK/'outputs'
summary = run_pairing(npz_paths, captures, OUT)
summary.update(run_id=RUN_ID, source_sha256=SOURCE_SHA256, source_p0_sha256=sha_file(report_path),
               source_p0_run=p0['run_id'], source_pcaps=capture_items,
               malicious_tls_archive=archive_report)
(OUT/'pairing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Candidates:', summary['candidates'])
print('Hash matching:', summary['match_counts'])
print('Temporal coverage:', json.dumps(summary['temporal_coverage'], indent=2))
if not any(x['matched_multi_packet_candidates'] for x in summary['temporal_coverage'].values()):
    print('STOP before training: no uniquely matched multi-packet sequence in this pilot. One packet has no inter-packet timing.')
print('Capture scan:', [(s['class'], s['complete'], s['sessions']) for s in summary['captures']])
print('Errors:', summary['errors'])
print('Training ready:', summary['training_ready'])
print('First-8 unmatched does NOT prove all original images unrecoverable; session/window rules still need verification.')
print('Outputs:', [(p.name, p.stat().st_size) for p in OUT.iterdir()])
'''), cell('markdown', '''## Upload hasil — cell terakhir terpisah
Membuat subfolder baru `P1_<run-id>` di folder pilot A; berisi ringkasan,
manifest kandidat, dan NPZ citra+sequence+mask. Tidak mengunggah PCAP atau checkpoint lama.
Run ulang cell ini melengkapi file yang belum ada; tidak menimpa file yang sudah ada.
'''), cell('code', '''folder_name = 'P1_' + RUN_ID
folders = drive.files().list(q=f"'{PILOT_FOLDER}' in parents and trashed=false and name='{folder_name}'",
    fields='files(id,mimeType)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get('files', [])
assert len(folders) <= 1, 'Ambiguous destination'
if folders:
    assert folders[0]['mimeType'] == 'application/vnd.google-apps.folder'
    destination = folders[0]['id']
else:
    destination = drive.files().create(body={'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder', 'parents': [PILOT_FOLDER]},
        fields='id', supportsAllDrives=True).execute()['id']
for path in sorted(OUT.iterdir()):
    existing = drive.files().list(q=f"'{destination}' in parents and trashed=false and name='{path.name}'",
        fields='files(id,md5Checksum)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get('files', [])
    if existing:
        assert len(existing) == 1 and existing[0].get('md5Checksum') == hashlib.md5(path.read_bytes()).hexdigest(), 'Existing file differs; not overwritten'
        print('Already verified:', path.name)
        continue
    result = drive.files().create(body={'name': path.name, 'parents': [destination]},
        media_body=MediaFileUpload(str(path), mimetype='application/octet-stream', resumable=True),
        fields='id,md5Checksum', supportsAllDrives=True).execute()
    assert result.get('md5Checksum') == hashlib.md5(path.read_bytes()).hexdigest()
    print('Uploaded + verified:', path.name)
print('Hasil P1:', 'https://drive.google.com/drive/folders/' + destination)
''')]
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': '02_PAIRING_PILOT_CPU.ipynb'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}}, 'cells': cells}
    for i, item in enumerate(cells):
        item['id'] = f'pilot-p1-{i:02}'
    target = ROOT/'pilot/02_PAIRING_PILOT_CPU.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
