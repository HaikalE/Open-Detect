"""One recovery notebook + in-place replacements for the eight training launchers."""
import argparse
import ast
import json
from pathlib import Path
import textwrap

import build_grouped_colabs as original

ROOT = Path(__file__).resolve().parent
TRAIN_COMMIT = 'c729403f7fd4d1207522e3ca793cafb5f8cb8eb0'


def helper_source(guard_only=False):
    source = (ROOT/'drive_recovery.py').read_text(encoding='utf-8')
    if not guard_only:
        return source
    names = {'file_id','meta','owner','account','list_files','require_A','verify_mount'}
    tree = ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source, node) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)) or
        isinstance(node, ast.FunctionDef) and node.name in names) + '\n'


AUTH = '''
from google.colab import auth
import google.auth
from googleapiclient.discovery import build
auth.authenticate_user()
credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
API = build('drive','v3',credentials=credentials,cache_discovery=False)
print('Drive API account:', account(API)['user']['emailAddress'])
print('Origin folder owner A:', owner(meta(API, PROJECT_ID)))
# Tokens/passwords are never printed, exported, or saved into the notebook.
'''


def build(output):
    output = Path(output)
    original.build(TRAIN_COMMIT, output)
    guard = helper_source(True)
    for scenario in original.SCENARIOS:
        path = output/f'OpenDetect_{scenario}_GROUPED.ipynb'
        nb = json.loads(path.read_text(encoding='utf-8'))
        old = ''.join(nb['cells'][3]['source'])
        old = old.replace("drive.mount('/content/drive')", '')
        old = old.replace("PACKAGE.mkdir(parents=True, exist_ok=True)",
            "verify_mount(API, PROJECT)\nPACKAGE.mkdir(parents=True, exist_ok=True)")
        setup = guard + '\n' + textwrap.dedent(AUTH) + "\nrequire_A(API, 6_000_000_000)\nfrom google.colab import drive\ndrive.mount('/content/drive')\n" + old
        nb['cells'][3] = original.cell('code', setup, 3)
        intro = ''.join(nb['cells'][0]['source'])
        intro = intro.replace('Server/Linux CLI tidak wajib. Notebook ini memakai Drive mount seperti workflow Colab sebelumnya.',
            'Server/Linux CLI tidak wajib. Mount Drive harus terverifikasi sebagai penulis akun A.')
        intro += '''\n## Pengaman akun dan penyimpanan — baca dulu

**Akun penulis Drive wajib pemilik folder origin A, bukan sekadar Editor B.**
Autentikasi Drive API dan mount harus cocok. Bila akun B terdeteksi, notebook
BERHENTI sebelum training. Tidak membuat folder pengganti dengan nama yang sama.
Tes file kecil memastikan folder ID origin dan pemilik file yang benar; minimal
6 GB ruang A dicek sebelum mulai. Itu cadangan awal, bukan jaminan sinkronisasi.

Jika Colab tidak mengizinkan otorisasi A dari sesi B, gunakan sesi akun A. Notebook
ini tidak menjanjikan compute B + storage A otomatis dan tidak mengakali pembatasan akun.
Jangan mengganti autentikasi/remount ketika training sedang berjalan.

Untuk output lama milik B: gunakan notebook RECOVERY terlebih dahulu. Backup
disimpan terpisah, **belum otomatis dimasukkan sebagai state resume canonical**.
Jangan Run all pada output kosong jika bermaksud melanjutkan state B yang belum
dipulihkan. Jangan hapus runtime lama sebelum checkpoint cloud/backup terverifikasi.
File salinan notebook di akun B tidak ikut berubah saat file origin ini diperbarui.
'''
        nb['cells'][0] = original.cell('markdown', intro, 0)
        run = ''.join(nb['cells'][-1]['source'])
        nb['cells'][-1] = original.cell('code', 'require_A(API, 6_000_000_000)\n' + run, 9)
        nb['metadata']['storage_guard'] = 'origin-owner-and-cloud-probe-v1'
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    sections = [
        ('markdown', '''
        # OpenDetect — Pemulihan B → A dan pembersihan B

        **Satu notebook, dipakai dalam tiga tahap. CPU cukup; tidak training.**
        **Versi root-metadata-v2:** checkpoint, JSON pendamping, log, dan hasil root
        masuk laporan kandidat. Nama yang sama tetap disimpan sebagai ID terpisah.
        A = pemilik folder THESIS IMPLEMENTASI origin. B = pemilik checkpoint yang
        membuat kuota penuh. Folder bersama tidak memindahkan kepemilikan file.

        1. **INVENTORY_B**: login B, tampilkan file B dalam outputs eksperimen dan
           kandidat checkpoint + metadata/log root. Laporan kandidat JSON otomatis diunduh.
           Pilih ID orphan yang benar; bukan otomatis semua last.pt.
           Setelah producer berhenti, bagikan file terpilih ke A dan unduh manifest JSON.
        2. **BACKUP_A**: buka notebook yang sama dengan otorisasi A, upload manifest.
           Copy server-side ke folder RECOVERY_B di THESIS IMPLEMENTASI A. Pemilik,
           ukuran, MD5 Drive, dan versi sumber diperiksa. Unduh receipt JSON.
        3. **CLEANUP_B**: kembali otorisasi B, upload receipt. Preview dahulu; pilih ID.
           Pindahkan ke Sampah setelah backup dicek ulang. Sampah masih memakai kuota.
           Hapus permanen hanya dalam eksekusi terpisah, konfirmasi eksplisit, setelah trash.

        **Jangan terminate runtime sumber yang masih menyimpan checkpoint belum tersinkron.**
        Alat ini hanya melihat file yang sudah tersedia di cloud; bukan penyelamat cache VM
        yang belum terunggah. Hentikan producer sebelum snapshot/cleanup; jangan mengubah
        akun mount pada runtime training aktif. Bila tidak yakin, lakukan inventaris saja.

        Backup tidak menimpa outputs A. Nama disertai ID untuk mempertahankan versi
        orphan yang bernama sama. Manifest menyimpan path asal. Pemulihan ke canonical
        resume perlu pengecekan slot+JSON yang sesuai; tidak otomatis memilih berdasarkan nama.
        Tidak membuka pickle checkpoint, tidak menghapus folder, tidak mengosongkan seluruh Sampah.
        Jika ganti akun, gunakan runtime notebook pemulihan baru/terputus (bukan runtime training).
        Pastikan baris **Drive API account** sesuai tahap; kesalahan akun ditolak.
        Laporan kandidat bukan manifest backup. Jika file terpilih masih 0, jangan
        lanjut BACKUP_A: isi EXTRA_ORPHAN_IDS dari laporan, termasuk JSON pendamping.
        File .writing ditandai parsial; jangan dianggap checkpoint valid untuk resume.
        '''),
        ('code', "PHASE = 'INVENTORY_B' # INVENTORY_B / BACKUP_A / CLEANUP_B\nEXTRA_ORPHAN_IDS = [] # ID persis dari tabel kandidat root yang memang milik eksperimen ini\nSTOPPED = '' # Setelah producer berhenti: SEMUA RUN SUMBER SUDAH BERHENTI\nSHARE_AND_EXPORT = False # True setelah meninjau inventaris; memberi A akses baca hanya file terpilih\nCLEANUP_ACTION = 'PREVIEW' # PREVIEW / TRASH / DELETE_PERMANENT\nSELECTED_IDS = [] # ID sumber persis dari receipt, bukan ID backup\nCONFIRMATION = '' # Diisi hanya saat cleanup; lihat petunjuk output"),
        ('markdown', '## 1. Autentikasi dan periksa identitas\n\nPilih B untuk inventaris/pembersihan, A untuk backup. Jangan membagikan token atau password.'),
        ('code', helper_source() + '\n' + textwrap.dedent(AUTH)),
        ('markdown', '## 2. Jalankan tahap yang dipilih\n\nDefault hanya inventaris; tidak ada penghapusan. Ubah konfigurasi setelah meninjau daftar.'),
        ('code', '''
        from google.colab import files
        from googleapiclient.http import MediaIoBaseUpload
        import io
        def incoming():
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Upload tepat satu manifest/receipt JSON')
            return json.loads(next(iter(uploaded.values())))
        def export_json(name, value):
            path = Path('/content')/name
            path.write_text(json.dumps(value, indent=2), encoding='utf-8')
            files.download(str(path))
        if PHASE == 'INVENTORY_B':
            manifest = inventory(API, EXTRA_ORPHAN_IDS)
            candidates = root_candidates(API)
            print('KANDIDAT ROOT — belum otomatis dipilih, termasuk JSON dan log:')
            for r in candidates:
                print(r['id'], r['name'], r['candidate_kind'], r.get('size'), r.get('modifiedTime'))
            print('JUMLAH KANDIDAT ROOT:', len(candidates))
            print('TOTAL BYTE KANDIDAT:', sum(int(r['size']) for r in candidates))
            export_json('OpenDetect_B_root_candidates.json', {
                'format':'opendetect-root-candidates-v2', 'owner_B':manifest['owner_B'],
                'notice':'CANDIDATES ONLY, not a backup receipt or deletion authorization',
                'files':candidates})
            print('FILE TERPILIH UNTUK BACKUP:', len(manifest['files']))
            for r in manifest['files']:
                print(r['id'], '/'.join(r['relative_parts']), r['size'])
            print('Total bytes:', sum(int(r['size']) for r in manifest['files']))
            if not manifest['files']:
                print('Belum dipilih. Kirim laporan kandidat untuk diperiksa atau isi EXTRA_ORPHAN_IDS dengan ID yang benar. Jangan pindahkan file manual.')
            if SHARE_AND_EXPORT:
                manifest = share_inventory(API, manifest, STOPPED)
                export_json('OpenDetect_B_inventory.json', manifest)
                print('Berikutnya: otorisasi A, PHASE BACKUP_A, upload manifest tersebut.')
        elif PHASE == 'BACKUP_A':
            manifest = incoming()
            receipt = backup_A(API, manifest, STOPPED)
            payload = json.dumps(receipt, indent=2).encode()
            saved = API.files().create(body={'name':'RECEIPT_'+uuid.uuid4().hex+'.json',
                'parents':[receipt['backup_folder']]}, media_body=MediaIoBaseUpload(io.BytesIO(payload),
                mimetype='application/json'), fields='id').execute()
            receipt['cloud_receipt_id'] = saved['id']
            export_json('OpenDetect_A_backup_receipt.json', receipt)
            print('Backup folder A ID:', receipt['backup_folder'])
            print('Berikutnya: B, PHASE CLEANUP_B, upload receipt. Jangan hapus tanpa receipt valid.')
        elif PHASE == 'CLEANUP_B':
            receipt = incoming()
            for r in receipt['files']:
                print('SOURCE:', r['id'], r['name'], r['size'], 'BACKUP:', r['backup_id'])
            if CLEANUP_ACTION == 'PREVIEW':
                print('Tidak ada yang dihapus. Isi SELECTED_IDS setelah meninjau.')
                print('TRASH confirmation: PINDAHKAN FILE TERVERIFIKASI KE SAMPAH')
                print('DELETE_PERMANENT confirmation: HAPUS PERMANEN FILE TERVERIFIKASI')
            elif CLEANUP_ACTION in ('TRASH','DELETE_PERMANENT'):
                audit = cleanup_B(API, receipt, SELECTED_IDS, STOPPED,
                    permanent=CLEANUP_ACTION=='DELETE_PERMANENT', confirmation=CONFIRMATION)
                export_json('OpenDetect_cleanup_log.json', audit)
                print('Sampah belum membebaskan kuota. Penghapusan permanen tidak dapat dipulihkan dari B; backup A tetap ada.')
            else:
                raise ValueError('Unknown cleanup action')
        else:
            raise ValueError('Unknown phase')
        '''),
    ]
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'},'colab':{'name':'OpenDetect_RECOVERY_B_TO_A.ipynb'}},
        'cells':[original.cell(k,s,i) for i,(k,s) in enumerate(sections)]}
    (output/'OpenDetect_RECOVERY_B_TO_A.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    build(p.parse_args().output)
