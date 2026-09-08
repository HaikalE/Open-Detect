"""Generate the existing recovery notebook's CPU-only cloud workflow replacement."""
import ast
import json
import sys
from pathlib import Path
from build_grouped_colabs import cell
from build_storage_notebooks import AUTH, helper_source

ROOT=Path(__file__).resolve().parent

INTRO='''
# OpenDetect Recovery — cloud-workflow-v4

**Notebook yang sama untuk A/B. CPU cukup, tanpa upload/download JSON manual.**
Ini pemulihan data, bukan cara melewati batas GPU Colab. Tidak memulai training.
Autentikasi Drive menentukan A/B; hak Editor tidak memindahkan kepemilikan atau kuota.

| Akun | ACTION | Apa yang terjadi / berikutnya |
|---|---|---|
| B | PREPARE_B | Preview file + identitas root. Pilih ID sumber, konfirmasi producer berhenti. Manifest disimpan di Drive, sumber dibagikan ke A. |
| A | BACKUP_A | Membaca manifest cloud, copy ke THESIS IMPLEMENTASI/RECOVERY_B_… milik A, verifikasi tiap file, simpan receipt cloud. |
| A | PLAN_A | Otomatis kenali skenario/fold/seed dari JSON + SHA256 PT, bandingkan konfigurasi dan epoch A. |
| A | INSTALL_A | Hanya satu skenario/fold terverifikasi. KEEP_A bila A sudah sama/lebih baru. File lama dipertahankan; tidak menghapus B. |
| B | PREVIEW_B | Periksa ulang backup A; tampilkan ID sumber dan token daftar untuk cleanup. Tidak menghapus. |
| B | TRASH_B | Konfirmasi + token daftar. Hanya file sumber receipt masuk Sampah, tetap memakai kuota B. |
| B | DELETE_B | Eksekusi terpisah, konfirmasi permanen + token. Wajib sudah di Sampah, backup diverifikasi ulang. Tidak mengosongkan seluruh Sampah. |

**Kasus lama C-1:** RECEIPT_ID sudah menunjuk receipt backup yang ada. Mulai A → PLAN_A.
Epoch A 52 pernah lolos, backup B 50; hasil pemeriksaan terkini yang menentukan, bukan angka ini.
Backup arsip bukan folder training. Tujuan lanjut dicetak sebagai
`outputs/<skenario>/resume_state/<dataset>_split_<split>_fold_<fold>` beserta link notebook training.
Fold yang sudah selesai tidak ditimpa. Trainer yang memverifikasi dan melewati fold selesai.

Untuk recovery baru: kosongkan RECEIPT_ID. Jika hanya satu job/receipt ditemukan,
dipilih otomatis; jika lebih dari satu, salin satu **ID** dari daftar (bukan upload JSON).
Root hanya kandidat: ID harus dipilih setelah preview, tidak otomatis semua file bernama last.pt.
JSON/log/hasil boleh diarsipkan; pemasangan otomatis hanya pasangan PT + metadata yang valid.
Config/folder canonical belum ada, riwayat epoch sama tetapi hash beda, format lama,
pasangan hilang atau identitas ambigu → STOP, arsip tetap aman; jangan Run all training dari nol.

## Sebelum mutasi

Hentikan penulisan di **A dan B**, satu sesi recovery saja. Jangan terminate runtime
sebelum checkpoint cloud tersedia. Data cache VM yang sudah hilang tidak bisa dipulihkan.
Saat ganti akun, gunakan sesi recovery bersih dan periksa email terotorisasi; jangan remount
runtime training aktif. Laporan/receipt tersimpan di Drive, akun tetap harus memberi izin.

PLAN/INSTALL hanya verifikasi cloud, tidak membuktikan model/optimizer/RNG dapat dimuat.
**Jangan hapus permanen B sampai resume nyata pada notebook training sudah berhasil.**
DELETE_B meminta pengakuan eksplisit pengguna; ini bukan tes otomatis keberhasilan training.
Jangan hapus backup A. File BEFORE_/STAGED_ dari pemasangan dipertahankan untuk pemulihan;
Drive tidak punya transaksi beberapa file sekaligus. Jika proses putus, PLAN_A dulu, jangan
menjalankan beberapa INSTALL bersamaan. Batas kuota/GPU Colab tetap berlaku.
'''

CONFIG='''
# @title Pilih tahap (default read-only)
ACTION = "PLAN_A" # @param ["PLAN_A", "PREPARE_B", "BACKUP_A", "INSTALL_A", "PREVIEW_B", "TRASH_B", "DELETE_B"]
JOB_FOLDER_ID = "" # @param {type:"string"}
RECEIPT_ID = "1MSwbn8u1NYfFi47uiYAcMt_pvDGkbK_x" # @param {type:"string"}
SOURCE_IDS = "" # @param {type:"string"}
# SOURCE_IDS: ID sumber dipisah koma, hanya setelah preview PREPARE_B.
SCENARIO = "C-1" # @param ["A-1", "A-2", "A-3", "B-1", "B-2", "B-3", "C-1", "C-2"]
FOLD = 2 # @param {type:"integer"}
STOPPED = "" # @param {type:"string"}
# Isi setelah SEMUA producer A/B berhenti: SEMUA RUN SUMBER SUDAH BERHENTI
CONFIRMATION = "" # @param {type:"string"}
PREVIEW_TOKEN = "" # @param {type:"string"}
RESUME_CONFIRMED = "" # @param {type:"string"}
'''

RUN='''
active=account(API)['user']['emailAddress'].lower()
a=owner(meta(API,PROJECT_ID))
print('ANDA ADALAH:', 'A (origin)' if active==a else 'B (sumber)', active)
if ACTION=='PREPARE_B':
    prepare_b(API,STOPPED,[s.strip() for s in SOURCE_IDS.split(',') if s.strip()])
elif ACTION=='BACKUP_A':
    manifest=load_manifest(API,JOB_FOLDER_ID.strip())
    receipt=backup_A(API,manifest,STOPPED)
    receipt_id=publish(API,receipt['backup_folder'],'RECEIPT',receipt)
    print('BACKUP SELESAI. Isi RECEIPT_ID =',receipt_id,'; lanjut PLAN_A, bukan langsung delete.')
elif ACTION in ('PLAN_A','INSTALL_A'):
    receipt=load_receipt(API,RECEIPT_ID.strip())
    plan=recovery_plan(API,receipt)
    print_plan(plan)
    if ACTION=='INSTALL_A':
        install_one(API,receipt,SCENARIO,FOLD,STOPPED,CONFIRMATION)
        plan=recovery_plan(API,receipt)
        publish(API,receipt['backup_folder'],'POST_INSTALL',plan)
    else:
        print('READ ONLY: tidak ada file diubah. INSTALL confirmation: PASANG CHECKPOINT TERVERIFIKASI KE A')
elif ACTION in ('PREVIEW_B','TRASH_B','DELETE_B'):
    receipt=load_receipt(API,RECEIPT_ID.strip())
    if active!=receipt['owner_B'] or active==a:
        raise ValueError('Harus login akun B pemilik sumber receipt')
    selected=[r['id'] for r in receipt['files']]
    token=hashlib.sha256(json.dumps([(r['id'],r['md5Checksum'],r['backup_id'])
        for r in receipt['files']],sort_keys=True).encode()).hexdigest()
    print('DAFTAR CLEANUP:')
    for r in receipt['files']: print(r['id'],r['name'],r['size'],'-> backup',r['backup_id'])
    print('Total:',len(selected),'files;',sum(int(r['size']) for r in receipt['files']),'bytes')
    print('PREVIEW_TOKEN =',token)
    print('TRASH: PINDAHKAN FILE TERVERIFIKASI KE SAMPAH')
    print('DELETE: HAPUS PERMANEN FILE TERVERIFIKASI')
    if ACTION!='PREVIEW_B':
        if PREVIEW_TOKEN!=token: raise ValueError('Lakukan PREVIEW_B dahulu dan salin token daftar persis')
        if ACTION=='DELETE_B' and RESUME_CONFIRMED!='SAYA SUDAH MELIHAT RESUME TRAINING BERHASIL DI A':
            raise ValueError('Tahan delete: tunggu resume model/RNG nyata berhasil di A. Lalu isi RESUME_CONFIRMED sesuai petunjuk.')
        selected=pending_cleanup(API,receipt,permanent=ACTION=='DELETE_B')
        if not selected: raise ValueError('Tidak ada sumber tersedia untuk diproses. Backup dipertahankan; jangan simpulkan semua sudah dihapus.')
        audit=cleanup_B(API,receipt,selected,STOPPED,permanent=ACTION=='DELETE_B',confirmation=CONFIRMATION)
        # B cannot write into A read-only archive. Keep audit in B-owned handoff folder.
        folder=API.files().create(body={'name':'OpenDetect_CLEANUP_AUDIT_'+uuid.uuid4().hex,
            'mimeType':'application/vnd.google-apps.folder','parents':[PROJECT_ID]},fields=FIELDS).execute()
        publish(API,folder['id'],'CLEANUP',audit)
        print('Selesai. Trash masih memakai kuota; permanent tidak dapat dipulihkan dari B. Backup A tidak dihapus.')
else:
    raise ValueError('Unknown action')
'''


def build(path):
    workflow=(ROOT/'cloud_recovery_workflow.py').read_text(encoding='utf-8')
    tree=ast.parse(workflow)
    # Embed source, not a mutable network import or external dependency on our repo.
    workflow='\n\n'.join(ast.get_source_segment(workflow,n) for n in tree.body
        if not isinstance(n,ast.ImportFrom) or n.module!='drive_recovery')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'},'recovery_version':'cloud-workflow-v4',
        'colab':{'name':'OpenDetect_RECOVERY_B_TO_A.ipynb'}},'cells':[
            cell('markdown',INTRO,0),cell('code',CONFIG,1),
            cell('markdown','## Autentikasi Drive — CPU cukup\nPeriksa email A/B sebelum tahap berikutnya.',2),
            cell('code',helper_source()+'\n'+workflow+'\n'+AUTH,3),
            cell('markdown','## Jalankan tahap terpilih\nDefault PLAN_A hanya membaca.',4),
            cell('code',RUN,5)]}
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
    return nb

if __name__=='__main__': build(sys.argv[1])
