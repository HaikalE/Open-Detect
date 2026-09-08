"""Update the same eight scenario notebooks to use a scoped owner-A relay."""
import json
from pathlib import Path
import sys
import build_grouped_colabs as old

ROOT=Path(__file__).resolve().parent
COMMIT='c729403f7fd4d1207522e3ca793cafb5f8cb8eb0'


def build(output):
    output=Path(output);old.build(COMMIT,output)
    transport={n:(ROOT/n).read_text(encoding='utf-8') for n in ('relay_client.py','relay_entry.py','drive_reader.py','restore_plan.py')}
    for scenario in old.SCENARIOS:
        p=output/f'OpenDetect_{scenario}_GROUPED.ipynb';nb=json.loads(p.read_text())
        nb['metadata']['storage_transport']='owner-relay-v3-selective-restore'
        intro=''.join(nb['cells'][0]['source'])
        intro=intro[:intro.index('**Cara menjalankan:**')]+f'''
## Penyimpanan multi-worker — owner-relay-v3-selective-restore

**Aktivasi layanan A wajib; tanpa URL/key valid notebook BERHENTI sebelum training.**
B-mn, B-mc, A, dan pekerja lain memakai notebook skenario masing-masing. Semua
file cloud dibuat atas otorisasi A; tidak memakai drive.mount atau Drive pekerja.
Download menggunakan izin baca akun pekerja langsung ke Drive A (bukan Apps Script).
Saat diminta login baca Drive, pilih akun pekerja B, BUKAN kredensial A.
A harus membagikan folder dataset dan outputs/{scenario} kepada B minimal Viewer
dengan izin download. Key relay bukan pengganti izin baca Drive. Tidak ada perubahan
sharing otomatis. File tersalin hanya ke disk VM /content, bukan kuota Drive B.
Model, loss, data, seed, dan file kode ilmiah tetap pinned commit di atas.
Transport tambahan membungkus batas epoch tanpa mengubah file repo tersebut.

1. A menyiapkan layanan dengan `relay/SETUP.md` di branch codex/grouped-evaluation.
2. Pada Colab Secrets (ikon kunci), isi `OPENDETECT_RELAY_URL` (URL /exec) dan
   `OPENDETECT_WORKER_KEY` dari A. Beri notebook akses. Jangan tulis key di sel,
   output, screenshot, chat atau GitHub. Key berbeda untuk tiap pekerja dan dapat dicabut.
3. Jalankan GPU → Run all. Key harus ditugaskan ke **{scenario}**. Autentikasi baca
   dilakukan sekali saat setup, sebelum training; error izin berhenti dengan jelas.
4. Skenario yang sama hanya satu lease aktif. Skenario berbeda boleh paralel;
   tetap tunduk kuota GPU masing-masing akun dan kuota layanan Google.

**Checkpoint aman di cloud ditandai `CLOUD COMMITTED TO A`, bukan sekadar
`CHECKPOINT COMMITTED` lokal.** Upload berjalan sinkron setelah epoch selesai;
gagal upload → training dihentikan, snapshot cloud sebelumnya tetap tersedia.
Runtime mati sebelum CLOUD COMMITTED → ulang dari snapshot cloud terakhir.
Log/hasil disertakan pada sinkronisasi, bukan disiarkan setiap baris.

Hasil A berada di `outputs/{scenario}/_relay/objects` dan `snapshots` (immutable).
Manifest snapshot menunjuk path logis `save_model/`, `results/`, `resume_state/`.
Notebook terbaru mengembalikan snapshot itu ke disk VM lokal lalu trainer
memverifikasi konfigurasi dan melanjutkan epoch/fold. A melanjutkan pekerjaan B
dengan membuka notebook yang sama dan key A yang diberi akses {scenario}.
Pemulihan selektif: model terbaik, hasil, skor, dan marker fold selesai tetap
diunduh dan diverifikasi. Folder resume_state fold itu dilewati HANYA setelah
verifikasi sukses. Semua slot fold belum selesai tetap dipulihkan agar fallback
bawaan trainer terjaga. File yang dilewati tetap tersimpan/dirujuk di cloud A;
tidak dihapus saat sinkronisasi berikutnya. Penanda selesai saja tidak cukup.
**Jangan memakai notebook mount lama atau recovery-v4 untuk memilih state relay terbaru.**
File canonical lama dipertahankan sebagai bootstrap, tidak diperbarui oleh relay.

Tidak ada penghapusan otomatis: snapshot lama memakai kuota A. Layanan berhenti
jika sisa ruang tidak cukup. Pemeliharaan arsip perlu dilakukan A setelah backup.
Lease tidak diambil alih berdasarkan waktu. Bila VM mati, A memastikan producer
lama berhenti lalu menjalankan releaseLostWorker sebelum pekerja pengganti mulai.
Semua runtime mount lama harus dihentikan sebelum initializeOwner; kunci relay
tidak dapat mengendalikan notebook lama yang sudah berjalan di luar layanan.

Kode/sinkronisasi sudah dites lokal; otorisasi Google, kirim file besar dan
resume GPU lintas akun tetap harus diuji setelah deployment. Gunakan hanya
pekerja tepercaya: checkpoint PyTorch berisi pickle yang dimuat trainer.
'''
        nb['cells'][0]=old.cell('markdown',intro,0)
        original=''.join(nb['cells'][3]['source'])
        original=original[original.index('REPO = Path'):]
        setup=f'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
from google.colab import userdata
# Access errors are intentionally fatal; no fallback to worker Drive.
RELAY_URL = userdata.get('OPENDETECT_RELAY_URL')
WORKER_KEY = userdata.get('OPENDETECT_WORKER_KEY')
TRANSPORT = {transport!r}
TRANSPORT_DIR = Path('/content/opendetect_transport')
TRANSPORT_DIR.mkdir(exist_ok=True)
for name, source in TRANSPORT.items():
    (TRANSPORT_DIR/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(TRANSPORT_DIR))
from relay_client import Relay
from drive_reader import colab_reader
RELAY = Relay(RELAY_URL,WORKER_KEY,SCENARIO)
print('RELAY READY:', RELAY.call('hello'))
RELAY.reader = colab_reader()
print('HYBRID READ READY: direct Drive reads; output writes remain owner A relay')
PROJECT = Path('/content/opendetect_worker')
PACKAGE = PROJECT / FOLDER
PACKAGE.mkdir(parents=True,exist_ok=True)
# New work directory avoids overwriting unsynchronized files from a prior failed run.
OUTPUT = PACKAGE / ('work_' + RELAY.session) / SCENARIO
'''+original
        nb['cells'][2]=old.cell('markdown','## 1. Koneksi layanan A dan kode pinned (tanpa mount Drive)',2)
        nb['cells'][3]=old.cell('code',setup,3)
        env=''.join(nb['cells'][5]['source'])
        env=env.replace("subprocess.run([PYTHON, '-c',", "subprocess.run(UV + ['pip','install','--python',PYTHON,'requests==2.32.5'],check=True)\nsubprocess.run([PYTHON, '-c',",1)
        nb['cells'][5]=old.cell('code',env,5)
        data='''
report=json.loads((REPO/'grouped_manifests/REPORT.json').read_text())
record=next(r for r in report['runs'] if r['scenario']==SCENARIO and r['seed']==2022)
cloud=RELAY.call('data')
for name in record['source_row_order']:
    expected=report['source_sha256'][name]
    assert cloud[name]['sha256']==expected, 'A dataset does not match pinned grouped indices'
    RELAY.download(cloud[name],REPO/'data/dataset'/name,dataset=name)
    assert sha256(REPO/'data/dataset'/name)==expected
print('Data terverifikasi, tidak disalin ke Drive pekerja')
'''
        nb['cells'][7]=old.cell('code',data,7)
        run='''
verify_source()
RELAY.claim() # refuse another active worker; same scenario cannot race
try:
    RELAY.restore(OUTPUT)
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(TRANSPORT_DIR),OPENDETECT_RELAY_URL=RELAY_URL,
        OPENDETECT_WORKER_KEY=WORKER_KEY,OPENDETECT_SCENARIO=SCENARIO,
        OPENDETECT_RELAY_SESSION=RELAY.session,OPENDETECT_LOCAL_OUTPUT=str(OUTPUT))
    command=[PYTHON,str(TRANSPORT_DIR/'relay_entry.py'),str(REPO/'run_grouped.py'),
        '--scenario',SCENARIO,'--output',str(OUTPUT),'--session_hours',str(SESSION_HOURS),
        '--num_workers','2','--gpu','0']
    process=subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    try:
        for line in process.stdout: print(line,end='',flush=True)
        if process.wait(): raise RuntimeError('Training/relay stopped; last verified cloud snapshot retained')
    finally:
        if process.poll() is None:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=20)
    RELAY.claim() # refresh latest generation produced by child processes
    shutil.copyfile(PACKAGE/('environment_'+SCENARIO+'.txt'),OUTPUT/'environment.txt')
    RELAY.sync(OUTPUT)
    print((OUTPUT/'SESSION_STATUS.json').read_text())
finally:
    # If process/runtime is killed before here, sticky lease needs owner release.
    RELAY.release()
'''
        nb['cells'][9]=old.cell('code',run,9)
        p.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':build(sys.argv[1])
