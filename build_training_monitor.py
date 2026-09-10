"""Build one read-only Colab dashboard for all grouped-training scenarios."""
import json
from pathlib import Path

def code(source, ident):
    return {'cell_type': 'code', 'execution_count': None, 'id': ident,
            'metadata': {}, 'outputs': [], 'source': source.splitlines(True)}

intro = """# OpenDetect — Status Training Semua Skenario

Dashboard **read-only** untuk A-1 sampai C-2. Notebook membaca snapshot terbaru
di Drive akun worker yang sedang diautentikasi; tidak menjalankan training,
tidak meng-upload, dan tidak menghapus checkpoint.

Status fold: SELESAI berarti seluruh artefak evaluasi tersedia; EPOCH n/100
berarti checkpoint aktif; BELUM MULAI berarti belum ada checkpoint. Jika skenario
dikerjakan akun berbeda, jalankan notebook sekali pada masing-masing akun.
"""

setup = r'''# @title 1. Login Drive worker dan siapkan pemeriksa
from google.colab import auth, userdata
import google.auth
from googleapiclient.discovery import build
import hashlib, json, requests
from IPython.display import display
import pandas as pd

SCENARIOS = ('A-1','A-2','A-3','B-1','B-2','B-3','C-1','C-2')
FORMAT = 'worker-drive-v1'
FIELDS = 'id,name,mimeType,size,sha256Checksum,parents,owners(emailAddress),trashed,appProperties'
auth.authenticate_user()
credentials, _ = google.auth.default()
drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
account = drive.about().get(fields='user(emailAddress),storageQuota').execute()
WORKER_EMAIL = account['user']['emailAddress'].lower()

def children(parent):
    result, token = [], None
    while True:
        page = drive.files().list(q=f"'{parent}' in parents and trashed=false",
            fields='nextPageToken,files('+FIELDS+')', pageSize=1000, pageToken=token).execute()
        result.extend(page.get('files', [])); token = page.get('nextPageToken')
        if not token: return result

def owned(record):
    return [x.get('emailAddress','').lower() for x in record.get('owners', [])] == [WORKER_EMAIL]

def read_json(record, parent):
    if (not record or not owned(record) or record.get('trashed') or
            parent not in record.get('parents', []) or int(record.get('size', 0)) > 2_000_000):
        raise ValueError('metadata JSON tidak valid')
    raw = drive.files().get_media(fileId=record['id']).execute()
    if hashlib.sha256(raw).hexdigest() != record.get('sha256Checksum'):
        raise ValueError('checksum JSON Drive berbeda')
    return json.loads(raw)

def a_generation(scenario):
    try:
        response = requests.post(userdata.get('OPENDETECT_RELAY_URL'),
            json={'key':userdata.get('OPENDETECT_WORKER_KEY'), 'scenario':scenario,
                  'lease':'monitor-read-only', 'action':'hello'}, timeout=30)
        data = response.json()
        return data['result']['generation'] if data.get('ok') else '?'
    except Exception:
        return 'secret tidak tersedia'
'''

dashboard = r'''# @title 2. Refresh dashboard semua skenario
rows, details = [], {}
root = children('root')
for scenario in SCENARIOS:
    try:
        folders = [f for f in root if f['name'] == 'OpenDetect_WORKER_'+scenario
                   and owned(f) and f.get('appProperties',{}).get('opendetect') == FORMAT]
        if len(folders) != 1:
            raise ValueError('folder tidak ada' if not folders else 'folder duplikat')
        folder = folders[0]; objects = children(folder['id'])
        pointers = [f for f in objects if f['name'] == 'CURRENT.json']
        if len(pointers) != 1: raise ValueError('CURRENT.json tidak unik')
        manifest = read_json(pointers[0], folder['id'])
        if manifest.get('format') != FORMAT or manifest.get('scenario') != scenario:
            raise ValueError('identitas manifest berbeda')
        listed = {f['id']: f for f in objects}; files = manifest.get('files', [])
        invalid = [r['path'] for r in files if r['id'] not in listed or
                   int(listed[r['id']].get('size',-1)) != r['size'] or
                   listed[r['id']].get('sha256Checksum') != r['sha256']]
        if invalid: raise ValueError(f'{len(invalid)} objek gagal verifikasi')
        paths = {r['path']: r for r in files}
        config_rec = paths.get('GROUPED_CONFIG.json')
        config = read_json(listed[config_rec['id']], folder['id']) if config_rec else {}
        dataset, split = config.get('dataset','?'), config.get('split','?')
        fold_rows, completed = [], 0
        for fold in range(5):
            name = f'{dataset}_split_{split}_fold_{fold}'
            required = (f'save_model/{name}.pt', f'results/{name}.json',
                        f'results/{name}.scores.npz', f'state/{name}.completed.json')
            done = all(p in paths for p in required)
            if done:
                marker = read_json(listed[paths[required[3]]['id']], folder['id'])
                done = (marker.get('config') == config and
                    marker.get('checkpoint_sha256') == paths[required[0]]['sha256'] and
                    marker.get('result_sha256') == paths[required[1]]['sha256'] and
                    marker.get('scores_sha256') == paths[required[2]]['sha256'])
            epoch = 100 if done else 0
            progress = paths.get(f'resume_state/{name}/progress.json')
            if progress:
                progress_data = read_json(listed[progress['id']], folder['id'])
                epoch = int(progress_data.get('completed_epochs', 0))
            metrics = {}
            result = paths.get(f'results/{name}.json')
            if result: metrics = read_json(listed[result['id']], folder['id'])
            completed += int(done)
            status = 'SELESAI' if done else (f'EPOCH {epoch}/100' if epoch else 'BELUM MULAI')
            fold_rows.append({'Fold':fold, 'Seed':2022+fold, 'Status':status, 'Epoch':epoch,
                'Closed acc':metrics.get('closed_accuracy','—'),
                'Closed F1':metrics.get('closed_f1','—'), 'AUROC':metrics.get('auroc','—'),
                'Open acc':metrics.get('open_accuracy','—')})
        receipt = manifest.get('receipt', {})
        receipt_files = {r['path']:r['sha256'] for r in receipt.get('files', [])}
        synced = bool(receipt) and all(receipt_files.get(r['path']) == r['sha256'] for r in files)
        active = next((x for x in fold_rows if 0 < x['Epoch'] < 100), None)
        rows.append({'Skenario':scenario, 'Fold selesai':f'{completed}/5',
            'Fold aktif':active['Fold'] if active else '—', 'Epoch aktif':active['Epoch'] if active else '—',
            'Worker rev':manifest.get('revision','?'), 'Ukuran':f"{sum(r['size'] for r in files)/2**30:.2f} GiB",
            'Upload A':'sinkron' if synced else 'belum terbaru', 'A generation':a_generation(scenario),
            'Catatan':''})
        details[scenario] = pd.DataFrame(fold_rows)
    except Exception as error:
        rows.append({'Skenario':scenario, 'Fold selesai':'—', 'Fold aktif':'—',
            'Epoch aktif':'—', 'Worker rev':'—', 'Ukuran':'—', 'Upload A':'—',
            'A generation':a_generation(scenario), 'Catatan':str(error)})

print('Akun worker:', WORKER_EMAIL)
display(pd.DataFrame(rows))
for scenario, table in details.items():
    print('\n'+scenario); display(table)
print('Dashboard selesai — tidak ada file yang diubah.')
'''

nb = {'cells':[
        {'cell_type':'markdown','id':'intro','metadata':{},'source':intro.splitlines(True)},
        code(setup,'setup'), code(dashboard,'dashboard')],
      'metadata':{'accelerator':'','colab':{'name':'OpenDetect_TRAINING_MONITOR.ipynb','provenance':[]},
          'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}},
      'nbformat':4,'nbformat_minor':5}

if __name__ == '__main__':
    output = Path(__file__).with_name('OpenDetect_TRAINING_MONITOR.ipynb')
    output.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(output)
