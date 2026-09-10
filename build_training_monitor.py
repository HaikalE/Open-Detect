"""Build one read-only Colab dashboard for all grouped-training scenarios."""
import json
from pathlib import Path

def code(source, ident):
    return {'cell_type': 'code', 'execution_count': None, 'id': ident,
            'metadata': {}, 'outputs': [], 'source': source.splitlines(True)}

intro = """# OpenDetect — Status Training Semua Skenario

Dashboard **read-only** untuk A-1 sampai C-2. Notebook membaca snapshot terbaru
yang sudah di-upload ke folder outputs milik Drive A; tidak menjalankan training,
tidak meng-upload, dan tidak menghapus checkpoint.

Status fold: SELESAI berarti seluruh artefak evaluasi tersedia; EPOCH n/100
berarti checkpoint aktif; BELUM MULAI berarti belum ada checkpoint.
"""

setup = r'''# @title 1. Login Drive worker dan siapkan pemeriksa
from google.colab import auth, userdata
import google.auth
from googleapiclient.discovery import build
import hashlib, json
from IPython.display import display
import pandas as pd

SCENARIOS = ('A-1','A-2','A-3','B-1','B-2','B-3','C-1','C-2')
FORMAT = 'worker-drive-v1'
OWNER_OUTPUTS = '1o8-O578YTfc9FAwGKBZvDGJlzUYE7awI'
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

'''

dashboard = r'''# @title 2. Refresh dashboard semua skenario
rows, details = [], {}
scenario_folders = {f['name']:f for f in children(OWNER_OUTPUTS)}
for scenario in SCENARIOS:
    try:
        folder = scenario_folders.get(scenario)
        if not folder: raise ValueError('folder skenario di A tidak ada')
        relay = next((f for f in children(folder['id']) if f['name']=='_relay'), None)
        if not relay: raise ValueError('snapshot relay di A tidak ada')
        relay_children = children(relay['id'])
        snapshots = next((f for f in relay_children if f['name']=='snapshots'), None)
        object_folder = next((f for f in relay_children if f['name']=='objects'), None)
        if not snapshots or not object_folder: raise ValueError('struktur snapshot A tidak lengkap')
        generations = [(int(f['name'][11:-5]),f) for f in children(snapshots['id'])
                       if f['name'].startswith('generation_') and f['name'].endswith('.json')]
        if not generations: raise ValueError('belum ada generation di A')
        generation, pointer = max(generations)
        manifest = read_json(pointer, snapshots['id'])
        objects = children(object_folder['id'])
        listed = {f['id']: f for f in objects}; files = manifest.get('files', [])
        invalid = [r['path'] for r in files if r['id'] not in listed or
                   int(listed[r['id']].get('size',-1)) != r['size'] or
                   listed[r['id']].get('sha256Checksum') != r['sha256']]
        if invalid: raise ValueError(f'{len(invalid)} objek gagal verifikasi')
        paths = {r['path']: r for r in files}
        config_rec = paths.get('GROUPED_CONFIG.json')
        config = read_json(listed[config_rec['id']], object_folder['id']) if config_rec else {}
        dataset, split = config.get('dataset','?'), config.get('split','?')
        fold_rows, completed = [], 0
        for fold in range(5):
            name = f'{dataset}_split_{split}_fold_{fold}'
            required = (f'save_model/{name}.pt', f'results/{name}.json',
                        f'results/{name}.scores.npz', f'state/{name}.completed.json')
            done = all(p in paths for p in required)
            if done:
                marker = read_json(listed[paths[required[3]]['id']], object_folder['id'])
                done = (marker.get('config') == config and
                    marker.get('checkpoint_sha256') == paths[required[0]]['sha256'] and
                    marker.get('result_sha256') == paths[required[1]]['sha256'] and
                    marker.get('scores_sha256') == paths[required[2]]['sha256'])
            epoch = 100 if done else 0
            progress = paths.get(f'resume_state/{name}/progress.json')
            if progress:
                progress_data = read_json(listed[progress['id']], object_folder['id'])
                epoch = int(progress_data.get('completed_epochs', 0))
            metrics = {}
            result = paths.get(f'results/{name}.json')
            if result: metrics = read_json(listed[result['id']], object_folder['id'])
            completed += int(done)
            status = 'SELESAI' if done else (f'EPOCH {epoch}/100' if epoch else 'BELUM MULAI')
            fold_rows.append({'Fold':fold, 'Seed':2022+fold, 'Status':status, 'Epoch':epoch,
                'Closed acc':metrics.get('closed_accuracy','—'),
                'Closed F1':metrics.get('closed_f1','—'), 'AUROC':metrics.get('auroc','—'),
                'Open acc':metrics.get('open_accuracy','—')})
        active = next((x for x in fold_rows if 0 < x['Epoch'] < 100), None)
        rows.append({'Skenario':scenario, 'Fold selesai':f'{completed}/5',
            'Fold aktif':active['Fold'] if active else '—', 'Epoch aktif':active['Epoch'] if active else '—',
            'A generation':generation, 'Ukuran snapshot':f"{sum(r['size'] for r in files)/2**30:.2f} GiB",
            'Catatan':''})
        details[scenario] = pd.DataFrame(fold_rows)
    except Exception as error:
        rows.append({'Skenario':scenario, 'Fold selesai':'—', 'Fold aktif':'—',
            'Epoch aktif':'—', 'A generation':'—', 'Ukuran snapshot':'—', 'Catatan':str(error)})

print('Drive A:', WORKER_EMAIL)
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
