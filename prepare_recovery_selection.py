"""Prepare reviewed source IDs for backup sharing, never cleanup or resume."""
import ast
import json
from pathlib import Path
import argparse


def prepare(report_path, notebook_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    assert report['format'] == 'opendetect-root-inspection-v3'
    records = report['candidates']
    assert len({r['id'] for r in records}) == len(records)
    assert all(r['name'] in ('last.pt','last.backup.pt','last.pt.json',
                            'last.backup.pt.json','progress.json') for r in records)
    assert all(r.get('md5Checksum') and not r.get('trashed') for r in records)
    assert all(len(r['owners']) == 1 and r['owners'][0]['emailAddress'].lower() == report['owner_B'] for r in records)
    snapshot = [{k:r[k] for k in ('id','name','size','md5Checksum','modifiedTime')} for r in records]
    path=Path(notebook_path)
    nb=json.loads(path.read_text(encoding='utf-8'))
    config = "PHASE = 'PREPARE_B'\nSTOPPED = 'SEMUA RUN SUMBER SUDAH BERHENTI'\nCLEANUP_ACTION = 'PREVIEW'\nSELECTED_IDS = []\nCONFIRMATION = ''\n"
    config += 'REVIEWED_OWNER_B = '+repr(report['owner_B'])+'\n'
    config += 'REVIEWED_SOURCES = '+repr(snapshot)+'\n'
    nb['cells'][1]['source']=config.splitlines(True)
    nb['cells'][0]['source'].insert(0, '# SIAP BACKUP — PREPARE_B (107 file yang ditinjau)\n\nRun all sebagai B: verifikasi ulang file terpilih, beri A akses baca, unduh manifest. Tidak menyalin checkpoint atau menghapus. Setelah manifest terunduh, gunakan BACKUP_A sebagai A; sekitar 16,2 GB ruang kosong A diperlukan.\n\n')
    code=''.join(nb['cells'][5]['source'])
    branch = '''if PHASE == 'PREPARE_B':
    if account(API)['user']['emailAddress'].lower() != REVIEWED_OWNER_B:
        raise ValueError('Pilih akun B yang sama dengan laporan pemeriksaan')
    ids = [r['id'] for r in REVIEWED_SOURCES]
    manifest = inventory(API, ids)
    manifest['files'] = [r for r in manifest['files'] if r['id'] in set(ids)]
    current = {r['id']:r for r in manifest['files']}
    for old in REVIEWED_SOURCES:
        same_content(current[old['id']], old, check_version=False)
        if current[old['id']]['modifiedTime'] != old['modifiedTime']:
            raise ValueError('Source modified since inspection; inspect again')
    assert len(current) == len(ids)
    print('VERIFIED SELECTION:', len(ids), 'files; bytes:', sum(int(r['size']) for r in manifest['files']))
    manifest = share_inventory(API, manifest, STOPPED)
    export_json('OpenDetect_B_inventory.json', manifest)
    print('MANIFEST READY. Berikutnya akun A, PHASE BACKUP_A, upload OpenDetect_B_inventory.json. Belum ada backup/delete.')
elif PHASE == 'INVENTORY_B':'''
    code=code.replace("if PHASE == 'INVENTORY_B':",branch,1)
    nb['cells'][5]['source']=code.splitlines(True)
    for c in nb['cells']:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
            c['outputs']=[]; c['execution_count']=None
    path.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
    print('Prepared',len(snapshot),'IDs; cleanup remains PREVIEW')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('report'); p.add_argument('notebook')
    a=p.parse_args(); prepare(a.report,a.notebook)
