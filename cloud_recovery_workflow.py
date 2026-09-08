"""Cloud handoff and identity-based recovery. No GPU, pickle or training required.

Drive has no cross-file transaction. Use ONE recovery session; stop all producers.
Existing canonical files are preserved, never deleted by installation.
"""
import hashlib
import io
import json
import re
import uuid
from drive_recovery import (PROJECT_ID, OUTPUTS_ID, FIELDS, STOP_PHRASE,
    meta, owner, account, children, list_files, inventory, root_candidates,
    inspect_root_metadata, share_inventory, backup_A, cleanup_B, validate_copy,
    require_A, same_content)

TRAINING_IDS = dict(zip(('A-1','A-2','A-3','B-1','B-2','B-3','C-1','C-2'), (
    '1Q90bcDkBsg9ZOZz1gzGhwzCDyHFThyO8','1o7XNa0qC5RHerrzAFNHZz_IG2d_k2zt6',
    '1GXIiW3ERQ92rRwRZaVEGX7wA49fiLgnZ','1JR7uRucmu2X3nhXWkYXUpExVdUTIIfFF',
    '19NUj-so2_mpQUNzxoabkXGNyocR1TRrv','1k7sGfDesjPtLLtsuLHuEkEOFTAFAlWZ7',
    '1XXUUMqwUZaSZTytpTVGrT723SfKdFK8F','1-9vlGZRRBXQhls3MIgnd1tEMxtLe4TEW')))
LEGACY_RECEIPT = '1MSwbn8u1NYfFi47uiYAcMt_pvDGkbK_x'


def read_json(api, identifier, expected_owner=None):
    before = meta(api, identifier)
    if (before.get('trashed') or int(before.get('size', 99999999)) > 8_000_000
            or (expected_owner and owner(before) != expected_owner)):
        raise ValueError('Wrong JSON owner, missing, or report exceeds 8 MB')
    payload = api.files().get_media(fileId=identifier).execute()
    if len(payload) != int(before['size']) or hashlib.md5(payload).hexdigest() != before['md5Checksum']:
        raise ValueError('Cloud JSON checksum mismatch')
    same_content(meta(api, identifier), before)
    return json.loads(payload)


def publish(api, folder, name, value):
    """Immutable reports: never replace another account's report."""
    from googleapiclient.http import MediaIoBaseUpload
    payload = json.dumps(value, sort_keys=True, indent=2).encode()
    r = api.files().create(body={'name':name+'_'+uuid.uuid4().hex+'.json',
        'parents':[folder]}, media_body=MediaIoBaseUpload(io.BytesIO(payload),
        mimetype='application/json'), fields=FIELDS).execute()
    if read_json(api, r['id'], account(api)['user']['emailAddress'].lower()) != value:
        raise ValueError('Published report readback failed')
    print('Laporan cloud:', r['id'])
    return r['id']


def unique_names(api, folder):
    result = {}
    for r in children(api, folder):
        if r['name'] in result:
            raise ValueError('Ambiguous duplicate name in canonical folder: '+r['name'])
        result[r['name']] = r
    return result


def identity(record):
    """Only known grouped schema; arbitrary root names are not provenance."""
    if record.get('format') != 2 or not re.fullmatch('[a-f0-9]{64}', record.get('sha256','')):
        raise ValueError('Unsupported checkpoint metadata')
    args = record['signature']['training_args']
    split = args['recovery_identity']['split']
    scenario, fold = split['scenario'], args['fold']
    if scenario not in TRAINING_IDS or type(fold) is not int or not 0 <= fold < 5:
        raise ValueError('Unknown scenario/fold')
    if split['seed'] != 2022 + fold or type(record['completed_epochs']) is not int or not 1 <= record['completed_epochs'] <= 100:
        raise ValueError('Unexpected seed/epoch')
    expected = 'USTC' if scenario.startswith('A-') else 'mal' if scenario.startswith('B-') else 'combined_USTC_mal'
    if args['dset'] != expected or args['split'] != int(scenario[-1])-1:
        raise ValueError('Dataset/scenario split mismatch')
    return scenario, fold


def config_matches(record, config):
    scenario, fold = identity(record)
    sig = record['signature']
    args = sig['training_args']
    return (config['scenario'] == scenario and config['dataset'] == args['dset']
        and sig['experiment_signature'] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        and args['recovery_identity']['split'] == config['splits'][fold])


def prepare_b(api, stopped, approved_ids):
    """Preview first. Caller must select exact IDs; root names never auto-authorize."""
    report = inspect_root_metadata(api)
    scoped = inventory(api)
    candidates = {r['id']: r for r in report['candidates']}
    for entry in report['metadata']:
        try:
            print('IDENTITAS:', identity(entry['content']), 'epoch', entry['content']['completed_epochs'],
                  'JSON', entry['id'], 'PT', entry.get('matching_checkpoint_ids', []))
        except (KeyError, ValueError, TypeError):
            pass
    print('ROOT belum dipilih:', [(r['id'],r['name']) for r in candidates.values()])
    print('Dalam outputs:', [(r['id'],r['name']) for r in scoped['files']])
    if not approved_ids:
        print('PREVIEW saja. Pilih ID sumber setelah meninjau, lalu PREPARE_B lagi. Tidak ada upload JSON manual.')
        return
    allowed = set(candidates) | {r['id'] for r in scoped['files']}
    if len(approved_ids) != len(set(approved_ids)) or not set(approved_ids) <= allowed:
        raise ValueError('IDs must be unique and in this current preview')
    m = inventory(api, [i for i in approved_ids if i in candidates])
    m['files'] = [r for r in m['files'] if r['id'] in approved_ids]
    m = share_inventory(api, m, stopped)
    # A-owned project is the anchor, not a same-name My Drive folder.
    folder = api.files().create(body={'name':'OpenDetect_HANDOFF_'+uuid.uuid4().hex,
        'mimeType':'application/vnd.google-apps.folder','parents':[PROJECT_ID]}, fields=FIELDS).execute()
    api.permissions().create(fileId=folder['id'], body={'type':'user','role':'reader',
        'emailAddress':m['owner_A']},sendNotificationEmail=False).execute()
    publish(api, folder['id'], 'INSPECTION', report)
    publish(api, folder['id'], 'MANIFEST', m)
    print('B selesai. Berikutnya akun A: BACKUP_A. JOB_FOLDER_ID =', folder['id'])


def load_manifest(api, job):
    a = require_A(api)
    if not job:
        jobs = [r for r in children(api, PROJECT_ID) if r['name'].startswith('OpenDetect_HANDOFF_')
                and r['mimeType']=='application/vnd.google-apps.folder' and owner(r)!=a]
        print('Jobs:', [(r['id'],owner(r)) for r in jobs])
        if len(jobs)!=1:
            raise ValueError('Pilih satu JOB_FOLDER_ID dari daftar; tidak memilih job terbaru secara tebakan')
        job=jobs[0]['id']
    folder=meta(api,job)
    if PROJECT_ID not in folder.get('parents',[]) or folder.get('trashed') or owner(folder)==a:
        raise ValueError('Wrong handoff folder')
    found=[r for r in children(api,job) if r['name'].startswith('MANIFEST_')]
    if len(found)!=1: raise ValueError('Expected one immutable manifest')
    m=read_json(api,found[0]['id'],owner(folder))
    if m['owner_B']!=owner(folder): raise ValueError('Manifest owner mismatch')
    return m


def load_receipt(api, identifier=''):
    a=owner(meta(api,PROJECT_ID))
    if not identifier:
        folders=[r for r in children(api,PROJECT_ID) if r['name'].startswith('RECOVERY_B_') and owner(r)==a]
        found=[]
        for f in folders:
            for r in children(api,f['id']):
                if r['name'].startswith('RECEIPT_') and owner(r)==a:
                    found.append(r)
        print('Receipts:', [(r['id'],r['name']) for r in found])
        if len(found)!=1: raise ValueError('Pilih RECEIPT_ID dari daftar, bukan upload file')
        identifier=found[0]['id']
    receipt=read_json(api,identifier,a)
    folder=meta(api,receipt['backup_folder'])
    if owner(folder)!=a or PROJECT_ID not in folder.get('parents',[]) or folder.get('trashed'):
        raise ValueError('Backup folder outside origin')
    if receipt['owner_A']!=a or receipt['project_id']!=PROJECT_ID or receipt['owner_B']==a:
        raise ValueError('Wrong receipt identity')
    files=receipt['files']
    if not files or len({r['id'] for r in files})!=len(files) or len({r['backup_id'] for r in files})!=len(files):
        raise ValueError('Empty/duplicate receipt')
    for r in files:
        validate_copy(r,meta(api,r['backup_id']),a,folder['id'])
    return receipt


def recovery_plan(api, receipt):
    """Identify all backed-up pairs and compare with exact canonical configs."""
    a=require_A(api)
    out=unique_names(api,OUTPUTS_ID)
    pairs=[]; problems=[]
    checkpoints={}
    for p in receipt['files']:
        if p['name'] in ('last.pt','last.backup.pt'):
            m=meta(api,p['backup_id'])
            digest=m.get('sha256Checksum')
            if owner(m)==a and not m.get('trashed') and digest:
                checkpoints.setdefault(digest,[]).append(p)
    for r in receipt['files']:
        if r['name'] not in ('last.pt.json','last.backup.pt.json'): continue
        try:
            record=read_json(api,r['backup_id'],a)
            scenario,fold=identity(record)
            matches=checkpoints.get(record['sha256'],[])
            if not matches: raise ValueError('No SHA256-matching checkpoint in backup')
            # Identical bytes may have several Drive IDs; lowest ID deterministic.
            p=sorted(matches,key=lambda x:x['backup_id'])[0]
            pairs.append(dict(scenario=scenario,fold=fold,epoch=record['completed_epochs'],
                              record=record,pt=p['backup_id'],json=r['backup_id']))
        except (KeyError,ValueError,TypeError) as e:
            problems.append({'metadata_id':r['backup_id'],'reason':str(e)})
    result=[]
    for key in sorted({(p['scenario'],p['fold']) for p in pairs}):
        candidates=[p for p in pairs if (p['scenario'],p['fold'])==key]
        candidate=max(candidates,key=lambda p:p['epoch'])
        row={**candidate,'decision':'STOP','reason':'Unverified canonical config'}
        try:
            scenario,fold=key
            output=out[scenario]
            if owner(output)!=a: raise ValueError('Canonical output not A-owned')
            names=unique_names(api,output['id'])
            config=read_json(api,names['GROUPED_CONFIG.json']['id'],a)
            if not config_matches(candidate['record'],config): raise ValueError('Config/code/data/split differs')
            base=f"{candidate['record']['signature']['training_args']['dset']}_split_{int(scenario[-1])-1}_fold_{fold}"
            # Never overwrite completed runs, even if their completion proof is bad.
            states=unique_names(api,names['state']['id']) if 'state' in names else {}
            if base+'.completed.json' in states:
                raise ValueError('Completed marker exists: preserve; use training verification, no installation')
            runs=unique_names(api,names['resume_state']['id'])
            target=runs[base]
            if owner(target)!=a: raise ValueError('Resume folder not owned by A')
            slots=unique_names(api,target['id']); valid=[]
            for slot in ('last.pt','last.backup.pt'):
                try:
                    rec=read_json(api,slots[slot+'.json']['id'],a)
                    if (owner(slots[slot])==a and slots[slot].get('sha256Checksum')==rec['sha256']
                            and config_matches(rec,config) and identity(rec)==key):
                        valid.append((rec['completed_epochs'],slot,rec['sha256']))
                except (KeyError,ValueError,TypeError): pass
            latest=max((v[0] for v in valid),default=0)
            same_epoch=[v for v in valid if v[0]==candidate['epoch']]
            if same_epoch and any(v[2]!=candidate['record']['sha256'] for v in same_epoch):
                raise ValueError('Same epoch but different checkpoint: divergent histories; manual review')
            if latest>=candidate['epoch']:
                decision='KEEP_A'
            else:
                decision='INSTALL_AVAILABLE'
            # Replace only the other slot, keeping the most recent valid slot intact.
            preserve=max(valid)[1] if valid else None
            slot='last.backup.pt' if preserve=='last.pt' else 'last.pt'
            row.update(decision=decision,reason='',canonical_epoch=latest,target_folder=target['id'],
                slot=slot,next_epoch=max(latest,candidate['epoch'])+1,
                training_url='https://colab.research.google.com/drive/'+TRAINING_IDS[scenario],
                destination='outputs/'+scenario+'/resume_state/'+base,
                config_id=names['GROUPED_CONFIG.json']['id'])
        except (KeyError,ValueError,TypeError) as e: row['reason']=str(e)
        result.append(row)
    return {'format':'opendetect-cloud-plan-v1','runs':result,'unmatched':problems,
            'model_load_rng_tested':False}


def print_plan(plan):
    for r in plan['runs']:
        print(r['scenario'],'fold',r['fold'],'backup epoch',r['epoch'],
              'A epoch',r.get('canonical_epoch'),r['decision'],r['reason'])
        if r.get('destination'): print('Tujuan:',r['destination'],'| Lanjut:',r['training_url'])
    print('Metadata belum berpasangan:',len(plan['unmatched']))
    print('Ini verifikasi cloud, BUKAN bukti model/RNG berhasil dimuat oleh training.')


def pending_cleanup(api, receipt, permanent=False):
    """Resume interrupted cleanup only for live, positively identified sources.

    A 404 means missing OR inaccessible: never assert it was deleted by us.
    Backups are independently rechecked by load_receipt and cleanup_B.
    """
    active=account(api)['user']['emailAddress'].lower()
    if active!=receipt['owner_B'] or active==receipt['owner_A']:
        raise ValueError('Wrong source account')
    pending=[]; unavailable=[]
    for r in receipt['files']:
        try:
            s=meta(api,r['id'])
        except Exception as e:
            if getattr(getattr(e,'resp',None),'status',None)!=404: raise
            unavailable.append(r['id'])
            continue
        if owner(s)!=active: raise ValueError('Source owner changed')
        same_content(s,r,check_version=not s.get('trashed'))
        if permanent and not s.get('trashed'):
            raise ValueError('Trash all selected sources first')
        pending.append(r['id'])
    if unavailable:
        print('TIDAK TERSEDIA (404; bukan bukti sudah terhapus):',unavailable)
    return pending


def install_one(api, receipt, scenario, fold, stopped, confirmation):
    require_A(api,1_000_000_000)
    if stopped!=STOP_PHRASE or confirmation!='PASANG CHECKPOINT TERVERIFIKASI KE A':
        raise ValueError('Stop ALL A/B producers; explicit install confirmation required')
    plan=recovery_plan(api,receipt)
    selected=[r for r in plan['runs'] if r['scenario']==scenario and r['fold']==fold]
    if len(selected)!=1: raise ValueError('Select exactly one identified scenario/fold')
    r=selected[0]
    if r['decision']=='KEEP_A':
        print('Tidak dipasang: A sama/lebih baru.'); return r
    if r['decision']!='INSTALL_AVAILABLE': raise ValueError(r['reason'])
    target=r['target_folder']; slot=r['slot']; before=unique_names(api,target)
    token=uuid.uuid4().hex
    staged=[]
    for source,suffix in ((r['pt'],''),(r['json'],'.json')):
        src=meta(api,source)
        copy=api.files().copy(fileId=source,body={'name':'STAGED_'+token+suffix,
             'parents':[target]},fields=FIELDS).execute()
        validate_copy(src,meta(api,copy['id']),require_A(api),target)
        staged.append(copy)
    # Compare again before publication; abort if a producer or another recovery wrote.
    if unique_names(api,target)!={**before,**{s['name']:s for s in staged}}:
        # Compare stable IDs/content/version rather than relying on list ordering.
        now=unique_names(api,target)
        for name,m in before.items(): same_content(now[name],m)
        if set(now)!=set(before)|{s['name'] for s in staged}: raise ValueError('Target changed')
    fresh=recovery_plan(api,receipt)
    current=next(x for x in fresh['runs'] if x['scenario']==scenario and x['fold']==fold)
    if current['decision']!='INSTALL_AVAILABLE' or current['slot']!=slot:
        raise ValueError('Plan changed; staged files preserved, no canonical replacement')
    for suffix in ('','.json'):
        old=before.get(slot+suffix)
        if old:
            api.files().update(fileId=old['id'],body={'name':'BEFORE_'+token+'_'+slot+suffix}).execute()
    # Publish PT first, companion JSON last. Other valid slot remains untouched.
    for copy,suffix in zip(staged,('','.json')):
        api.files().update(fileId=copy['id'],body={'name':slot+suffix}).execute()
    after=unique_names(api,target)
    rec=read_json(api,after[slot+'.json']['id'],require_A(api))
    if rec!=r['record'] or after[slot].get('sha256Checksum')!=rec['sha256']:
        raise ValueError('Post-install verification failed; previous files preserved')
    print('PASANG TERVERIFIKASI:',r['destination'],'epoch',r['epoch'])
    return r
