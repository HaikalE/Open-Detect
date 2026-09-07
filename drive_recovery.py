"""Account-aware Drive backup. No training, pickle loading, or blanket Drive deletion."""
import hashlib
import json
import re
from pathlib import Path
import time
import uuid

PROJECT_ID = '1Sry3j9KonUEQ88besc12_zxkyI0DoXfI'
OUTPUTS_ID = '1o8-O578YTfc9FAwGKBZvDGJlzUYE7awI'
FIELDS = 'id,name,mimeType,size,md5Checksum,version,modifiedTime,parents,trashed,owners(emailAddress),appProperties'
STOP_PHRASE = 'SEMUA RUN SUMBER SUDAH BERHENTI'


def file_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Use a raw Drive file/folder ID, not a URL')
    return value


def meta(api, identifier):
    return api.files().get(fileId=file_id(identifier), fields=FIELDS).execute()


def owner(record):
    owners = record.get('owners', [])
    if len(owners) != 1 or not owners[0].get('emailAddress'):
        raise ValueError('Expected one verifiable My Drive owner; Shared Drive is a different workflow')
    return owners[0]['emailAddress'].lower()


def account(api):
    return api.about().get(fields='user(emailAddress),storageQuota').execute()


def list_files(api, query):
    result, token = [], None
    while True:
        page = api.files().list(q=query, fields=f'nextPageToken,files({FIELDS})',
                                pageSize=1000, pageToken=token).execute()
        result.extend(page.get('files', []))
        token = page.get('nextPageToken')
        if not token:
            return result


def children(api, identifier):
    return list_files(api, f"'{file_id(identifier)}' in parents and trashed = false")


def require_A(api, minimum_bytes=0):
    project = meta(api, PROJECT_ID)
    a = owner(project)
    info = account(api)
    active = info['user']['emailAddress'].lower()
    if active != a:
        raise RuntimeError(f'STOP: Drive terotorisasi {active}, tetapi folder origin milik {a}. Editor bukan pemilik. Pilih akun A; belum boleh training/backup A.')
    quota = info.get('storageQuota', {})
    if 'limit' in quota:
        free = int(quota['limit']) - int(quota.get('usage', 0))
        if free < minimum_bytes:
            raise RuntimeError(f'STOP: kuota A tersisa {free/1e9:.2f} GB; butuh cadangan {minimum_bytes/1e9:.2f} GB.')
        print(f'Owner A verified: {a}; free storage {free/1e9:.2f} GB')
    elif minimum_bytes:
        print('Quota limit not exposed by Google; cannot guarantee available space.')
    return a


def verify_mount(api, project_path):
    """Verify both folder identity and actual FUSE writer ownership, not path name."""
    a = require_A(api, 6_000_000_000)
    path = Path(project_path)
    if not path.is_dir():
        raise RuntimeError('Folder origin not mounted. Do not create a same-name replacement folder.')
    probe_name = 'opendetect_owner_probe_' + uuid.uuid4().hex + '.txt'
    payload = uuid.uuid4().hex.encode()
    probe = path / probe_name
    probe.write_bytes(payload)
    expected = hashlib.md5(payload).hexdigest()
    for _ in range(12):
        found = list_files(api, f"'{PROJECT_ID}' in parents and name = '{probe_name}' and trashed = false")
        if len(found) == 1 and found[0].get('md5Checksum') == expected:
            if owner(found[0]) != a:
                raise RuntimeError('STOP: mount menulis sebagai akun B. Jangan mulai training. Probe kecil dipertahankan untuk diagnosis.')
            api.files().delete(fileId=found[0]['id']).execute()  # Only our verified tiny temporary probe.
            print('PASS: folder ID origin + mounted writer A verified')
            return
        time.sleep(2)
    raise RuntimeError('STOP: cloud probe not confirmed in origin folder. Mount/account/sync unresolved; no training started.')


def binary(record):
    return (not record.get('trashed') and record.get('md5Checksum') and
            'size' in record and not record['mimeType'].startswith('application/vnd.google-apps.'))


def inventory(api, extra_ids=()):
    """Only B-owned files below the specific experiment output tree, plus explicit IDs."""
    b = account(api)['user']['emailAddress'].lower()
    a = owner(meta(api, PROJECT_ID))
    if a == b:
        raise RuntimeError('Inventory B must run authenticated as B, not origin A')
    records, visited = {}, set()
    def walk(folder, relative):
        if folder in visited:
            raise ValueError('Repeated folder ID')
        visited.add(folder)
        for r in children(api, folder):
            rel = relative + [r['name']]
            if r['mimeType'] == 'application/vnd.google-apps.folder':
                walk(r['id'], rel)
            elif binary(r) and owner(r) == b:
                records[r['id']] = {**r, 'relative_parts': rel, 'scope': 'experiment_outputs'}
    walk(OUTPUTS_ID, [])
    for identifier in extra_ids:
        r = meta(api, identifier)
        if not binary(r) or owner(r) != b:
            raise ValueError('Explicit orphan is not a live B-owned binary file: ' + identifier)
        records.setdefault(identifier, {**r, 'relative_parts': ['ORPHANS', r['id'], r['name']], 'scope': 'explicit_file_id'})
    return {'format': 'opendetect-recovery-v1', 'owner_A': a, 'owner_B': b,
            'project_id': PROJECT_ID, 'files': list(records.values())}


def root_candidates(api):
    # A candidate name is NOT permission to delete/copy it; user selects exact IDs.
    return list_files(api, "'root' in parents and 'me' in owners and trashed = false and (name = 'last.pt' or name = 'last.backup.pt')")


def share_inventory(api, manifest, stopped):
    if stopped != STOP_PHRASE:
        raise RuntimeError('Pause/stop source training first, then confirm the exact phrase')
    b = account(api)['user']['emailAddress'].lower()
    if b != manifest['owner_B']:
        raise ValueError('Wrong source account')
    for record in manifest['files']:
        now = meta(api, record['id'])
        if not binary(now) or owner(now) != b or now['md5Checksum'] != record['md5Checksum']:
            raise ValueError('Source changed; redo inventory')
        perms = api.permissions().list(fileId=now['id'], fields='permissions(type,emailAddress,role)').execute()
        if not any(p.get('emailAddress', '').lower() == manifest['owner_A'] for p in perms.get('permissions', [])):
            api.permissions().create(fileId=now['id'], body={'type': 'user', 'role': 'reader',
                'emailAddress': manifest['owner_A']}, sendNotificationEmail=False).execute()
        # Sharing can change the Drive version; snapshot metadata AFTER sharing.
        latest = meta(api, now['id'])
        if latest['md5Checksum'] != record['md5Checksum']:
            raise ValueError('Source changed during sharing')
        record.update(latest)
    return manifest


def same_content(source, snapshot, check_version=True):
    keys = ('id', 'name', 'size', 'md5Checksum')
    # Moving to trash can change metadata; content identity must still match.
    if check_version:
        keys += ('modifiedTime',)
    for key in keys:
        if str(source.get(key)) != str(snapshot.get(key)):
            raise ValueError('Source changed: ' + key)
    if check_version and source.get('version') != snapshot.get('version'):
        raise ValueError('Source version changed; redo inventory/backup')


def validate_copy(source, backup, a, folder):
    if (source['id'] == backup['id'] or owner(backup) != a or backup.get('trashed') or
            folder not in backup.get('parents', []) or source['md5Checksum'] != backup.get('md5Checksum') or
            int(source['size']) != int(backup.get('size', -1))):
        raise ValueError('Backup owner/location/size/checksum validation failed')


def backup_A(api, manifest, stopped):
    if stopped != STOP_PHRASE:
        raise RuntimeError('Source training must be stopped before a stable backup')
    a = require_A(api, sum(int(r['size']) for r in manifest['files']) + 1_000_000_000)
    if manifest['format'] != 'opendetect-recovery-v1' or manifest['owner_A'] != a or manifest['project_id'] != PROJECT_ID:
        raise ValueError('Wrong manifest/project')
    if manifest['owner_B'] == a or len({r['id'] for r in manifest['files']}) != len(manifest['files']):
        raise ValueError('Invalid source owner/duplicate IDs')
    # Deterministic folder per source account; reruns find prior copies by source+version.
    name = 'RECOVERY_B_' + hashlib.sha256(manifest['owner_B'].encode()).hexdigest()[:12]
    matches = list_files(api, f"'{PROJECT_ID}' in parents and name = '{name}' and trashed = false")
    if len(matches) > 1:
        raise ValueError('Duplicate recovery folders; inspect before proceeding')
    folder = matches[0] if matches else api.files().create(body={'name':name, 'mimeType':'application/vnd.google-apps.folder',
        'parents':[PROJECT_ID]}, fields=FIELDS).execute()
    if owner(folder) != a or folder['mimeType'] != 'application/vnd.google-apps.folder':
        raise ValueError('Recovery folder is not owned by A')
    permissions = api.permissions().list(fileId=folder['id'], fields='permissions(emailAddress)').execute()
    if not any(p.get('emailAddress','').lower() == manifest['owner_B'] for p in permissions.get('permissions', [])):
        api.permissions().create(fileId=folder['id'], body={'type':'user', 'role':'reader',
            'emailAddress':manifest['owner_B']}, sendNotificationEmail=False).execute()
    receipt = {**manifest, 'backup_folder':folder['id'], 'files':[]}
    for r in manifest['files']:
        now = meta(api, r['id'])
        same_content(now, r)
        if owner(now) != manifest['owner_B'] or not binary(now):
            raise ValueError('Not an unchanged B-owned file')
        key = hashlib.sha256((r['id'] + ':' + r['version']).encode()).hexdigest()
        copies = list_files(api, f"'{folder['id']}' in parents and trashed = false and appProperties has {{ key='recoveryKey' and value='{key}' }}")
        if len(copies) > 1:
            raise ValueError('Ambiguous backup copies')
        copied = copies[0] if copies else api.files().copy(fileId=r['id'], body={
            'name':r['id'] + '__' + r['name'], 'parents':[folder['id']],
            'appProperties':{'recoveryKey':key}}, fields=FIELDS).execute()
        # API copy is server-side; no checkpoint download to RAM and no pickle load.
        same_content(meta(api, r['id']), r)
        verified = meta(api, copied['id'])
        validate_copy(r, verified, a, folder['id'])
        receipt['files'].append({**r, 'backup_id':verified['id']})
        print('BACKUP VERIFIED:', r['name'], r['size'], 'bytes', flush=True)
    return receipt


def cleanup_B(api, receipt, selected_ids, stopped, permanent=False, confirmation=''):
    if stopped != STOP_PHRASE:
        raise RuntimeError('Do not delete while source training is running')
    active = account(api)['user']['emailAddress'].lower()
    a = owner(meta(api, PROJECT_ID))
    if active != receipt['owner_B'] or active == a or receipt['owner_A'] != a:
        raise ValueError('Wrong cleanup account')
    folder = meta(api, receipt['backup_folder'])
    if owner(folder) != a or PROJECT_ID not in folder.get('parents', []) or folder.get('trashed'):
        raise ValueError('Backup folder not safely in A')
    expected = 'HAPUS PERMANEN FILE TERVERIFIKASI' if permanent else 'PINDAHKAN FILE TERVERIFIKASI KE SAMPAH'
    if confirmation != expected:
        raise RuntimeError('Explicit cleanup phrase required: ' + expected)
    indexed = {r['id']: r for r in receipt['files']}
    if not selected_ids or len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(indexed):
        raise ValueError('Select explicit, unique source IDs from this verified receipt only')
    # Validate every selected source+backup before performing the first deletion.
    for identifier in selected_ids:
        r = indexed[identifier]
        source, backup = meta(api, identifier), meta(api, r['backup_id'])
        if owner(source) != active:
            raise ValueError('Refusing to delete a file not owned by B')
        same_content(source, r, check_version=not source.get('trashed'))
        validate_copy(r, backup, a, folder['id'])
        if permanent and not source.get('trashed'):
            raise ValueError('Trash first, verify backup, then permanently delete in a separate run')
    log = []
    for identifier in selected_ids:
        # Recheck immediately before each irreversible operation too.
        r = indexed[identifier]
        source = meta(api, identifier)
        if owner(source) != active or (permanent and not source.get('trashed')):
            raise ValueError('Source owner/trash state changed; cleanup stopped')
        same_content(source, r, check_version=not source.get('trashed'))
        validate_copy(r, meta(api, r['backup_id']), a, folder['id'])
        if permanent:
            api.files().delete(fileId=identifier).execute()
        elif not source.get('trashed'):
            api.files().update(fileId=identifier, body={'trashed':True}).execute()
        log.append({'id':identifier, 'action':'permanently_deleted' if permanent else 'trashed'})
        print(log[-1], flush=True)
    return log
