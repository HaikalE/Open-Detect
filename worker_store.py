"""Bounded worker-owned Drive snapshots, verified before acknowledging an epoch.

Only superseded objects created by this store are deleted after manifest readback.
Unknown/orphan files are preserved. No PyTorch imports or credential serialization.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from relay_client import Relay, digest, safe_path
from drive_reader import DriveReader

FIELDS = 'id,name,mimeType,size,sha256Checksum,parents,owners(emailAddress),trashed,appProperties'
FORMAT = 'worker-drive-v1'
RESERVE = 1_500_000_000


class WorkerStore:
    def __init__(self, api, session, scenario, create=False):
        self.api, self.reader, self.scenario = api, DriveReader(session), scenario
        info = api.about().get(fields='user(emailAddress),storageQuota').execute()
        self.owner = info['user']['emailAddress'].lower()
        name = 'OpenDetect_WORKER_' + scenario
        found = self.children('root')
        found = [f for f in found if f['name'] == name and self.owned(f)]
        if len(found) > 1:
            raise ValueError('Duplicate worker folders; select/reconcile before continuing')
        if not found:
            if not create:
                raise ValueError('No worker checkpoint folder for this account/scenario')
            folder = api.files().create(body={'name': name, 'mimeType': 'application/vnd.google-apps.folder',
                'appProperties': {'opendetect': FORMAT, 'scenario': scenario}}, fields=FIELDS).execute()
        else:
            folder = found[0]
        if (folder['mimeType'] != 'application/vnd.google-apps.folder' or
                folder.get('appProperties', {}).get('opendetect') != FORMAT):
            raise ValueError('Unrecognized worker folder; preserved')
        self.folder = folder['id']
        children = self.children(self.folder)
        pointers = [f for f in children if f['name'] == 'CURRENT.json']
        if len(pointers) > 1 or (not pointers and children):
            raise ValueError('Incomplete/ambiguous worker initialization; files preserved')
        self.pointer = pointers[0]['id'] if pointers else None
        self.manifest = self.read_json(self.pointer) if self.pointer else None
        if self.manifest and (self.manifest.get('format') != FORMAT or
                self.manifest.get('scenario') != scenario or self.manifest.get('owner') != self.owner):
            raise ValueError('Worker manifest identity mismatch')
        print('WORKER DRIVE:', self.owner, '| folder:', self.folder, flush=True)

    def owned(self, record):
        return [x.get('emailAddress', '').lower() for x in record.get('owners', [])] == [self.owner]

    def meta(self, identifier):
        return self.api.files().get(fileId=identifier, fields=FIELDS).execute()

    def children(self, folder):
        result, token = [], None
        while True:
            page = self.api.files().list(q=f"'{folder}' in parents and trashed=false",
                fields='nextPageToken,files(' + FIELDS + ')', pageSize=1000, pageToken=token).execute()
            result.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return result

    def read_json(self, identifier):
        m = self.meta(identifier)
        if not self.owned(m) or m.get('trashed') or self.folder not in m.get('parents', []) or int(m['size']) > 2_000_000:
            raise ValueError('Invalid worker manifest')
        content = self.api.files().get_media(fileId=identifier).execute()
        if hashlib.sha256(content).hexdigest() != m.get('sha256Checksum'):
            raise ValueError('Worker manifest checksum mismatch')
        return json.loads(content)

    def check_space(self, incoming=0):
        q = self.api.about().get(fields='storageQuota').execute()['storageQuota']
        if 'limit' not in q:
            raise RuntimeError('Drive quota unavailable; cannot preflight worker storage')
        free = int(q['limit']) - int(q.get('usage', 0))
        if free < incoming + RESERVE:
            raise RuntimeError(f'Worker Drive low: {free/1e9:.2f} GB free; need {(incoming+RESERVE)/1e9:.2f} GB. Use CPU transfer cell.')
        return free

    def publish(self, value):
        from googleapiclient.http import MediaIoBaseUpload
        if self.pointer and self.read_json(self.pointer) != self.manifest:
            raise RuntimeError('Worker manifest changed; refusing overwrite')
        payload = json.dumps(value, sort_keys=True).encode()
        media = MediaIoBaseUpload(io.BytesIO(payload), mimetype='application/json')
        if self.pointer:
            self.api.files().update(fileId=self.pointer, media_body=media, fields='id').execute()
        else:
            r = self.api.files().create(body={'name': 'CURRENT.json', 'parents': [self.folder],
                'appProperties': {'opendetect': FORMAT}}, media_body=media, fields='id').execute()
            self.pointer = r['id']
        if self.read_json(self.pointer) != value:
            raise RuntimeError('Worker manifest readback failed; objects retained')
        self.manifest = value

    def bind(self, relay):
        """Called under an exclusive service lease, including first initialization."""
        if not self.manifest:
            self.publish({'format': FORMAT, 'owner': self.owner, 'scenario': self.scenario,
                'base_generation': relay.generation, 'files': [], 'revision': 0})
        if self.manifest['base_generation'] != relay.generation:
            # Recover an A commit whose response/local receipt was lost. Never use epoch/mtime guesses.
            current = {f['path']: f for f in relay.files}
            files = self.manifest['files']
            if not files or any(f['path'] not in current or
                    current[f['path']]['sha256'] != f['sha256'] for f in files):
                raise RuntimeError('Worker/A histories differ; preserve both and reconcile before training')
            self.publish({**self.manifest, 'base_generation': relay.generation})

    def restore(self, output, relay):
        if not self.manifest['files']:
            relay.restore(output)
            return
        old_files, old_reader = relay.files, relay.reader
        try:
            relay.files, relay.reader = self.manifest['files'], self.reader
            for record in relay.files:
                self.verify(record)
            relay.restore(output)
        finally:
            relay.files, relay.reader = old_files, old_reader

    def verify(self, record):
        m = self.meta(record['id'])
        if (not self.owned(m) or m.get('trashed') or self.folder not in m.get('parents', []) or
                int(m.get('size', -1)) != record['size'] or m.get('sha256Checksum') != record['sha256'] or
                m.get('appProperties', {}).get('opendetect') != FORMAT):
            raise ValueError('Worker object owner/location/hash mismatch: ' + record['path'])

    def upload(self, path, relative):
        from googleapiclient.http import MediaFileUpload
        checksum, size = digest(path), Path(path).stat().st_size
        media = MediaFileUpload(str(path), mimetype='application/octet-stream',
            chunksize=8*1024*1024, resumable=True)
        try:
            request = self.api.files().create(body={'name': uuid.uuid4().hex + '__' + Path(relative).name,
                'parents': [self.folder], 'appProperties': {'opendetect': FORMAT, 'path': relative}},
                media_body=media, fields=FIELDS)
            result = None
            while result is None:
                status, result = request.next_chunk(num_retries=4)
                if status:
                    print(f'SAVE WORKER {relative}: {100*status.progress():.0f}%', flush=True)
        finally:
            media.stream().close()
        record = {'id': result['id'], 'path': relative, 'sha256': checksum, 'size': size}
        self.verify(record)
        return record

    def sync(self, output, relay):
        relay.claim()  # existing lease: confirms this is still the only active producer
        if relay.generation != self.manifest['base_generation']:
            raise RuntimeError('A changed during training; worker files preserved')
        self.drain_deletions()
        old = {f['path']: f for f in self.manifest['files']}
        records = dict(old)  # completed folds omitted during restore remain referenced
        with tempfile.TemporaryDirectory(prefix='opendetect-stage-') as staging:
            changes = []
            for p in sorted(Path(output).rglob('*')):
                if not p.is_file() or p.name.endswith(('.writing', '.relay-part', '.relay-part.json')):
                    continue
                relative = p.relative_to(output).as_posix()
                safe_path(output, relative)
                target = Path(staging) / str(len(changes))
                shutil.copyfile(p, target)
                if relative in old and digest(target) == old[relative]['sha256']:
                    target.unlink()
                    continue
                changes.append((target, relative))
            if not changes:
                return
            self.check_space(sum(p.stat().st_size for p, _ in changes))
            for p, relative in changes:
                records[relative] = self.upload(p, relative)
            relay.claim()
            for record in records.values():
                self.verify(record)
            live = {f['id'] for f in records.values()}
            pending = [f for f in old.values() if f['id'] not in live]
            self.publish({**self.manifest, 'files': list(records.values()),
                'pending_delete': pending, 'revision': self.manifest['revision'] + 1})
        print('CLOUD COMMITTED TO WORKER:', self.scenario, self.manifest['revision'], flush=True)
        self.drain_deletions()

    def drain_deletions(self):
        from googleapiclient.errors import HttpError
        pending = self.manifest.get('pending_delete', [])
        if not pending:
            return
        if self.read_json(self.pointer) != self.manifest:
            raise RuntimeError('Worker manifest changed; cleanup stopped')
        for record in self.manifest['files']:
            self.verify(record)
        live = {f['id'] for f in self.manifest['files']}
        if any(f['id'] in live for f in pending):
            raise ValueError('Cleanup attempted to remove a live checkpoint')
        # Repeated cleanup after a disconnect is safe; never scan unrelated files.
        for record in pending:
            try:
                self.verify(record)
                self.api.files().delete(fileId=record['id']).execute()
            except HttpError as error:
                if error.resp.status != 404:
                    raise
                print('Cleanup ID already absent/inaccessible:', record['id'])
        self.publish({**self.manifest, 'pending_delete': []})

    def cleanup_completed(self, output, relay):
        from restore_plan import verified_completed
        completed = verified_completed(output, self.scenario)
        current = {f['path']: f for f in relay.files}
        local = {f['path']: f for f in self.manifest['files']}
        # Recheck A's current files, rather than treating an old receipt as backup proof.
        for name in completed:
            for path in ('save_model/'+name+'.pt', 'results/'+name+'.json',
                         'results/'+name+'.scores.npz', 'state/'+name+'.completed.json', 'GROUPED_CONFIG.json'):
                if path not in current or current[path]['sha256'] != local[path]['sha256']:
                    raise ValueError('Completed artifacts not present in A: ' + path)
                m = self.meta(current[path]['id'])
                if m.get('trashed') or m.get('sha256Checksum') != current[path]['sha256']:
                    raise ValueError('A artifact changed; worker files retained')
        remove = [f for f in local.values() if f['path'].startswith('resume_state/')
                  and f['path'].split('/')[1] in completed]
        if not remove:
            print('No completed resume files to clean; active slots retained.')
            return
        for f in remove:
            self.verify(f)
        ids = {f['id'] for f in remove}
        self.drain_deletions()
        self.publish({**self.manifest, 'files': [f for f in local.values() if f['id'] not in ids],
            'pending_delete': remove, 'revision': self.manifest['revision'] + 1})
        self.drain_deletions()
        print('Removed completed resume state from worker:', len(remove), 'files. Best models/results retained in worker and A.')


def connect(scenario, create=False):
    from google.colab import auth, userdata
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from googleapiclient.discovery import build
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    store = WorkerStore(build('drive', 'v3', credentials=credentials, cache_discovery=False),
        AuthorizedSession(credentials), scenario, create=create)
    relay = Relay(userdata.get('OPENDETECT_RELAY_URL'),
        userdata.get('OPENDETECT_WORKER_KEY'), scenario)
    relay.call('hello')
    relay.reader = store.reader
    return store, relay


def transfer(scenario, preview=False, cleanup_completed=True):
    """CPU only; existing scoped A upload service avoids switching Google accounts."""
    store, relay = connect(scenario)
    if not store.manifest or not store.manifest['files']:
        raise ValueError('No durable worker snapshot to transfer')
    files = store.manifest['files']
    print('TRANSFER:', len(files), 'files;', sum(f['size'] for f in files)/1e9, 'GB')
    if preview:
        for f in files:
            print(f['path'], f['size'])
        return
    relay.claim()
    try:
        store.bind(relay)
        with tempfile.TemporaryDirectory(prefix='opendetect-handoff-') as tmp:
            output = Path(tmp) / scenario
            store.restore(output, relay)
            from restore_plan import verified_completed
            completed = verified_completed(output, scenario)
            # Completed resume archives stay in B; ordinary artifacts and both active slots transfer.
            old = {f['path']: f['sha256'] for f in relay.files}
            changed = any(old.get(p.relative_to(output).as_posix()) != digest(p)
                for p in output.rglob('*') if p.is_file())
            if changed:
                relay.sync(output)
            expected = {f['path']: f['sha256'] for f in relay.files}
            relay.claim()
            if {f['path']: f['sha256'] for f in relay.files} != expected:
                raise RuntimeError('A snapshot readback differs; worker files retained')
            for f in relay.files:
                m = store.meta(f['id'])
                if m.get('trashed') or m.get('sha256Checksum') != f['sha256'] or int(m.get('size', -1)) != f['size']:
                    raise ValueError('A cloud verification failed; worker files retained')
            receipt = {'generation': relay.generation, 'worker_revision': store.manifest['revision'],
                'files': relay.files}
            # Receipt first; losing a response never authorizes deleting worker checkpoints.
            store.publish({**store.manifest, 'base_generation': relay.generation, 'receipt': receipt})
            if cleanup_completed:
                store.cleanup_completed(output, relay)
            print('UPLOAD VERIFIED TO A. Active worker checkpoints retained. A can now resume.')
    finally:
        relay.release()
