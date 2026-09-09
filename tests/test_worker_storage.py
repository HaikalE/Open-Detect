import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from worker_store import WorkerStore, FORMAT, RESERVE
from build_worker_notebooks import build, update_notebook


class Call:
    def __init__(self, f):
        self.f = f
    def execute(self):
        return self.f()
    def next_chunk(self, **kwargs):
        return None, self.f()


class Drive:
    """In-memory API with actual upload bytes/hash/owner metadata and injected failures."""
    def __init__(self):
        self.data, self.records, self.events = {}, {}, []
        self.free = 20_000_000_000
        self.fail_upload = False
        self.fail_publish = False
        self.counter = 0
    def files(self):
        return self
    def about(self):
        return self
    def get(self, fileId=None, **kwargs):
        if fileId is None:
            return Call(lambda: {'user': {'emailAddress': 'b@test'},
                'storageQuota': {'limit': str(self.free), 'usage': '0'}})
        return Call(lambda: copy.deepcopy(self.records[fileId]))
    def list(self, q, **kwargs):
        folder = q.split("'")[1]
        return Call(lambda: {'files': [copy.deepcopy(x) for x in self.records.values()
            if folder in x.get('parents', []) and not x.get('trashed')]})
    def get_media(self, fileId):
        return Call(lambda: self.data[fileId])
    def put(self, identifier, body, media):
        if media:
            payload = media.getbytes(0, media.size())
            self.data[identifier] = payload
            body.update(size=str(len(payload)), sha256Checksum=hashlib.sha256(payload).hexdigest())
        self.records[identifier] = {**self.records.get(identifier, {}), **body,
            'id': identifier, 'owners': [{'emailAddress': 'b@test'}]}
        return copy.deepcopy(self.records[identifier])
    def create(self, body, media_body=None, **kwargs):
        def do():
            if self.fail_upload and media_body:
                raise RuntimeError('upload interrupted')
            self.counter += 1
            identifier = 'f' + str(self.counter)
            body.setdefault('parents', ['root'])
            body.setdefault('mimeType', 'application/octet-stream')
            return self.put(identifier, body, media_body)
        return Call(do)
    def update(self, fileId, media_body=None, **kwargs):
        def do():
            if self.fail_publish:
                raise RuntimeError('pointer update interrupted')
            self.events.append(('commit', fileId))
            return self.put(fileId, {}, media_body)
        return Call(do)
    def delete(self, fileId):
        def do():
            self.events.append(('delete', fileId))
            del self.records[fileId]
            self.data.pop(fileId, None)
            return {}
        return Call(do)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.api = Drive()
        self.store = WorkerStore(self.api, Mock(), 'A-2', create=True)
        self.relay = Mock(generation=34, files=[])
        self.store.bind(self.relay)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'GROUPED_CONFIG.json').write_text('{}')
        self.slot = self.root/'resume_state/run/last.pt'
        self.slot.parent.mkdir(parents=True)
        self.slot.write_bytes(b'epoch34')
    def tearDown(self):
        self.tmp.cleanup()
    def test_rolling_deletes_only_superseded_after_verified_commit(self):
        self.store.sync(self.root, self.relay)
        initial = copy.deepcopy(self.store.manifest)
        old = next(f['id'] for f in initial['files'] if f['path'].endswith('.pt'))
        orphan = self.api.create({'name': 'unrelated', 'parents': [self.store.folder]}).execute()['id']
        backup = self.slot.with_name('last.backup.pt')
        backup.write_bytes(b'epoch35')
        self.store.sync(self.root, self.relay)
        self.slot.write_bytes(b'epoch36')
        self.api.events.clear()
        self.store.sync(self.root, self.relay)
        self.assertNotIn(old, self.api.records)
        self.assertIn(orphan, self.api.records)
        self.assertEqual(len([f for f in self.store.manifest['files'] if f['path'].endswith('.pt')]), 2)
        self.assertEqual(self.api.events[0][0], 'commit')
        self.assertIn(('delete', old), self.api.events)
        restored = WorkerStore(self.api, Mock(), 'A-2')
        self.assertEqual(restored.manifest, self.store.manifest)
    def test_failed_upload_keeps_previous_manifest_and_objects(self):
        self.store.sync(self.root, self.relay)
        old = copy.deepcopy(self.store.manifest)
        self.slot.write_bytes(b'epoch35')
        self.api.fail_upload = True
        with self.assertRaisesRegex(RuntimeError, 'upload interrupted'):
            self.store.sync(self.root, self.relay)
        self.assertEqual(self.store.read_json(self.store.pointer), old)
        for f in old['files']:
            self.store.verify(f)
    def test_failed_pointer_does_not_delete_old_files(self):
        self.store.sync(self.root, self.relay)
        old = copy.deepcopy(self.store.manifest)
        self.slot.write_bytes(b'epoch35')
        self.api.fail_publish = True
        self.api.events.clear()
        with self.assertRaises(RuntimeError):
            self.store.sync(self.root, self.relay)
        self.assertFalse(any(e[0] == 'delete' for e in self.api.events))
        self.assertEqual(self.store.read_json(self.store.pointer), old)
    def test_quota_stops_before_upload(self):
        self.api.free = RESERVE
        before = len(self.api.records)
        with self.assertRaisesRegex(RuntimeError, 'low'):
            self.store.sync(self.root, self.relay)
        self.assertEqual(len(self.api.records), before)
    def test_stale_worker_history_refused(self):
        self.store.sync(self.root, self.relay)
        self.relay.generation = 35
        with self.assertRaisesRegex(RuntimeError, 'histories differ'):
            self.store.bind(self.relay)
    def test_lost_A_response_can_reconcile_exact_hashes(self):
        self.store.sync(self.root, self.relay)
        self.relay.files = [{**f, 'id': 'A_' + f['id']} for f in self.store.manifest['files']]
        self.relay.generation = 35
        self.store.bind(self.relay)
        self.assertEqual(self.store.manifest['base_generation'], 35)
    def test_fenced_worker_cannot_upload(self):
        self.relay.claim.side_effect = RuntimeError('Lease lost')
        before = len(self.api.records)
        with self.assertRaises(RuntimeError):
            self.store.sync(self.root, self.relay)
        self.assertEqual(len(self.api.records), before)
    def test_wrong_owner_prevents_restore(self):
        self.store.sync(self.root, self.relay)
        f = self.store.manifest['files'][0]
        self.api.records[f['id']]['owners'] = [{'emailAddress': 'other@test'}]
        with self.assertRaises(ValueError):
            self.store.verify(f)
    def test_cleanup_retry_after_delete_response_lost(self):
        from googleapiclient.errors import HttpError
        self.store.sync(self.root, self.relay)
        self.slot.write_bytes(b'epoch35')
        original = self.api.delete
        def lost_response(fileId):
            def do():
                original(fileId).execute()
                raise RuntimeError('response lost after deletion')
            return Call(do)
        with patch.object(self.api, 'delete', side_effect=lost_response):
            with self.assertRaisesRegex(RuntimeError, 'response lost'):
                self.store.sync(self.root, self.relay)
        self.assertTrue(self.store.manifest['pending_delete'])
        original_get = self.api.get
        def get(fileId=None, **kwargs):
            if fileId is not None and fileId not in self.api.records:
                def missing():
                    raise HttpError(Mock(status=404, reason='Not Found'), b'{}')
                return Call(missing)
            return original_get(fileId=fileId, **kwargs)
        with patch.object(self.api, 'get', side_effect=get):
            self.store.drain_deletions()
            self.store.drain_deletions()
        self.assertEqual(self.store.manifest['pending_delete'], [])
        for f in self.store.manifest['files']:
            self.store.verify(f)
    def test_cleanup_never_deletes_live_slot(self):
        self.store.sync(self.root, self.relay)
        self.store.publish({**self.store.manifest, 'pending_delete': [self.store.manifest['files'][0]]})
        self.api.events.clear()
        with self.assertRaisesRegex(ValueError, 'live checkpoint'):
            self.store.drain_deletions()
        self.assertFalse(any(e[0] == 'delete' for e in self.api.events))
    def test_cpu_upload_repeat_and_completed_cleanup_keep_active_slots(self):
        from relay_client import Relay
        from worker_store import transfer
        cfg = {'scenario':'A-2','folds':5,'dataset':'USTC','split':1,
            'base_seed':2022,'protocol':'exact-image-group-stratified-80-10-10-v1-preview','splits':list(range(5))}
        name = 'USTC_split_1_fold_0'
        data = {'GROUPED_CONFIG.json':json.dumps(cfg).encode(),
            'save_model/'+name+'.pt':b'best', 'results/'+name+'.scores.npz':b'scores',
            'results/'+name+'.json':json.dumps({'split_identity':0,'protocol':cfg['protocol'],
                'fold':0,'split_seed':2022}).encode(), 'resume_state/'+name+'/last.pt':b'completed'}
        marker = {'config':cfg}
        for key, p in [('checkpoint','save_model/'+name+'.pt'),('result','results/'+name+'.json'),('scores','results/'+name+'.scores.npz')]:
            marker[key+'_sha256'] = hashlib.sha256(data[p]).hexdigest()
        data['state/'+name+'.completed.json'] = json.dumps(marker).encode()
        for relative, payload in data.items():
            p = self.root/relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
        self.slot.with_name('last.backup.pt').write_bytes(b'active fallback')
        self.store.sync(self.root, self.relay)
        r = Relay('https://script.google.com/macros/s/test/exec','x'*64,'A-2')
        remote = {'generation':34, 'files':[]}
        commits = []
        def call(action, **kwargs):
            if action == 'claim':
                return {'lease':r.session, 'manifest':copy.deepcopy(remote)}
            if action == 'commit':
                remote.update(generation=remote['generation']+1, files=copy.deepcopy(kwargs['files']))
                commits.append(remote['generation'])
                return {'generation':remote['generation']}
            if action == 'release':
                return {}
            raise AssertionError(action)
        def download(record, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.api.data[record['id']])
        with patch('worker_store.connect', return_value=(self.store,r)), patch.object(r,'call',side_effect=call), \
                patch.object(r,'download',side_effect=download), patch.object(r,'upload',side_effect=self.store.upload):
            transfer('A-2')
            transfer('A-2')
        self.assertEqual(commits, [35])
        active = [f for f in self.store.manifest['files'] if f['path'].startswith('resume_state/')]
        self.assertEqual({f['path'] for f in active}, {'resume_state/run/last.pt', 'resume_state/run/last.backup.pt'})
        self.assertNotIn('torch', sys.modules)
    def test_edit_preserves_other_cells_and_is_idempotent(self):
        from build_relay_notebooks import build as original_build
        with tempfile.TemporaryDirectory() as d:
            original_build(d)
            original = json.loads((Path(d)/'OpenDetect_A-2_GROUPED.ipynb').read_text(encoding='utf-8'))
            changed = update_notebook(copy.deepcopy(original))
            self.assertEqual(changed['cells'][:9], original['cells'][:9])
            self.assertEqual(len(changed['cells']), len(original['cells'])+1)
            self.assertEqual(update_notebook(copy.deepcopy(changed)), changed)
    def test_notebooks_include_self_contained_cpu_cell(self):
        with tempfile.TemporaryDirectory() as d:
            build(d)
            notebooks = list(Path(d).glob('*.ipynb'))
            self.assertEqual(len(notebooks), 8)
            for p in notebooks:
                nb = json.loads(p.read_text(encoding='utf-8'))
                self.assertEqual(nb['metadata']['storage_transport'], FORMAT)
                self.assertEqual(nb['metadata']['source_commit'], 'c729403f7fd4d1207522e3ca793cafb5f8cb8eb0')
                last = ''.join(nb['cells'][-1]['source'])
                self.assertIn('PREVIEW_ONLY = False', last)
                self.assertIn('worker_store.py', last)
                self.assertIn('transfer(SCENARIO, preview=PREVIEW_ONLY', last)
                self.assertNotIn('verify_source()', last)
                for c in nb['cells']:
                    if c['cell_type'] == 'code':
                        ast.parse(''.join(c['source']))


class BoundaryTests(unittest.TestCase):
    def test_nested_runner_waits_for_cloud_ack_and_fails_on_disconnect(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'resume_support.py').write_text('def save_training(): pass\ndef publish_json(*a): pass\n')
            (p/'train.py').write_text('from resume_support import save_training\nsave_training()\nprint("NEXT_EPOCH", flush=True)\n')
            (p/'run_grouped.py').write_text('import subprocess,sys\np=subprocess.Popen([sys.executable, "' +
                str(p/'train.py').replace('\\', '/') + '"],stdout=subprocess.PIPE,text=True)\nfor line in p.stdout: print(line,end="",flush=True)\nraise SystemExit(p.wait())\n')
            wrapper = Path(__file__).resolve().parents[1]/'worker_entry.py'
            process = subprocess.Popen([sys.executable, str(wrapper), str(p/'run_grouped.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=dict(os.environ, OPENDETECT_BOUNDARY_TOKEN='test'))
            self.assertEqual(process.stdout.readline().strip(), 'WORKER_BOUNDARY:test')
            output, errors = process.communicate(input='', timeout=15)
            self.assertNotEqual(process.returncode, 0)
            self.assertNotIn('NEXT_EPOCH', output)
            self.assertIn('next epoch not started', errors)


if __name__ == '__main__':
    unittest.main()
