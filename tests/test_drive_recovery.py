import copy
import ast
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import drive_recovery as d


class Request:
    def __init__(self, fn): self.fn = fn
    def execute(self): return self.fn()


class Fake:
    def __init__(self):
        self.email = 'b@example.com'
        self.records = {
            d.PROJECT_ID: {'id': d.PROJECT_ID, 'owners': [{'emailAddress':'a@example.com'}]},
            'folder': {'id':'folder', 'parents':[d.PROJECT_ID], 'owners':[{'emailAddress':'a@example.com'}]},
            'source': {'id':'source','name':'last.pt','size':'12','md5Checksum':'abc',
                       'version':'1','modifiedTime':'before','owners':[{'emailAddress':'b@example.com'}]},
            'backup': {'id':'backup','name':'source__last.pt','size':'12','md5Checksum':'abc',
                       'parents':['folder'],'owners':[{'emailAddress':'a@example.com'}]},
        }
        self.events = []
    def files(self): return self
    def about(self): return self
    def get(self, fileId=None, **kw):
        return Request(lambda: copy.deepcopy(self.records[fileId]) if fileId else
                       {'user':{'emailAddress':self.email}, 'storageQuota':{'limit':'99999999999','usage':'0'}})
    def update(self, fileId, body):
        def run():
            self.events.append(('trash',fileId))
            self.records[fileId].update(body, version='2', modifiedTime='after-trash')
        return Request(run)
    def delete(self, fileId):
        def run():
            self.events.append(('delete',fileId))
            del self.records[fileId]
        return Request(run)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.api = Fake()
        self.receipt = {'owner_A':'a@example.com','owner_B':'b@example.com',
            'backup_folder':'folder','files':[{**copy.deepcopy(self.api.records['source']), 'backup_id':'backup'}]}
    def clean(self, **kw):
        options = dict(selected_ids=['source'], stopped=d.STOP_PHRASE,
            confirmation='PINDAHKAN FILE TERVERIFIKASI KE SAMPAH')
        options.update(kw)
        return d.cleanup_B(self.api,self.receipt,**options)
    def test_wrong_training_account(self):
        with self.assertRaises(RuntimeError): d.require_A(self.api)
    def test_confirmations_required(self):
        for kw in ({'stopped':''},{'confirmation':''},{'selected_ids':[]},{'selected_ids':['other']}):
            with self.assertRaises((RuntimeError,ValueError)): self.clean(**kw)
        self.assertEqual(self.api.events,[])
    def test_changed_content_protected(self):
        self.api.records['source']['md5Checksum']='changed'
        with self.assertRaises(ValueError): self.clean()
        self.assertEqual(self.api.events,[])
    def test_wrong_backup_owner_protected(self):
        self.api.records['backup']['owners']=[{'emailAddress':'b@example.com'}]
        with self.assertRaises(ValueError): self.clean()
        self.assertEqual(self.api.events,[])
    def test_wrong_source_owner_protected(self):
        self.api.records['source']['owners']=[{'emailAddress':'a@example.com'}]
        with self.assertRaises(ValueError): self.clean()
        self.assertEqual(self.api.events,[])
    def test_backup_size_and_location_protected(self):
        for key,value in [('size','13'),('parents',['elsewhere']),('trashed',True)]:
            with patch.dict(self.api.records['backup'],{key:value}):
                with self.assertRaises(ValueError): self.clean()
        self.assertEqual(self.api.events,[])
    def test_permanent_requires_trash(self):
        with self.assertRaises(ValueError): self.clean(permanent=True,confirmation='HAPUS PERMANEN FILE TERVERIFIKASI')
        self.assertEqual(self.api.events,[])
    def test_trash_then_permanent(self):
        self.clean()
        self.clean(permanent=True,confirmation='HAPUS PERMANEN FILE TERVERIFIKASI')
        self.assertEqual(self.api.events,[('trash','source'),('delete','source')])
        self.assertIn('backup',self.api.records)
    def test_validate_all_before_first_delete(self):
        r=copy.deepcopy(self.api.records['source']); r['id']='second'
        self.api.records['second']=r
        self.receipt['files'].append({**r,'backup_id':'backup'})
        self.api.records['second']['md5Checksum']='changed'
        with self.assertRaises(ValueError): self.clean(selected_ids=['source','second'])
        self.assertEqual(self.api.events,[])
    def test_backup_refuses_active_producer(self):
        with self.assertRaises(RuntimeError): d.backup_A(self.api,{},'')
    def test_backup_refuses_wrong_account(self):
        with self.assertRaises(RuntimeError): d.backup_A(self.api,{'files':[]},d.STOP_PHRASE)
    def test_generated_notebooks(self):
        import build_storage_notebooks as b
        with tempfile.TemporaryDirectory(prefix='opendetect_notebook_qa_') as tmp:
            b.build(tmp)
            notebooks=list(Path(tmp).glob('*.ipynb'))
            self.assertEqual(len(notebooks),9)
            for path in notebooks:
                nb=json.loads(path.read_text(encoding='utf-8'))
                for cell in nb['cells']:
                    if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
                if 'RECOVERY' not in path.name:
                    setup=''.join(nb['cells'][3]['source'])
                    self.assertLess(setup.index('require_A(API,'),setup.index("drive.mount("))
                    self.assertLess(setup.index('verify_mount(API, PROJECT)'),setup.index('PACKAGE.mkdir('))
                    self.assertTrue(''.join(nb['cells'][-1]['source']).startswith('require_A(API,'))
                    self.assertEqual(nb['metadata']['source_commit'],b.TRAIN_COMMIT)
                else:
                    config=''.join(nb['cells'][1]['source'])
                    self.assertIn("CLEANUP_ACTION = 'PREVIEW'",config)
                    self.assertIn('SHARE_AND_EXPORT = False',config)


if __name__ == '__main__': unittest.main()
