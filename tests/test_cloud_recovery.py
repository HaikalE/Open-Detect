import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cloud_recovery_workflow as w
from build_cloud_recovery import build


def metadata(epoch=50, digest='a'*64):
    split={'scenario':'C-1','seed':2024,'indices_sha256':'indices'}
    config={'scenario':'C-1','dataset':'combined_USTC_mal','splits':[{}, {}, split]}
    r={'format':2,'completed_epochs':epoch,'sha256':digest,'signature':{
        'experiment_signature':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'training_args':{'dset':'combined_USTC_mal','split':0,'fold':2,
            'recovery_identity':{'split':split}}}}
    return r,config


class CloudTests(unittest.TestCase):
    def test_identity_not_filename(self):
        r,c=metadata()
        self.assertEqual(w.identity(r),('C-1',2))
        self.assertTrue(w.config_matches(r,c))

    def test_identity_rejects_wrong_dataset_seed_fold_format(self):
        original,_=metadata()
        for mutation in ('dataset','seed','fold','format'):
            r=copy.deepcopy(original)
            if mutation=='dataset': r['signature']['training_args']['dset']='mal'
            if mutation=='seed': r['signature']['training_args']['recovery_identity']['split']['seed']=2025
            if mutation=='fold': r['signature']['training_args']['fold']=99
            if mutation=='format': r['format']=1
            with self.assertRaises(ValueError): w.identity(r)

    def test_changed_config_rejected(self):
        r,c=metadata();c['other_code']='changed'
        self.assertFalse(w.config_matches(r,c))

    def setup_plan(self, canonical_epoch=52, canonical_digest='b'*64, completed=False):
        r,c=metadata()
        canonical,_=metadata(canonical_epoch,canonical_digest)
        def item(i,n,**kw): return dict(id=i,name=n,owners=[{'emailAddress':'a'}],**kw)
        folder={w.OUTPUTS_ID:{'C-1':item('output','C-1')},
            'output':{n:item(n,n) for n in ('GROUPED_CONFIG.json','state','resume_state')},
            'state':{},'resume_state':{'combined_USTC_mal_split_0_fold_2':item('target','combined_USTC_mal_split_0_fold_2')},
            'target':{'last.backup.pt':item('canon','last.backup.pt',sha256Checksum=canonical_digest),
                      'last.backup.pt.json':item('canonjson','last.backup.pt.json')}}
        if completed: folder['state']['combined_USTC_mal_split_0_fold_2.completed.json']=item('done','done')
        docs={'backupjson':r,'GROUPED_CONFIG.json':c,'canonjson':canonical}
        records={'backuppayload':item('backuppayload','last.pt',sha256Checksum=r['sha256'])}
        receipt={'files':[{'name':'last.pt','backup_id':'backuppayload'},
                          {'name':'last.pt.json','backup_id':'backupjson'}]}
        return folder,docs,records,receipt

    def plan(self, epoch=52, digest='b'*64, completed=False):
        folders,docs,records,receipt=self.setup_plan(epoch,digest,completed)
        with patch.object(w,'require_A',return_value='a'),patch.object(w,'unique_names',side_effect=lambda a,i:folders[i]),patch.object(w,'read_json',side_effect=lambda a,i,*x:docs[i]),patch.object(w,'meta',side_effect=lambda a,i:records[i]):
            return w.recovery_plan(None,receipt)

    def test_newer_a_kept(self):
        r=self.plan()['runs'][0]
        self.assertEqual(r['decision'],'KEEP_A')
        self.assertEqual(r['next_epoch'],53)
        self.assertIn('C-1/resume_state/combined_USTC_mal_split_0_fold_2',r['destination'])

    def test_newer_backup_install_other_slot(self):
        r=self.plan(49)['runs'][0]
        self.assertEqual(r['decision'],'INSTALL_AVAILABLE')
        self.assertEqual(r['slot'],'last.pt') # preserve last.backup.pt epoch49

    def test_equal_epoch_divergence_stops(self):
        self.assertEqual(self.plan(50)['runs'][0]['decision'],'STOP')

    def test_equal_identical_kept(self):
        self.assertEqual(self.plan(50,'a'*64)['runs'][0]['decision'],'KEEP_A')

    def test_epoch_100_routes_to_evaluation(self):
        r=self.plan(100)['runs'][0]
        self.assertIsNone(r['next_epoch'])
        self.assertEqual(r['next_step'],'evaluation_then_next_fold')

    def test_divergent_backup_latest_epoch_stops(self):
        folders,docs,records,receipt=self.setup_plan()
        docs['otherjson']=copy.deepcopy(docs['backupjson']);docs['otherjson']['sha256']='c'*64
        records['otherpt']=dict(id='otherpt',owners=[{'emailAddress':'a'}],sha256Checksum='c'*64)
        receipt['files'] += [dict(name='last.pt',backup_id='otherpt'),dict(name='last.pt.json',backup_id='otherjson')]
        with patch.object(w,'require_A',return_value='a'),patch.object(w,'unique_names',side_effect=lambda a,i:folders[i]),patch.object(w,'read_json',side_effect=lambda a,i,*x:docs[i]),patch.object(w,'meta',side_effect=lambda a,i:records[i]):
            self.assertEqual(w.recovery_plan(None,receipt)['runs'][0]['decision'],'STOP')

    def test_completed_never_installed(self):
        self.assertEqual(self.plan(completed=True)['runs'][0]['decision'],'STOP')

    def test_install_requires_confirmation_before_writes(self):
        with patch.object(w,'require_A',return_value='a'):
            with self.assertRaises(ValueError): w.install_one(None,{},'C-1',2,'','')

    def exercise_install(self, fail_publication=False):
        folders,docs,records,receipt=self.setup_plan(49)
        for group in folders.values():
            for m in group.values(): records[m['id']]=m
        records['backupjson']={'id':'backupjson','name':'last.pt.json','owners':[{'emailAddress':'a'}]}
        for r in records.values(): r.update(size='1',md5Checksum='hash',version='1',modifiedTime='t')
        class Request:
            def __init__(self, fn): self.fn=fn
            def execute(self): return self.fn()
        class API:
            def files(self): return self
            def copy(self,fileId,body,**kw):
                def run():
                    r=copy.deepcopy(records[fileId]);r.update(body,id='copy'+fileId)
                    records[r['id']]=r; folders['target'][r['name']]=r
                    if fileId in docs: docs[r['id']]=copy.deepcopy(docs[fileId])
                    return copy.deepcopy(r)
                return Request(run)
            def update(self,fileId,body):
                def run():
                    if fail_publication and body['name']=='last.pt.json': raise RuntimeError('simulated disconnect')
                    r=records[fileId];del folders['target'][r['name']]
                    r.update(body);folders['target'][r['name']]=r
                return Request(run)
        with patch.object(w,'require_A',return_value='a'),patch.object(w,'unique_names',side_effect=lambda a,i:copy.deepcopy(folders[i])),patch.object(w,'read_json',side_effect=lambda a,i,*x:copy.deepcopy(docs[i])),patch.object(w,'meta',side_effect=lambda a,i:copy.deepcopy(records[i])):
            if fail_publication:
                with self.assertRaises(RuntimeError):
                    w.install_one(API(),receipt,'C-1',2,w.STOP_PHRASE,'PASANG CHECKPOINT TERVERIFIKASI KE A')
            else:
                w.install_one(API(),receipt,'C-1',2,w.STOP_PHRASE,'PASANG CHECKPOINT TERVERIFIKASI KE A')
                self.assertEqual(w.recovery_plan(API(),receipt)['runs'][0]['decision'],'KEEP_A')
        self.assertEqual(folders['target']['last.backup.pt']['id'],'canon')
        self.assertEqual(docs['canonjson']['completed_epochs'],49)
        return folders

    def test_install_publish_and_idempotent_plan(self):
        folders=self.exercise_install()
        self.assertIn('last.pt.json',folders['target'])

    def test_interrupted_install_keeps_previous_valid_slot(self):
        self.exercise_install(True)

    def test_notebook_no_upload_no_gpu_default_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            nb=build(Path(d)/'recovery.ipynb')
        source='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertNotIn('files.upload(',source)
        self.assertNotIn('files.download(',source)
        self.assertNotIn('torch.load(',source)
        self.assertIn('ACTION = "PLAN_A"',source)
        self.assertNotIn('accelerator',nb['metadata'])
        for c in nb['cells']:
            if c['cell_type']=='code': ast.parse(''.join(c['source']))


if __name__=='__main__': unittest.main()
