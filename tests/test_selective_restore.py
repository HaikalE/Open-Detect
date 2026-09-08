import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from relay_client import Relay, digest
from restore_plan import verified_completed


class SelectiveTests(unittest.TestCase):
    def fixture(self):
        cfg={'scenario':'C-1','folds':5,'dataset':'combined_USTC_mal','split':0,
             'base_seed':2022,'protocol':'exact-image-group-stratified-80-10-10-v1-preview','splits':list(range(5))}
        n='combined_USTC_mal_split_0_fold_0'
        objects={'GROUPED_CONFIG.json':json.dumps(cfg).encode(),
            'save_model/'+n+'.pt':b'best',
            'results/'+n+'.scores.npz':b'scores',
            'results/'+n+'.json':json.dumps({'split_identity':0,'protocol':cfg['protocol'],'fold':0,'split_seed':2022}).encode(),
            'resume_state/'+n+'/last.pt':b'completed resume',
            'resume_state/combined_USTC_mal_split_0_fold_1/last.pt':b'active resume',
            'resume_state/combined_USTC_mal_split_0_fold_1/last.backup.pt':b'fallback resume'}
        marker={'config':cfg}
        for key,p in [('checkpoint','save_model/'+n+'.pt'),('result','results/'+n+'.json'),('scores','results/'+n+'.scores.npz')]:
            marker[key+'_sha256']=hashlib.sha256(objects[p]).hexdigest()
        objects['state/'+n+'.completed.json']=json.dumps(marker).encode()
        r=Relay('https://script.google.com/macros/s/test/exec','x'*64,'C-1')
        r.files=[{'path':p,'id':str(i),'size':len(b),'sha256':hashlib.sha256(b).hexdigest()} for i,(p,b) in enumerate(objects.items())]
        return r,objects,n,cfg

    def restore(self,r,objects,d):
        fetched=[]
        def download(record,dest):
            fetched.append(record['path']);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(objects[record['path']])
        with patch.object(r,'download',side_effect=download):r.restore(d)
        return fetched

    def test_skip_only_verified_complete_keep_both_unfinished(self):
        r,objects,n,cfg=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            fetched=self.restore(r,objects,d)
            self.assertNotIn('resume_state/'+n+'/last.pt',fetched)
            self.assertEqual(sum(p.startswith('resume_state/') for p in fetched),2)
            self.assertIn('save_model/'+n+'.pt',fetched)
            # Execute the real pinned runner's completed-fold verifier as well.
            from run_grouped import verify_completed
            root=Path(d)
            verify_completed(root/'state'/(n+'.completed.json'),cfg,
                root/'save_model'/(n+'.pt'),root/'results'/(n+'.json'),
                root/'results'/(n+'.scores.npz'),0)

    def test_missing_marker_keeps_all_resume(self):
        r,objects,n,_=self.fixture()
        r.files=[f for f in r.files if not f['path'].startswith('state/')]
        with tempfile.TemporaryDirectory() as d:
            fetched=self.restore(r,objects,d)
            self.assertEqual(sum(p.startswith('resume_state/') for p in fetched),3)

    def test_corrupt_completed_artifact_stops_before_pruning(self):
        r,objects,n,_=self.fixture();objects['save_model/'+n+'.pt']=b'wrong'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'artifact'):self.restore(r,objects,d)

    def test_wrong_scenario_stops(self):
        r,objects,n,_=self.fixture();r.scenario='B-1'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'scenario'):self.restore(r,objects,d)

    def test_wrong_metric_identity_stops(self):
        r,objects,n,_=self.fixture()
        p='results/'+n+'.json';m=json.loads(objects[p]);m['split_seed']=999;objects[p]=json.dumps(m).encode()
        k='state/'+n+'.completed.json';marker=json.loads(objects[k]);marker['result_sha256']=hashlib.sha256(objects[p]).hexdigest();objects[k]=json.dumps(marker).encode()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'identity'):self.restore(r,objects,d)

    def test_sync_carries_remote_only_archives_after_new_client_claim(self):
        r,objects,n,_=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            self.restore(r,objects,d)
            child=Relay(r.url,r.key,r.scenario);child.files=list(r.files)
            (Path(d)/'unfinished.relay-part').write_bytes(b'never upload')
            with patch.object(child,'upload',side_effect=AssertionError('unexpected upload')),patch.object(child,'call',return_value={'generation':1}) as call:
                child.sync(d)
            committed=call.call_args.kwargs['files']
            self.assertEqual(committed,r.files)
            self.assertIn('resume_state/'+n+'/last.pt',[f['path'] for f in committed])

    def test_empty_snapshot_fresh_run(self):
        r=Relay('https://script.google.com/macros/s/test/exec','x'*64,'C-1')
        with tempfile.TemporaryDirectory() as d:r.restore(d)

if __name__=='__main__':unittest.main()
