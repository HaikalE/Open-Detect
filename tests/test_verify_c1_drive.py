import hashlib
import json
import unittest
from verify_c1_drive import verify_c1


class VerifyTests(unittest.TestCase):
    def run_check(self, epoch):
        config={'scenario':'C-1','dataset':'combined_USTC_mal',
                'splits':[{'seed':2022},{'seed':2023},{'seed':2024}]}
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        records={}; blobs={}; folders={}
        def add(folder,name,value=None):
            identifier=folder+'/'+name
            r={'id':identifier,'name':name,'sha256Checksum':'hash'}
            if value is not None:
                payload=json.dumps(value).encode();blobs[identifier]=payload
                r.update(size=str(len(payload)),md5Checksum=hashlib.md5(payload).hexdigest())
            records[identifier]=r;folders.setdefault(folder,[]).append(r)
            return identifier
        output='1cE5XtQJnSnzdzF60wDM9K3Iw5atD5C2N'
        records[output]={'parents':['1o8-O578YTfc9FAwGKBZvDGJlzUYE7awI']}
        add(output,'GROUPED_CONFIG.json',config)
        ids={n:add(output,n) for n in ('resume_state','state','save_model','results')}
        slot=add(ids['resume_state'],'combined_USTC_mal_split_0_fold_2')
        for n in ('last.pt','last.backup.pt'):
            add(slot,n)
            add(slot,n+'.json',{'sha256':'hash','completed_epochs':epoch,'signature':{
                'experiment_signature':config_hash,'training_args':{'fold':2,
                'recovery_identity':{'split':config['splits'][2]}}}})
        for f in (0,1):
            n=f'combined_USTC_mal_split_0_fold_{f}'
            add(ids['state'],n+'.completed.json',{'config':config,'checkpoint_sha256':'hash',
                'result_sha256':'hash','scores_sha256':'hash'})
            add(ids['save_model'],n+'.pt')
            add(ids['results'],n+'.json',{'split_identity':config['splits'][f]})
            add(ids['results'],n+'.scores.npz')
        class API:
            def files(self): return self
            def get_media(self,fileId):
                class Request:
                    def execute(self): return blobs[fileId]
                return Request()
        return verify_c1(API(),lambda api,i:records[i],lambda r:'A',lambda api:'A',lambda api,i:folders[i])
    def test_higher_epoch_preserved(self):
        r=self.run_check(54)
        self.assertEqual(r['next_epoch'],55)
        self.assertFalse(r['writes_performed'])
    def test_lower_epoch_requires_review(self):
        r=self.run_check(49)
        self.assertTrue(r['decision'].startswith('STOP'))
        self.assertFalse(r['writes_performed'])

if __name__=='__main__': unittest.main()
