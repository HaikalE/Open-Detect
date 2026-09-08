"""Read-only canonical C-1 verification before any restore; prevent epoch rollback."""
import hashlib
import json


def verify_c1(api, meta, owner, require_A, children):
    a = require_A(api)
    def small(identifier):
        m=meta(api,identifier)
        if m.get('trashed') or int(m.get('size',9999999)) > 262144:
            raise ValueError('Not a small live metadata file')
        data=api.files().get_media(fileId=identifier).execute()
        if len(data)!=int(m['size']) or hashlib.md5(data).hexdigest()!=m['md5Checksum']:
            raise ValueError('Metadata changed or checksum mismatch')
        return json.loads(data)
    def names(folder):
        found={}
        for item in children(api,folder):
            if item['name'] in found:
                raise ValueError('Duplicate canonical name: '+item['name'])
            found[item['name']]=item
        return found
    def matches(record,expected):
        return not record.get('trashed') and record.get('sha256Checksum')==expected
    output='1cE5XtQJnSnzdzF60wDM9K3Iw5atD5C2N'
    root=meta(api,output)
    if owner(root)!=a or '1o8-O578YTfc9FAwGKBZvDGJlzUYE7awI' not in root.get('parents',[]):
        raise ValueError('Wrong canonical output folder')
    out=names(output)
    config=small(out['GROUPED_CONFIG.json']['id'])
    assert config['scenario']=='C-1' and config['dataset']=='combined_USTC_mal'
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    resume=names(out['resume_state']['id'])
    slots=names(resume['combined_USTC_mal_split_0_fold_2']['id'])
    report={'format':'opendetect-c1-canonical-check-v1','writes_performed':False,
            'slots':[], 'completed_folds':[], 'backup_epoch':50}
    for name in ('last.pt','last.backup.pt'):
        entry={'name':name}
        try:
            record=small(slots[name+'.json']['id'])
            signature=record['signature']
            identity=signature['training_args']['recovery_identity']['split']
            ok=(matches(slots[name],record['sha256']) and
                signature['experiment_signature']==config_hash and
                identity==config['splits'][2] and identity['seed']==2024 and
                signature['training_args']['fold']==2)
            entry.update(epoch=record['completed_epochs'],hash_and_identity_valid=ok,
                         checkpoint_id=slots[name]['id'],metadata_id=slots[name+'.json']['id'])
        except Exception as e:
            entry['error']=str(e)
        report['slots'].append(entry)
    states=names(out['state']['id']); models=names(out['save_model']['id']); results=names(out['results']['id'])
    for fold in (0,1):
        base=f'combined_USTC_mal_split_0_fold_{fold}'
        marker=small(states[base+'.completed.json']['id'])
        metrics=small(results[base+'.json']['id'])
        ok=marker['config']==config and metrics['split_identity']==config['splits'][fold]
        for key,item in [('checkpoint',models[base+'.pt']),('result',results[base+'.json']),('scores',results[base+'.scores.npz'])]:
            ok=ok and matches(item,marker[key+'_sha256'])
        report['completed_folds'].append({'fold':fold,'hash_and_identity_valid':ok})
    valid=[s['epoch'] for s in report['slots'] if s.get('hash_and_identity_valid')]
    report['latest_verified_epoch']=max(valid) if valid else None
    if valid and max(valid)>=50 and all(f['hash_and_identity_valid'] for f in report['completed_folds']):
        report['decision']='KEEP_CANONICAL_NO_RESTORE_NEEDED; model load/RNG restore still untested'
        report['next_epoch']=max(valid)+1
    else:
        report['decision']='STOP_RESTORE_REQUIRES_REVIEW; nothing overwritten'
    print(json.dumps(report,indent=2))
    return report
