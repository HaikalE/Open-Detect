"""Conservative transfer-only pruning; pinned runner verifies again before skip.

Only resume_state directories of completed, locally hash-verified folds are
omitted. Never prune best models, metrics/scores, unfinished slots, or cloud data.
"""
import json
from pathlib import Path
from relay_client import digest


def verified_completed(output, scenario):
    output=Path(output)
    config_path=output/'GROUPED_CONFIG.json'
    if not config_path.exists():
        if any((output/'state').glob('*.completed.json')):
            raise ValueError('Completion marker without grouped configuration')
        return set()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('scenario')!=scenario or config.get('folds')!=5 or
        config.get('protocol')!='exact-image-group-stratified-80-10-10-v1-preview'):
        raise ValueError('Restore configuration scenario/folds mismatch')
    dataset=config['dataset'];split=config['split']
    if not isinstance(dataset,str) or not dataset.replace('_','').isalnum() or type(split) is not int:
        raise ValueError('Invalid grouped identity')
    names=[f'{dataset}_split_{split}_fold_{fold}' for fold in range(5)]
    actual={p.name for p in (output/'state').glob('*.completed.json')}
    if actual-{name+'.completed.json' for name in names}:
        raise ValueError('Unexpected completion marker; no speculative pruning')
    completed=set()
    for fold,name in enumerate(names):
        marker=output/'state'/(name+'.completed.json')
        if not marker.exists():continue
        record=json.loads(marker.read_text(encoding='utf-8'))
        if record['config']!=config:
            raise ValueError('Completed run config mismatch')
        for key,relative in [('checkpoint','save_model/'+name+'.pt'),
                             ('result','results/'+name+'.json'),
                             ('scores','results/'+name+'.scores.npz')]:
            if digest(output/relative)!=record[key+'_sha256']:
                raise ValueError('Completed artifact changed: '+relative)
        metrics=json.loads((output/'results'/(name+'.json')).read_text(encoding='utf-8'))
        if (metrics['split_identity']!=config['splits'][fold] or
            metrics['protocol']!=config['protocol'] or
            metrics['fold']!=fold or metrics['split_seed']!=config['base_seed']+fold):
            raise ValueError('Completed result identity mismatch')
        completed.add(name)
    return completed
