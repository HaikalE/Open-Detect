"""External transport wrapper. Scientific repository files remain byte-identical."""
import os
from pathlib import Path
import runpy
import sys
from relay_client import from_env


def main():
    script=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
    sys.path.insert(0,str(script.parent))
    relay=from_env();relay.claim()
    output=Path(os.environ['OPENDETECT_LOCAL_OUTPUT'])
    if script.name=='train.py':
        import resume_support
        original=resume_support.save_training
        def save(*args,**kwargs):
            result=original(*args,**kwargs)
            relay.sync(output) # block next epoch until cloud verification succeeds
            return result
        resume_support.save_training=save
        runpy.run_path(str(script),run_name='__main__')
        relay.sync(output)
    elif script.name=='run_grouped.py':
        # runpy loads __main__ separately; patch subprocess command construction instead.
        import subprocess
        original=subprocess.Popen
        wrapper=str(Path(__file__).resolve())
        def popen(command,*args,**kwargs):
            if isinstance(command,list) and len(command)>1 and Path(command[1]).name in ('train.py','test.py'):
                command=[command[0],wrapper,*command[1:]]
            return original(command,*args,**kwargs)
        subprocess.Popen=popen
        runpy.run_path(str(script),run_name='__main__')
        relay.claim() # refresh generation after child synchronizations
        relay.sync(output)
    elif script.name=='test.py':
        runpy.run_path(str(script),run_name='__main__')
        relay.sync(output)
    else:
        raise ValueError('Unsupported science entrypoint')

if __name__=='__main__': main()
