"""Pause science subprocesses at durable boundaries; parent owns Drive credentials."""
import os
from pathlib import Path
import runpy
import sys


def boundary():
    token = os.environ['OPENDETECT_BOUNDARY_TOKEN']
    print('WORKER_BOUNDARY:' + token, flush=True)
    if sys.stdin.readline().strip() != 'WORKER_ACK:' + token:
        raise RuntimeError('Worker Drive save failed/disconnected; next epoch not started')


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    if script.name == 'train.py':
        import resume_support
        original = resume_support.save_training
        def save(*args, **kwargs):
            result = original(*args, **kwargs)
            boundary()
            return result
        resume_support.save_training = save
    elif script.name == 'run_grouped.py':
        import resume_support
        publish = resume_support.publish_json
        def publish_json(path, data):
            publish(path, data)
            if str(path).endswith('.completed.json'):
                boundary()
        resume_support.publish_json = publish_json
        import subprocess
        original = subprocess.Popen
        def popen(command, *args, **kwargs):
            if isinstance(command, list) and len(command) > 1 and Path(command[1]).name in ('train.py', 'test.py'):
                command = [command[0], str(Path(__file__).resolve()), *command[1:]]
            return original(command, *args, **kwargs)
        subprocess.Popen = popen
    elif script.name != 'test.py':
        raise ValueError('Unsupported science entrypoint')
    runpy.run_path(str(script), run_name='__main__')
    boundary()


if __name__ == '__main__':
    main()
