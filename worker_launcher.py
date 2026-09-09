"""Parent Colab process retains Google authentication and acknowledges cloud saves."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import uuid


def run(store, relay, python, repo, transport, hours, environment):
    relay.claim()
    process = None
    try:
        store.bind(relay)
        store.check_space(600_000_000)
        # Each attempt restores verified cloud state; failed local attempts are preserved.
        work = Path(tempfile.mkdtemp(prefix='opendetect-training-'))
        output = work / store.scenario
        store.restore(output, relay)
        if (output / 'GROUPED_CONFIG.json').exists():
            store.sync(output, relay)  # migrate A snapshot before consuming GPU epochs
        if shutil.disk_usage(work).free < 3_000_000_000:
            raise RuntimeError('Local disk low; need 3 GB checkpoint/staging headroom')
        token = uuid.uuid4().hex
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(transport), OPENDETECT_BOUNDARY_TOKEN=token)
        command = [python, str(Path(transport)/'worker_entry.py'), str(Path(repo)/'run_grouped.py'),
            '--scenario', store.scenario, '--output', str(output), '--session_hours', str(hours),
            '--num_workers', '2', '--gpu', '0']
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        for line in process.stdout:
            if line.strip() == 'WORKER_BOUNDARY:' + token:
                store.sync(output, relay)
                if shutil.disk_usage(work).free < 3_000_000_000:
                    raise RuntimeError('Local disk reserve reached; checkpoint is on worker Drive')
                process.stdin.write('WORKER_ACK:' + token + '\n')
                process.stdin.flush()
            else:
                print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Trainer stopped; worker cloud checkpoint and local files retained at ' + str(output))
        shutil.copyfile(environment, output/'environment.txt')
        store.sync(output, relay)
        print((output/'SESSION_STATUS.json').read_text())
        print('Continue on this worker, or use the final CPU cell for handoff to A.')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
        # Same-scenario handoff remains manual: upload B before starting a different worker.
        relay.release()
