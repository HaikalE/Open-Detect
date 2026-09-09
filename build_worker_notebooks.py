"""Change only training code cell 9; append one self-contained CPU transfer cell."""
import json
from pathlib import Path
import sys

import build_grouped_colabs as base
from build_relay_notebooks import build as build_relay

ROOT = Path(__file__).resolve().parent
MODULES = ('relay_client.py', 'drive_reader.py', 'restore_plan.py',
           'worker_store.py', 'worker_entry.py', 'worker_launcher.py')


def bootstrap(transport):
    return f'''from pathlib import Path
import sys
TRANSPORT = {transport!r}
TRANSPORT_DIR = Path('/content/opendetect_deferred_transport')
TRANSPORT_DIR.mkdir(exist_ok=True)
for name, source in TRANSPORT.items():
    (TRANSPORT_DIR/name).write_text(source, encoding='utf-8')
sys.path.insert(0, str(TRANSPORT_DIR))
for name in TRANSPORT:
    sys.modules.pop(name[:-3], None)
'''


def update_notebook(nb):
    """Preserve all existing cells except the training cell, including user outputs."""
    scenario = None
    import ast
    for node in ast.parse(''.join(nb['cells'][1]['source'])).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCENARIO' for t in node.targets):
            scenario = ast.literal_eval(node.value)
    if scenario not in base.SCENARIOS or nb['cells'][9]['id'] != 'grouped-9':
        raise ValueError('Unexpected notebook layout/scenario')
    transport = {name: (ROOT/name).read_text(encoding='utf-8') for name in MODULES}
    training = '''# @title Training: dua slot di Drive worker, upload ke A manual
# Cell ini menggantikan alur penyimpanan relay per epoch pada pengantar lama.
# Akun yang diautentikasi menyimpan checkpoint di My Drive/OpenDetect_WORKER_<scenario>.
# Versi checkpoint tergantikan dihapus SETELAH cloud baru terverifikasi.
# Dua slot aktif + best-model tetap disimpan. Jangan pindah worker sebelum upload A.
verify_source()
''' + bootstrap(transport) + '''
from worker_store import connect
from worker_launcher import run
STORE, WORKER_RELAY = connect(SCENARIO, create=True)
run(STORE, WORKER_RELAY, PYTHON, REPO, TRANSPORT_DIR, SESSION_HOURS,
    PACKAGE/('environment_'+SCENARIO+'.txt'))
'''
    nb['cells'][9] = base.cell('code', training, 9)
    transfer = f'''# @title Upload ke A (CPU; cell ini bisa dijalankan sendiri)
SCENARIO = {scenario!r}
PREVIEW_ONLY = False # @param {{type:'boolean'}}
CLEAN_COMPLETED_AFTER_UPLOAD = True # @param {{type:'boolean'}}
# Hentikan cell training dahulu; lease relay menolak runtime training yang masih aktif.
# Upload dua slot fold aktif + best-model, hasil, config, dan log ke A.
# Cleanup hanya resume fold SELESAI setelah hasil A diverifikasi ulang.
# Best-model, hasil, dan dua slot fold aktif tetap disimpan.
# Ganti runtime GPU ke CPU boleh menghapus /content; cell ini membaca Drive worker.
# Tidak perlu menjalankan setup GPU, download dataset, atau cell training lagi.
''' + bootstrap(transport) + '''
from worker_store import transfer
import shutil
import subprocess
has_gpu = bool(shutil.which('nvidia-smi')) and subprocess.run(
    ['nvidia-smi', '-L'], capture_output=True).returncode == 0
if has_gpu:
    print('Upload dilewati saat runtime GPU. Ganti ke CPU lalu jalankan cell ini saja.')
else:
    transfer(SCENARIO, preview=PREVIEW_ONLY, cleanup_completed=CLEAN_COMPLETED_AFTER_UPLOAD)
'''
    existing = [i for i, c in enumerate(nb['cells']) if c.get('id') == 'deferred-upload']
    cell = base.cell('code', transfer, 10)
    cell['id'] = 'deferred-upload'
    if existing:
        if len(existing) != 1:
            raise ValueError('Duplicate upload cells')
        nb['cells'][existing[0]] = cell
    else:
        nb['cells'].append(cell)
    nb['metadata']['storage_transport'] = 'worker-drive-v1'
    return nb


def build(output):
    output = Path(output)
    build_relay(output)
    for scenario in base.SCENARIOS:
        path = output/f'OpenDetect_{scenario}_GROUPED.ipynb'
        nb = update_notebook(json.loads(path.read_text(encoding='utf-8')))
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        original = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
        Path(sys.argv[1]).write_text(json.dumps(update_notebook(original), indent=1,
                                               ensure_ascii=False), encoding='utf-8')
    else:
        build(sys.argv[1])
