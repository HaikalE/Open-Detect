import ast
import json
from pathlib import Path
import sys

path=Path(sys.argv[1])
nb=json.loads(path.read_text(encoding='utf-8'))
nb['cells'][0]['source'].insert(0,'# PERIKSA RESUME C-1 DI A\n\nRun all sebagai A, CPU cukup. State canonical ternyata sudah ada: jangan timpa dengan epoch 50 sebelum diperiksa. Tidak training, tidak menghapus, tidak mengubah hasil. Laporan akan diunduh.\n\n')
nb['cells'][1]['source']=["PHASE = 'VERIFY_C1_A'\n"]
helper=Path(__file__).with_name('verify_c1_drive.py').read_text(encoding='utf-8')
nb['cells'][3]['source'].append('\n'+helper+'\n')
code=''.join(nb['cells'][5]['source'])
code=code.replace("if PHASE == 'PREPARE_B':", "if PHASE == 'VERIFY_C1_A':\n    report = verify_c1(API, meta, owner, require_A, children)\n    export_json('OpenDetect_C1_A_resume_check.json', report)\nelif PHASE == 'PREPARE_B':",1)
nb['cells'][5]['source']=code.splitlines(True)
for cell in nb['cells']:
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']))
        cell['outputs']=[];cell['execution_count']=None
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
