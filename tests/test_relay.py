import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from relay_client import Relay, safe_path
from build_relay_notebooks import build


class RelayTests(unittest.TestCase):
    def client(self): return Relay('https://script.google.com/macros/s/EXAMPLE/exec','x'*64,'C-1','a'*32)
    def test_url_and_key_fail_closed(self):
        for url in ('','http://script.google.com/x/exec','https://evil.test/x/exec','https://script.google.com/x/dev'):
            with self.assertRaises(ValueError): Relay(url,'x'*64,'C-1')
        with self.assertRaises(ValueError): Relay('https://script.google.com/x/exec','','C-1')

    def test_paths_cannot_escape(self):
        for path in ('../a','/a','a/../../b','a\\b','a//b','.secret','a/.secret'):
            with self.assertRaises(ValueError): safe_path('/tmp/test',path)

    def test_failed_commit_does_not_advance_generation(self):
        r=self.client();r.generation=9
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'GROUPED_CONFIG.json').write_text('{}')
            def call(action,**kwargs): raise RuntimeError('simulated cloud failure')
            with patch.object(r,'upload',return_value={'path':'GROUPED_CONFIG.json','id':'f','size':2,'sha256':'h'}),patch.object(r,'call',side_effect=call):
                with self.assertRaises(RuntimeError):r.sync(d)
        self.assertEqual(r.generation,9)
        self.assertEqual(r.files,[])

    def test_unchanged_files_not_reuploaded(self):
        r=self.client();record={'path':'GROUPED_CONFIG.json','id':'f','size':2,'sha256':hashlib.sha256(b'{}').hexdigest()};r.files=[record]
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/record['path']).write_bytes(b'{}')
            with patch.object(r,'upload',side_effect=AssertionError('unexpected upload')),patch.object(r,'call',return_value={'generation':1}):r.sync(d)
        self.assertEqual(r.generation,1)

    def test_no_overwrite_uncommitted_local_output(self):
        r=self.client()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt';p.write_bytes(b'important')
            with self.assertRaises(ValueError):r.restore(d)
            self.assertEqual(p.read_bytes(),b'important')

    def test_corrupt_download_not_published(self):
        r=self.client();record={'path':'f','size':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f'
            with patch.object(r,'call',return_value={'data':'eHl6','end':3,'sha256':record['sha256']}):
                with self.assertRaises(ValueError):r.download(record,p)
            self.assertFalse(p.exists())

    def test_new_notebooks_keep_pinned_science_and_no_mount(self):
        with tempfile.TemporaryDirectory() as d:
            build(d)
            notebooks=list(Path(d).glob('*.ipynb'));self.assertEqual(len(notebooks),8)
            for path in notebooks:
                nb=json.loads(path.read_text(encoding='utf-8'));source='\n'.join(''.join(c['source']) for c in nb['cells'])
                self.assertNotIn('drive.mount(',source)
                self.assertNotIn('authenticate_user()',source)
                self.assertEqual(nb['metadata']['source_commit'],'c729403f7fd4d1207522e3ca793cafb5f8cb8eb0')
                self.assertIn("userdata.get('OPENDETECT_WORKER_KEY')",source)
                for c in nb['cells']:
                    if c['cell_type']=='code':ast.parse(''.join(c['source']))

if __name__=='__main__':unittest.main()
