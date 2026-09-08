import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import requests
from drive_reader import DriveReader
from relay_client import Relay


def response(code=206, body=b'abc', content_range='bytes 0-2/3'):
    r=Mock(status_code=code,headers={'Content-Range':content_range})
    r.__enter__=Mock(return_value=r);r.__exit__=Mock(return_value=False)
    r.iter_content.return_value=iter([body])
    return r


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.record={'id':'safe_id','path':'last.pt','size':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
        self.session=Mock()
        self.reader=DriveReader(self.session)

    def test_exact_range(self):
        self.session.get.return_value=response()
        self.assertEqual(self.reader.read(self.record,0),b'abc')
        self.assertEqual(self.session.get.call_args.kwargs['headers']['Range'],'bytes=0-2')
        self.session.post.assert_not_called()

    def test_no_public_fallback_on_denied(self):
        self.session.get.return_value=response(403)
        with self.assertRaisesRegex(RuntimeError,'share'):self.reader.read(self.record,0)
        self.assertEqual(self.session.get.call_count,1)

    def test_wrong_range_rejected(self):
        self.session.get.return_value=response(content_range='bytes 1-3/4')
        with self.assertRaises(ValueError):self.reader.read(self.record,0)

    def test_oversized_response_rejected(self):
        self.session.get.return_value=response(body=b'abcd')
        with self.assertRaises(ValueError):self.reader.read(self.record,0)

    def test_retries_bounded(self):
        self.session.get.side_effect=requests.Timeout('private details')
        with patch('drive_reader.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'4 attempts'):self.reader.read(self.record,0)
        self.assertEqual(self.session.get.call_count,4)

    def test_existing_partial_resume_hash_verified_no_relay_read(self):
        relay=Relay('https://script.google.com/macros/s/demo/exec','x'*64,'C-1',reader=self.reader)
        self.session.get.return_value=response(body=b'bc',content_range='bytes 1-2/3')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt';p.with_name('last.pt.relay-part').write_bytes(b'a')
            p.with_name('last.pt.relay-part.json').write_text(json.dumps({'size':3,'sha256':self.record['sha256']}))
            with patch.object(relay,'call',side_effect=AssertionError('No relay data calls')):
                relay.download(self.record,p)
            self.assertEqual(p.read_bytes(),b'abc')
        self.assertEqual(self.session.get.call_args.kwargs['headers']['Range'],'bytes=1-2')

    def test_direct_corrupt_data_not_published(self):
        relay=Relay('https://script.google.com/macros/s/demo/exec','x'*64,'C-1',reader=self.reader)
        self.session.get.return_value=response(body=b'xyz')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'
            with self.assertRaisesRegex(ValueError,'hash'):relay.download(self.record,p)
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main()
