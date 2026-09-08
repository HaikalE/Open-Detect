"""Per-scenario capability client; local VM files, A-owned immutable snapshots.
No Drive mount, Drive B writes, pickle loading, or owner OAuth token required.
"""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import uuid
import requests


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    parts=PurePosixPath(relative)
    if (not relative or '\\' in relative or parts.is_absolute() or
            any(p in ('','.','..') or p.startswith('.') for p in relative.split('/'))):
        raise ValueError('Unsafe snapshot path')
    root=Path(root).resolve(); path=(root/relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Snapshot escapes output')
    return path


class Relay:
    def __init__(self,url,key,scenario,session=None,reader=None):
        url=url.strip()
        u=urllib.parse.urlparse(url)
        if u.scheme!='https' or u.hostname!='script.google.com' or not u.path.endswith('/exec') or u.query:
            raise ValueError('Set deployed Apps Script HTTPS /exec URL; no training started')
        if not key or len(key)<32: raise ValueError('Missing scoped worker key in Colab Secrets')
        self.url=url;self.key=key;self.scenario=scenario
        self.session=session or uuid.uuid4().hex;self.generation=0;self.files=[]
        self.reader=reader

    def call(self, action, **kwargs):
        # No URL/token/body in error output. Mutations are NOT blindly retried.
        try:
            response=requests.post(self.url,json=dict(key=self.key,scenario=self.scenario,
                lease=self.session,action=action,**kwargs),timeout=180)
            response.raise_for_status();data=response.json()
        except Exception:
            raise RuntimeError('Relay request failed; keep local files and inspect/retry from latest cloud snapshot') from None
        if not data.get('ok'): raise RuntimeError('Relay refused: '+str(data.get('error','unknown')))
        return data['result']

    def claim(self):
        result=self.call('claim',session=self.session)
        if result['lease']!=self.session: raise ValueError('Lease mismatch')
        m=result['manifest'];self.generation=m['generation'];self.files=m['files']
        return m

    def download(self, record, destination, dataset=None):
        destination=Path(destination)
        if destination.exists():
            if digest(destination)==record['sha256']: return
            raise ValueError('Different local file preserved: '+str(destination))
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.relay-part')
        # Resume only a local transfer with an exact identity sidecar.
        side=temporary.with_name(temporary.name+'.json')
        ident={'size':record['size'],'sha256':record['sha256']}
        if temporary.exists() and (not side.exists() or json.loads(side.read_text())!=ident):
            raise ValueError('Different partial download preserved')
        side.write_text(json.dumps(ident))
        offset=temporary.stat().st_size if temporary.exists() else 0
        started=time.monotonic();initial=offset;last_report=started
        print('DOWNLOAD:',dataset or record.get('path','file'),f'{offset/1048576:.1f}/{record["size"]/1048576:.1f} MiB',
              'direct Drive' if self.reader else 'relay',flush=True)
        with temporary.open('ab') as out:
            while offset<record['size']:
                if self.reader:
                    data=self.reader.read(record,offset)
                    result={'sha256':record['sha256'],'end':offset+len(data)}
                else:
                    args={'offset':offset}
                    args.update(dataset=dataset) if dataset else args.update(path=record['path'])
                    result=self.call('read',**args)
                    data=base64.b64decode(result['data'],validate=True)
                if result['sha256']!=record['sha256'] or result['end']!=offset+len(data) or not data:
                    raise ValueError('Download identity/range mismatch')
                out.write(data);out.flush();offset+=len(data)
                now=time.monotonic()
                if now-last_report>=5 or offset==record['size']:
                    speed=(offset-initial)/1048576/max(now-started,.001)
                    print(f'  {offset/1048576:.1f}/{record["size"]/1048576:.1f} MiB ({100*offset/record["size"]:.1f}%) {speed:.2f} MiB/s',flush=True)
                    last_report=now
        if temporary.stat().st_size!=record['size'] or digest(temporary)!=record['sha256']:
            raise ValueError('Downloaded hash mismatch; partial file preserved')
        os.replace(temporary,destination);side.unlink()
        print('HASH VERIFIED:',dataset or record.get('path','file'),flush=True)

    def restore(self, output):
        output=Path(output)
        expected={r['path'] for r in self.files}
        if output.exists():
            present={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()
                     and '.relay-part' not in p.name}
            if present-expected: raise ValueError('Local output contains uncommitted files; preserved, use a fresh work directory')
        print('RESTORE:',len(self.files),'files',f'{sum(r["size"] for r in self.files)/1048576:.1f} MiB',flush=True)
        for r in self.files: self.download(r,safe_path(output,r['path']))
        print('A SNAPSHOT RESTORED:',self.scenario,'generation',self.generation,flush=True)

    def upload(self,path,relative):
        path=Path(path);size=path.stat().st_size;checksum=digest(path)
        result=self.call('begin',path=relative,size=size,sha256=checksum)
        url=result['url'];u=urllib.parse.urlparse(url)
        if u.scheme!='https' or u.hostname!='www.googleapis.com': raise ValueError('Unexpected upload capability host')
        # Session URL grants only this staged file upload. Never log/persist it.
        offset=0;failures=0
        with path.open('rb') as source:
            while offset<size:
                source.seek(offset);chunk=source.read(8*1024*1024)
                try:
                    r=requests.put(url,data=chunk,headers={'Content-Range':f'bytes {offset}-{offset+len(chunk)-1}/{size}'},timeout=180,allow_redirects=False)
                    if r.status_code in (200,201): offset=size;break
                    if r.status_code==308:
                        acknowledged=int(r.headers['Range'].split('-')[-1])+1 if 'Range' in r.headers else 0
                        if not 0<=acknowledged<=size: raise ValueError('Invalid upload range')
                        if acknowledged>offset:
                            offset=acknowledged
                            continue
                except requests.RequestException: pass
                failures+=1
                if failures>5: raise RuntimeError('Upload failed; no cloud snapshot committed')
                time.sleep(min(failures*2,10))
                try:
                    status=requests.put(url,headers={'Content-Range':f'bytes */{size}'},data=b'',timeout=60,allow_redirects=False)
                except requests.RequestException:
                    continue
                if status.status_code in (200,201): offset=size;break
                if status.status_code!=308: raise RuntimeError('Upload session unavailable; staged data retained in A')
                offset=int(status.headers['Range'].split('-')[-1])+1 if 'Range' in status.headers else 0
        if digest(path)!=checksum: raise ValueError('Local upload source changed')
        return {'path':relative,'id':result['id'],'size':size,'sha256':checksum}

    def sync(self, output):
        output=Path(output);old={r['path']:r for r in self.files};records=[]
        # Called synchronously AFTER completed epoch or evaluator exit. Never background-copy a live PT.
        for source in sorted(output.rglob('*')):
            if not source.is_file() or source.name.endswith('.writing'): continue
            relative=source.relative_to(output).as_posix();safe_path(output,relative)
            checksum=digest(source)
            if relative in old and old[relative]['sha256']==checksum:
                records.append(old[relative]);continue
            # Logs may still be flushed by runner; copy once to immutable local staging.
            temporary=output.parent/('.relay-upload-'+uuid.uuid4().hex)
            try:
                shutil.copyfile(source,temporary)
                if digest(temporary)!=checksum: raise RuntimeError('Output changed during snapshot; stop rather than publish torn files')
                records.append(self.upload(temporary,relative))
            finally:
                if temporary.exists(): temporary.unlink()
        if not records: raise ValueError('Refusing empty snapshot')
        result=self.call('commit',generation=self.generation,files=records)
        self.generation=result['generation'];self.files=records
        print('CLOUD COMMITTED TO A:',self.scenario,'generation',self.generation,flush=True)

    def release(self): return self.call('release')


def from_env():
    return Relay(os.environ['OPENDETECT_RELAY_URL'],os.environ['OPENDETECT_WORKER_KEY'],
                 os.environ['OPENDETECT_SCENARIO'],os.environ['OPENDETECT_RELAY_SESSION'])
