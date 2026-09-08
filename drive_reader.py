"""Read-only data plane authenticated as the worker, never as owner A.

Only downloads exact IDs from the relay manifest. No mount, writes, permission
changes, public links, or A credentials. Final SHA256 is checked by Relay.
"""
import re
import time
import requests


class DriveReader:
    def __init__(self, session):
        self.session = session

    def read(self, record, offset):
        file_id = record['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
            raise ValueError('Invalid manifest file ID')
        size = record['size']
        if not isinstance(size, int) or not 0 <= offset < size:
            raise ValueError('Invalid download size/offset')
        end = min(offset + 8*1024*1024, size) - 1
        for attempt in range(4):
            try:
                with self.session.get(
                    'https://www.googleapis.com/drive/v3/files/' + file_id,
                    params={'alt': 'media', 'supportsAllDrives': 'true'},
                    headers={'Range': f'bytes={offset}-{end}'},
                    timeout=(15, 90), stream=True, allow_redirects=False,
                ) as response:
                    if response.status_code in (401, 403, 404):
                        raise RuntimeError('Direct Drive read denied. Authenticate the worker and ask A to share dataset/scenario folders with that account as Viewer. Key access alone is not Drive access.')
                    if response.status_code==429 or response.status_code>=500:
                        raise requests.ConnectionError('Transient Drive response')
                    expected = f'bytes {offset}-{end}/{size}'
                    if response.status_code==206:
                        if response.headers.get('Content-Range') != expected:
                            raise ValueError('Drive returned a different range')
                    elif not (response.status_code==200 and offset==0 and end==size-1):
                        raise ValueError('Drive ignored range; refused unbounded download')
                    data=bytearray()
                    for block in response.iter_content(1024*1024):
                        data.extend(block)
                        if len(data)>end-offset+1:
                            raise ValueError('Drive response exceeded requested range')
                    if len(data)!=end-offset+1:
                        raise requests.ConnectionError('Incomplete range')
                    return bytes(data)
            except requests.RequestException:
                if attempt==3:
                    raise RuntimeError('Direct Drive transfer failed after 4 attempts; partial download preserved') from None
                time.sleep(2**attempt)


def colab_reader():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    return DriveReader(AuthorizedSession(credentials))
