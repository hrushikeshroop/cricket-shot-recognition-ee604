from pathlib import Path
from collections import Counter
import re
import json
import requests
import zipfile
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
p = list((ROOT/'data/raw').rglob('*.avi'))
print('Classes', Counter(x.parent.name for x in p))
groups = Counter(re.sub(r'_\d+_\d+$', '', x.stem) for x in p)
print('Source groups', len(groups), json.dumps(groups, indent=2))

class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k == 'href' and v and ('zip' in v.lower() or 'author_kit' in v.lower()):
                print('Author kit URL', v)

Links().feed(requests.get('https://cvpr2022.thecvf.com/author-guidelines', timeout=30).text)
d = requests.get('https://api.github.com/repos/cvpr-org/author-kit/tags', timeout=30).json()
print('CVPR tags', [x['name'] for x in d])
for path in ['C:/Program Files/Microsoft Office/root/Office16/POWERPNT.EXE', 'C:/Program Files/LibreOffice/program/soffice.exe']:
    print(path, Path(path).exists())

url = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip'
out = ROOT/'tools/tectonic'
out.mkdir(parents=True, exist_ok=True)
if not (out/'tectonic.exe').exists():
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    import io
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out)
print('Tectonic ready', flush=True)
