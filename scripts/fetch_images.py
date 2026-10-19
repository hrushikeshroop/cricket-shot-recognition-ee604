"""Fetch the public four-class still-image dataset and preserve its metadata."""
from pathlib import Path
import json
import zipfile
import requests
from fetch_assets import fetch
ROOT=Path(__file__).resolve().parents[1]

def main():
    ref='aneesh10/cricket-shot-dataset'
    meta=requests.get('https://www.kaggle.com/api/v1/datasets/view/'+ref,timeout=60);meta.raise_for_status()
    out=ROOT/'data/images';out.mkdir(parents=True,exist_ok=True)
    (out/'source_metadata.json').write_text(json.dumps(meta.json(),indent=2),encoding='utf-8')
    version=2
    archive,item=fetch('https://www.kaggle.com/api/v1/datasets/download/'+ref+f'?datasetVersionNumber={version}',f'data/downloads/cricket-images-v{version}.zip', '9d497543aca86587c65e167b19f4704e1e639af61da0dbe30a3a7d1485d53e62')
    (out/'provenance.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
    raw=out/'raw';raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (raw/member.filename).resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe archive path')
            if not (raw/member.filename).exists():z.extract(member,raw)
        print('Images archive entries',len(z.infolist()),z.namelist()[:5],flush=True)

if __name__=='__main__':main()
