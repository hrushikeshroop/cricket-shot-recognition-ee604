"""Download public upstream assets, keeping URLs and checksums for provenance."""
from pathlib import Path
import hashlib
import json
import requests
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def fetch(url, target, expected_sha256=None):
    target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with requests.get(url, stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with target.with_suffix(target.suffix + '.part').open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        target.with_suffix(target.suffix + '.part').replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f'Checksum mismatch for {target.name}. Refusing changed or incomplete asset.')
    print(target.relative_to(ROOT), target.stat().st_size, digest, flush=True)
    return target, {'url': url, 'path': str(target.relative_to(ROOT)), 'sha256': digest,
                    'bytes': target.stat().st_size, 'accessed': '2026-10-17'}

def main():
    provenance = []
    ref = 'amittalmale/cricket-shots'
    meta = requests.get('https://www.kaggle.com/api/v1/datasets/view/' + ref, timeout=60)
    meta.raise_for_status()
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'data/source_metadata.json').write_text(json.dumps(meta.json(), indent=2), encoding='utf-8')
    archive, item = fetch('https://www.kaggle.com/api/v1/datasets/download/' + ref + '?datasetVersionNumber=1', 'data/downloads/cricket-shots-v1.zip', '1b4ceb40a8c976c2a06b532aa0280b82fdf41cfc1d83cd228e9159cfa5b5053d')
    provenance.append(item)
    out = ROOT / 'data/raw'
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            dest = (out / member.filename).resolve()
            if not dest.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path')
            if not dest.exists():
                z.extract(member, out)
        print('Archive entries', len(z.infolist()), flush=True)
        print('\n'.join(z.namelist()[:10]), flush=True)
    _, item = fetch('https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx', 'models/mobilenetv2-12.onnx', 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5')
    provenance.append(item)
    (ROOT / 'data/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
