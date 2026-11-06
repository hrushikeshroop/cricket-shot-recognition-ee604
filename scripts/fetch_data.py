"""Download pinned author archives for a reproducible non-commercial research study.

No downloaded code is executed. Original footage rights remain upstream; cite
CricShot10k. Inference already includes its small labelled demonstration set.
"""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,shutil,subprocess
import gdown
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fetch(record):
    label=record['label'];base=ROOT/'data';archive=base/'downloads'/(label+'.rar');archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.is_file():
        result=gdown.download(id=record['file_id'],output=str(archive)+'.part',quiet=True,resume=True,use_cookies=False)
        if not result:raise RuntimeError('Archive unavailable: '+label)
        Path(result).replace(archive)
    if sha(archive)!=record['sha256']:raise ValueError('Author archive changed: '+label+'. Use a separately audited experiment.')
    target=base/'raw'/label;target.mkdir(parents=True,exist_ok=True);marker=target/'.extracted.json'
    if marker.is_file():
        assert json.loads(marker.read_text())['archive_sha256']==record['sha256'];return label
    tar=shutil.which('bsdtar') or shutil.which('tar')
    if not tar:raise RuntimeError('Install libarchive/bsdtar for RAR extraction. Windows includes tar.')
    names=subprocess.check_output([tar,'-tf',str(archive)],text=True).splitlines()
    for name in names:
        normalized=name.replace('\\','/');path=PurePosixPath(normalized)
        if path.is_absolute() or '..' in path.parts or ':' in normalized:raise ValueError('Unsafe archive member')
    verbose=subprocess.check_output([tar,'-tvf',str(archive)],text=True)
    if any(line.startswith(('l','h')) for line in verbose.splitlines()):raise ValueError('Archive links are not accepted.')
    subprocess.run([tar,'-xf',str(archive),'-C',str(target)],check=True)
    marker.write_text(json.dumps({'archive_sha256':record['sha256']}));print('Fetched',label,flush=True);return label


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    provenance=json.loads((ROOT/'data/provenance.json').read_text())
    if sha(ROOT/'data/source_videos.csv')!=provenance['source_table_sha256']:raise ValueError('Source table snapshot changed.')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(fetch,provenance['archives']))
    print('All 15 archives match the recorded snapshot.',flush=True)


if __name__=='__main__':main()
