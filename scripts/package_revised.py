"""Archive the full comparison or a compact demo; verify every payload hash."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={".venv",".build",".git",".pytest_cache","__pycache__",".codex",".agents",".aws"}
STORED={".zip",".jpg",".jpeg",".png",".webp",".avi",".mp4",".pdf",".pptx",".docx"}


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def select_files(kind):
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(p in EXCLUDED for p in relative.parts):continue
        name=relative.as_posix()
        if path.suffix in {".pyc",".part",".log",".aux",".blg",".bbl"}:continue
        if name.startswith("output/"):
            if path.name!="CricketShot_Overleaf.zip" and path.suffix not in {".pdf",".pptx",".PNG"}:continue
        if kind=="demo":
            if name.startswith(("data/raw/","data/downloads/","data/images/raw/","tools/","course_notes/")):continue
            if name.startswith("data/") and path.suffix in {".npy",".npz"}:continue
            if "/cache/" in name or "/feature_cache/" in name or "/roi_feature_cache/" in name:continue
            if name=="models/mobilenetv2-12.onnx":continue
        yield path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--kind",choices=["full","demo"],default="full")
    args=parser.parse_args();output=ROOT/"output";output.mkdir(exist_ok=True)
    archive=output/("CricketShot_Revised_"+("Full" if args.kind=="full" else "Demo")+".zip")
    partial=archive.with_suffix(".zip.part")
    files=list(select_files(args.kind));entries=[];started=time.monotonic();last=started
    print(f"Packaging {len(files)} {args.kind} files",flush=True)
    prefix="CricketShot_EE604_Revised/"
    guide=("# Start here\n\nRead README.md and docs/demo_guide.md. Open the report and slides under output/. "
           "See docs/course_mapping.md and docs/viva_guide.md. On a new Windows machine with Python 3.11+, "
           "run setup.bat then run_demo.bat and open http://127.0.0.1:6040. No machine environment is included.\n\n"
           "The full package includes raw data, original dataset ZIPs, computed research features, models, evidence and lecture PDFs. "
           "The smaller demo omits training assets, lecture PDFs and build tools; full-data tests require the full package. "
           "Machine environments, temporary build/Python caches, account metadata and duplicate handoff archives are excluded.\n\n"
           "The original CNN scores remain 65.5% accuracy and 64.9% macro F1. HOG scores 35.3% and 34.5%. "
           "Novelty and generalization remain limited. Team identities and the public repository URL are in README.md and the report. "
           "The public code repository is linked in README.md. ARCHIVE_MANIFEST.json lists all payload sizes and SHA-256 values.\n")
    with zipfile.ZipFile(partial,"w",zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for i,path in enumerate(files,1):
            name=path.relative_to(ROOT).as_posix()
            entries.append({"path":name,"bytes":path.stat().st_size,"sha256":digest(path)})
            z.write(path,prefix+name,compress_type=zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
            if time.monotonic()-last>=10 or i==len(files):
                print(f"Packed {i}/{len(files)} ({time.monotonic()-started:.0f}s)",flush=True);last=time.monotonic()
        content=guide.encode("utf-8");z.writestr(prefix+"COMPARE_FIRST.md",content)
        entries.append({"path":"COMPARE_FIRST.md","bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
        manifest={"kind":args.kind,"workspace_file_count":len(files),"files":entries,"excluded_directories":sorted(EXCLUDED)}
        manifest_bytes=(json.dumps(manifest,indent=2)+"\n").encode("utf-8")
        z.writestr(prefix+"ARCHIVE_MANIFEST.json",manifest_bytes)
    with zipfile.ZipFile(partial) as z:
        names=z.namelist();expected={prefix+entry["path"] for entry in entries}|{prefix+"ARCHIVE_MANIFEST.json"}
        assert set(names)==expected and len(names)==len(expected)
        assert z.read(prefix+"ARCHIVE_MANIFEST.json")==manifest_bytes
        for entry in entries:
            h=hashlib.sha256();size=0
            with z.open(prefix+entry["path"]) as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):h.update(block);size+=len(block)
            assert size==entry["bytes"] and h.hexdigest()==entry["sha256"],entry["path"]
    assert [p.relative_to(ROOT).as_posix() for p in select_files(args.kind)]==[p.relative_to(ROOT).as_posix() for p in files]
    partial.replace(archive);archive_hash=digest(archive)
    result={"archive":archive.name,"bytes":archive.stat().st_size,"sha256":archive_hash,
            "workspace_files":len(files),"entries":len(entries)+1,"all_payload_hashes_sizes_and_crcs_pass":True,
            "workspace_membership_rechecked":True,"seconds":round(time.monotonic()-started,1)}
    (output/(archive.stem+"_verification.json")).write_text(json.dumps(result,indent=2))
    archive.with_suffix(".zip.sha256").write_text(archive_hash+"  "+archive.name+"\n",encoding="ascii")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
