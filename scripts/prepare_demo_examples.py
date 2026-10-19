"""Recreate the selected local examples without changing experiment evidence.

Five examples reuse the illustrations already shipped with the slides. Fetch
the pinned archives with --download only for the optional original disagreement
image and backup clips. Download requires requests. No training is performed.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cricketshot.images import load_image


def restore_images():
    with (ROOT / "data/images/processed/manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    examples = json.loads((ROOT / "demo/image_examples.json").read_text())
    for entry in examples:
        if entry.get("bundled"):
            if not (ROOT / entry["path"]).is_file():
                raise FileNotFoundError("Bundled slide illustration missing: "+entry["path"])
            continue
        row = rows[entry["id"]]
        if row["label"] != entry["label"] or row["split"] != entry["split"]:
            raise ValueError("Example metadata does not match the recorded manifest")
        source = ROOT / row["path"]
        if not source.is_file():
            raise FileNotFoundError("Raw image missing. Run this script with --download first.")
        target = ROOT / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        load_image(source).save(target)
    return len(examples)


def restore_videos():
    from PIL import Image
    from cricketshot.core import decode_video
    with (ROOT / "data/processed/manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    examples = json.loads((ROOT / "demo/examples.json").read_text())
    for entry in examples:
        source = ROOT / rows[entry["id"]]["path"]
        if not source.is_file():
            raise FileNotFoundError("Raw clip missing. Use --download --include-video.")
        target = ROOT / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        thumbnail = ROOT / entry["thumbnail"]
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        if not thumbnail.exists():
            frames, _ = decode_video(target)
            im = Image.fromarray(frames[len(frames)//2])
            im.thumbnail((320, 180))
            im.save(thumbnail, quality=85)
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-video", action="store_true")
    args = parser.parse_args()
    if args.download:
        subprocess.run([sys.executable, str(ROOT / "scripts/fetch_images.py")], cwd=ROOT, check=True)
        if args.include_video:
            subprocess.run([sys.executable, str(ROOT / "scripts/fetch_assets.py")], cwd=ROOT, check=True)
    print("Restored image examples:", restore_images())
    if args.include_video:
        print("Restored video examples:", restore_videos())


if __name__ == "__main__":
    main()
