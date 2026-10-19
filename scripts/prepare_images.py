"""Audit, deduplicate, split and cache two-view features for real cricket images."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import hashlib
import json
from collections import Counter
import imagehash
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from cricketshot.core import Encoder
from cricketshot.images import load_image, image_views


def main():
    out = ROOT / "data/images/processed"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    encoder = Encoder(view="full")
    paths = sorted(
        p
        for p in (ROOT / "data/images/raw").rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )
    rows = []
    features = []
    phashes = []
    thumbs = []
    errors = []
    aliases = {
        "drives": "drive",
        "drive": "drive",
        "legglance-flick": "legglance-flick",
        "pullshot": "pullshot",
        "sweep": "sweep",
    }
    seen = {}
    for n, path in enumerate(paths):
        label = aliases.get(path.parent.name.lower())
        if not label:
            continue
        try:
            image = load_image(path)
            sha = hashlib.sha256(
                image.size[0].to_bytes(4, "big")
                + image.size[1].to_bytes(4, "big")
                + image.tobytes()
            ).hexdigest()
            views = image_views(image)
            cp = cache / (sha + ".npy")
            if cp.exists():
                feat = np.load(cp, allow_pickle=False)
            else:
                feat = encoder.encode([np.asarray(v) for v in views])
                np.save(cp, feat, allow_pickle=False)
            rows.append(
                {
                    "id": f"image_{n:05d}",
                    "path": path.relative_to(ROOT).as_posix(),
                    "label": label,
                    "rgb_sha256": sha,
                    "width": image.width,
                    "height": image.height,
                }
            )
            features.append(feat)
            phashes.append(int(str(imagehash.phash(image)), 16))
            thumbs.append(np.asarray(image.resize((16, 16)), dtype=np.int16))
        except Exception as exc:
            errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
        if (n + 1) % 200 == 0:
            print(f"Image features {n+1}/{len(paths)}, valid {len(rows)}", flush=True)
    if not rows:
        raise ValueError("No labeled images found. Run fetch_images.py first.")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    exact = 0
    near = []
    for i, row in enumerate(rows):
        if row["rgb_sha256"] in seen:
            union(i, seen[row["rgb_sha256"]])
            exact += 1
        else:
            seen[row["rgb_sha256"]] = i
    for i in range(len(rows)):
        for j in range(i):
            if (phashes[i] ^ phashes[j]).bit_count() <= 6:
                mad = float(np.abs(thumbs[i] - thumbs[j]).mean())
                if mad <= 18:
                    union(i, j)
                    if rows[i]["rgb_sha256"] != rows[j]["rgb_sha256"]:
                        near.append([i, j, mad])
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    conflicts = [
        ii for ii in components.values() if len({rows[i]["label"] for i in ii}) > 1
    ]
    excluded = {i for ii in conflicts for i in ii}
    # Collapse decoded exact duplicates to one representative within each label.
    representatives = []
    hash_seen = set()
    for i, row in enumerate(rows):
        if i not in excluded and row["rgb_sha256"] not in hash_seen:
            hash_seen.add(row["rgb_sha256"])
            representatives.append(i)
    kept = [rows[i] for i in representatives]
    feats = np.stack([features[i] for i in representatives])
    labels = np.array([r["label"] for r in kept])
    groups = np.array([find(i) for i in representatives])
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=604).split(
            np.zeros(len(kept)), labels, groups
        )
    )
    split = np.full(len(kept), "train", dtype="<U5")
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    for i, row in enumerate(kept):
        row["group"] = f"visual_{groups[i]:05d}"
        row["split"] = str(split[i])
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    np.savez_compressed(
        out / "features.npz",
        features=feats,
        labels=labels,
        split=split,
        groups=groups,
        ids=np.array([r["id"] for r in kept]),
    )
    audit = {
        "files_found": len(paths),
        "decoded_labeled": len(rows),
        "errors": errors,
        "exact_duplicate_excess": exact,
        "near_duplicate_pairs": len(near),
        "conflicting_components": len(conflicts),
        "conflicting_images_excluded": len(excluded),
        "exact_representatives_kept": len(kept),
        "visual_groups": len(set(groups)),
        "class_counts": dict(Counter(labels)),
        "split_counts": {
            s: dict(Counter(labels[split == s])) for s in ("train", "val", "test")
        },
        "split_groups": {
            s: len(set(groups[split == s])) for s in ("train", "val", "test")
        },
        "split_seed": 604,
        "source_ids_available": False,
        "rule": "Exact decoded RGB hash or pHash <=6 and thumbnail RGB MAD <=18",
        "warning": "Visual groups do not establish match/player independence. Related frames may remain across splits.",
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
