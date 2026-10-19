"""Audit transformed copies before the final image comparison.

Initial hash-only scores are preserved. The final grouping adds dihedral pHash
and geometric ORB verification of frozen-feature nearest-neighbor candidates.
The protocol revision follows visual QA, so final scores are exploratory.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import json
import shutil
import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from cricketshot.images import load_image
from cricketshot.core import normalize


def main():
    out = ROOT / "data/images/processed"
    initial = out / "initial_hash_audit"
    initial.mkdir(exist_ok=True)
    for name in ("manifest.csv", "audit.json", "features.npz"):
        if not (initial / name).exists():
            shutil.copy2(out / name, initial / name)
    evidence = ROOT / "results/images_initial_hash"
    evidence.mkdir(exist_ok=True)
    for name in (
        "metrics.json",
        "test_predictions.npz",
        "test_predictions.csv",
        "latency.json",
    ):
        p = ROOT / "results/images" / name
        if p.exists() and not (evidence / name).exists():
            shutil.copy2(p, evidence / name)
    rows = list(
        csv.DictReader((initial / "manifest.csv").open(encoding="utf-8", newline=""))
    )
    with np.load(initial / "features.npz", allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    audit = json.loads((initial / "audit.json").read_text())
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    old_groups = {}
    for i, row in enumerate(rows):
        if row["group"] in old_groups:
            union(i, old_groups[row["group"]])
        else:
            old_groups[row["group"]] = i
    # ORB descriptors are matched only as an audit tool, never as label features.
    orb = cv2.ORB_create(nfeatures=400, edgeThreshold=5, patchSize=15, fastThreshold=7)
    hashes = []
    thumbs = []
    keypoints = []
    descriptors = []
    areas = []
    for i, row in enumerate(rows):
        im = load_image(ROOT / row["path"])
        im.thumbnail((384, 384))
        variants = []
        for rot in (0, 90, 180, 270):
            variant = im.rotate(rot, expand=True)
            variants.extend([variant, ImageOps.mirror(variant)])
        hashes.append([int(str(imagehash.phash(v)), 16) for v in variants])
        thumbs.append(
            [np.asarray(v.resize((16, 16)), dtype=np.int16) for v in variants]
        )
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        keypoints.append(np.array([k.pt for k in kp], dtype=np.float32))
        descriptors.append(desc)
        areas.append(im.width * im.height)
        if (i + 1) % 1000 == 0:
            print("Transform audit descriptors", i + 1, flush=True)
    rotation_edges = []
    for i in range(len(rows)):
        for j in range(i):
            if find(i) == find(j):
                continue
            # Transform i into the orientation of j. Eight variants cover quarter
            # turns and mirrors; arbitrary rotations rely on local-feature checks.
            for k, h in enumerate(hashes[i]):
                if (h ^ hashes[j][0]).bit_count() <= 6 and np.abs(
                    thumbs[i][k] - thumbs[j][0]
                ).mean() <= 24:
                    union(i, j)
                    rotation_edges.append((i, j))
                    break
    print("New transform hash edges", len(rotation_edges), flush=True)
    x = normalize(data["features"][:, 0, :])
    candidates = set()
    with threadpool_limits(limits=2):
        nn = NearestNeighbors(
            n_neighbors=16, metric="cosine", algorithm="brute", n_jobs=2
        ).fit(x)
        distances, neighbors = nn.kneighbors(x)
    for i, (dd, ii) in enumerate(zip(distances, neighbors)):
        for distance, j in zip(dd[1:], ii[1:]):
            if distance <= 0.20 and find(i) != find(int(j)):
                candidates.add(tuple(sorted((i, int(j)))))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    orb_edges = []
    for n, (i, j) in enumerate(sorted(candidates)):
        if find(i) == find(j):
            continue
        d1, d2 = descriptors[i], descriptors[j]
        if d1 is None or d2 is None or min(len(d1), len(d2)) < 12:
            continue
        pairs = matcher.knnMatch(d1, d2, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
        ]
        if len(good) < 16:
            continue
        p1 = np.float32([keypoints[i][m.queryIdx] for m in good])
        p2 = np.float32([keypoints[j][m.trainIdx] for m in good])
        h, mask = cv2.findHomography(
            p1, p2, cv2.RANSAC, 3.0, maxIters=500, confidence=0.99
        )
        if h is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        count = int(mask.sum())
        if count < 12 or mask.mean() < 0.65:
            continue
        cov1 = cv2.contourArea(cv2.convexHull(p1[mask])) / areas[i]
        cov2 = cv2.contourArea(cv2.convexHull(p2[mask])) / areas[j]
        if min(cov1, cov2) < 0.08:
            continue
        union(i, j)
        orb_edges.append(
            {
                "a": rows[i]["id"],
                "b": rows[j]["id"],
                "inliers": count,
                "ratio": float(mask.mean()),
                "coverage": [float(cov1), float(cov2)],
            }
        )
        if len(orb_edges) % 100 == 0:
            print("Verified geometric edges", len(orb_edges), flush=True)
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    conflicts = [
        ii for ii in components.values() if len({rows[i]["label"] for i in ii}) > 1
    ]
    excluded = {i for ii in conflicts for i in ii}
    kept = [i for i in range(len(rows)) if i not in excluded]
    final_rows = [rows[i] for i in kept]
    labels = data["labels"][kept]
    groups = np.array([find(i) for i in kept])
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=604).split(
            np.zeros(len(kept)), labels, groups
        )
    )
    split = np.full(len(kept), "train", dtype="<U5")
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    from collections import Counter

    for i, row in enumerate(final_rows):
        row["group"] = f"photo_{groups[i]:05d}"
        row["split"] = str(split[i])
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    final_data = {k: v[kept] for k, v in data.items()}
    final_data["groups"] = groups
    final_data["split"] = split
    np.savez_compressed(out / "features.npz", **final_data)
    audit.update(
        {
            "initial_visual_groups": audit["visual_groups"],
            "visual_groups": len(set(groups)),
            "exact_representatives_kept": len(kept),
            "class_counts": dict(Counter(labels)),
            "transform_hash_edges": len(rotation_edges),
            "orb_verified_edges": len(orb_edges),
            "orb_candidates": len(candidates),
            "final_conflicting_components": len(conflicts),
            "final_conflicting_images_excluded": len(excluded),
            "split_counts": {
                s: dict(Counter(labels[split == s])) for s in ("train", "val", "test")
            },
            "split_groups": {
                s: len(set(groups[split == s])) for s in ("train", "val", "test")
            },
            "rule": "Original pHash/RGB components + quarter-turn/mirror pHash <=6 and MAD <=24 + ORB/RANSAC verified frozen-feature neighbors",
            "protocol_status": "Strengthened after initial image score and visual QA. Final comparison is exploratory under revised grouping.",
            "warning": "Transform and geometric grouping reduce further leakage but do not establish match or player independence.",
        }
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    edge_record = {
        "transform_edges": [[rows[i]["id"], rows[j]["id"]] for i, j in rotation_edges],
        "orb_edges": orb_edges,
        "excluded_ids": [rows[i]["id"] for i in sorted(excluded)],
    }
    (out / "transformed_duplicate_edges.json").write_text(
        json.dumps(edge_record, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
