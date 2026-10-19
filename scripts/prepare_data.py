"""Audit real videos, establish source-disjoint splits, cache frozen CNN features."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import csv
import hashlib
import json
import re
from collections import Counter
import cv2
import imagehash
import numpy as np
import onnx
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from cricketshot.core import Encoder, decode_video, color_descriptor

def source_name(stem):
    name = re.sub(r'_\d+_\d+$', '', stem)
    name = re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()
    name = re.sub(r'\b(icc|wt20|match|highlights|cricket|semi|final)\b', '', name)
    name = re.sub(r'\bv\b', 'vs', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Two Ireland/Oman highlight titles differ only in v/vs and the word match.
    # Sorting team names also conservatively merges a reversed title pairing.
    return ' vs '.join(sorted(s.strip() for s in name.split(' vs ')))

def build_encoder():
    source = ROOT / 'models/mobilenetv2-12.onnx'
    out = ROOT / 'models/mobilenetv2_features.onnx'
    if out.exists():
        return
    model = onnx.load(source)
    # Expose the global pooled 1280-dimensional tensor before ImageNet's head.
    pools = [n for n in model.graph.node if n.op_type == 'GlobalAveragePool']
    if len(pools) != 1:
        raise ValueError('Unexpected upstream encoder graph')
    tensor = pools[0].output[0]
    onnx.utils.extract_model(str(source), str(out), [model.graph.input[0].name], [tensor])
    onnx.checker.check_model(onnx.load(out))

def main():
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    build_encoder()
    encoder = Encoder(view='full')
    paths = sorted((ROOT / 'data/raw').rglob('*.avi'))
    if not paths:
        raise RuntimeError('No videos. Run python scripts/fetch_assets.py first.')
    rows, features, colors, hashes, errors = [], [], [], [], []
    cache = out / 'feature_cache'
    cache.mkdir(exist_ok=True)
    for n, path in enumerate(paths):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cp = cache / (sha + '.npz')
        try:
            if cp.exists():
                with np.load(cp, allow_pickle=False) as a:
                    feat, col, hs = a['features'], a['color'], a['phash']
                    meta = json.loads(str(a['metadata']))
            else:
                frames, meta = decode_video(path)
                feat, col = encoder.encode(frames), color_descriptor(frames)
                hs = np.array([str(imagehash.phash(Image.fromarray(frames[i]))) for i in (0, 4, 8)])
                np.savez_compressed(cp, features=feat, color=col, phash=hs, metadata=json.dumps(meta))
            rows.append({'id': f'clip_{n:04d}', 'path': path.relative_to(ROOT).as_posix(),
                         'label': path.parent.name, 'source': source_name(path.stem),
                         'original_name': path.name, 'sha256': sha,
                         'frames': meta['decoded_frames'], 'duration_seconds': meta['duration_seconds']})
            features.append(feat)
            colors.append(col)
            hashes.append([int(h, 16) for h in hs])
        except Exception as exc:
            errors.append({'path': path.relative_to(ROOT).as_posix(), 'error': str(exc)})
        if (n+1) % 25 == 0:
            print(f'Encoded {n+1}/{len(paths)} clips, errors={len(errors)}', flush=True)
    # Union source groups when exact or strongly similar clips cross source names.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    sources, shas = {}, {}
    for i, row in enumerate(rows):
        for key, mapping in ((row['source'], sources), (row['sha256'], shas)):
            if key in mapping:
                union(i, mapping[key])
            else:
                mapping[key] = i
    near = []
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['source'] == rows[j]['source']:
                continue
            distances = [(a ^ b).bit_count() for a, b in zip(hashes[i], hashes[j])]
            if max(distances) <= 6 and np.mean(distances) <= 4:
                union(i, j)
                near.append({'a': rows[j]['id'], 'b': rows[i]['id'], 'distances': distances,
                             'label_conflict': rows[j]['label'] != rows[i]['label']})
    labels = np.array([r['label'] for r in rows])
    groups = np.array([find(i) for i in range(len(rows))])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17).split(np.zeros(len(rows)), labels, groups))
    split = np.full(len(rows), 'train', dtype='<U5')
    split[folds[0][1]] = 'test'
    split[folds[1][1]] = 'val'
    for i, row in enumerate(rows):
        row['group'] = f'source_{groups[i]:04d}'
        row['split'] = str(split[i])
    for label in sorted(set(labels)):
        for s in ('train', 'val', 'test'):
            if not any((labels == label) & (split == s)):
                raise ValueError(f'Class {label} absent from {s}')
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / 'features.npz', features=np.stack(features), color=np.stack(colors),
                        labels=labels, split=split, groups=groups, ids=np.array([r['id'] for r in rows]))
    audit = {'seed': 17, 'split_policy': 'StratifiedGroupKFold(5), test fold 0, validation fold 1, other folds train',
             'upstream_files': len(paths), 'decoded_files': len(rows), 'errors': errors,
             'class_counts': dict(Counter(labels)), 'sources': len(sources), 'union_groups': len(set(groups)),
             'exact_duplicate_files': len(rows) - len(shas), 'cross_source_near_duplicates': near,
             'near_duplicate_rule': '64-bit pHash at first/middle/last sampled frame: max Hamming <=6 and mean <=4',
             'split_counts': {s: dict(Counter(labels[split == s])) for s in ('train', 'val', 'test')},
             'split_groups': {s: len(set(groups[split == s])) for s in ('train', 'val', 'test')},
             'feature_shape': list(np.stack(features).shape), 'frame_sampling': '9 uniform samples, endpoints included'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    main()
