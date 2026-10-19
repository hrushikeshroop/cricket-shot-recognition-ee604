"""Second, explicitly exploratory, experiment after observing the pilot failure.

The split is unchanged. ROI geometry was fixed before computing ROI validation
and test results. This is development on a reused holdout, not pristine testing.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import csv
import json
import shutil
import numpy as np
from cricketshot.core import Encoder, decode_video

def main():
    d=ROOT/'data/processed'
    # Preserve the full-frame experiment and never overwrite its evidence.
    original=d/'features_fullframe.npz'
    if not original.exists():shutil.copy2(d/'features.npz',original)
    cache=d/'roi_feature_cache';cache.mkdir(exist_ok=True)
    rows=list(csv.DictReader((d/'manifest.csv').open(encoding='utf-8',newline='')))
    encoder=Encoder(view='batter_roi');all_features=[]
    for i,row in enumerate(rows):
        cp=cache/(row['sha256']+'.npy')
        if cp.exists():feat=np.load(cp,allow_pickle=False)
        else:
            frames,_=decode_video(ROOT/row['path']);feat=encoder.encode(frames)
            np.save(cp,feat,allow_pickle=False)
        all_features.append(feat)
        if (i+1)%50==0:print(f'ROI features {i+1}/{len(rows)}',flush=True)
    with np.load(original,allow_pickle=False) as a:
        data={k:a[k] for k in a.files}
    data['features']=np.stack(all_features)
    np.savez_compressed(d/'features.npz',**data)
    record={'view':'batter_roi','normalized_bounds':[.30,.02,.70,.65],
            'motivation':'Full-frame pilot showed poor recognition. Small batters and broadcast context motivated a fixed upper-center crop.',
            'evaluation_status':'Exploratory reused holdout. Original test results were observed before this revision.',
            'split':'Unchanged source-grouped split, seed 17','detector':False}
    (ROOT/'results/roi_protocol.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('ROI features ready.',flush=True)

if __name__=='__main__':main()
