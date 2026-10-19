"""Time end-to-end CPU inference on deterministic held-out clips, without caches."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import csv
import json
import platform
import numpy as np
from cricketshot.core import Recognizer

def main():
    rows=list(csv.DictReader((ROOT/'data/processed/manifest.csv').open(encoding='utf-8',newline='')))
    rows=[r for r in rows if r['split']=='test']
    model=Recognizer()
    model.predict(ROOT/rows[0]['path'])
    sample=[rows[i] for i in np.linspace(0,len(rows)-1,min(25,len(rows)),dtype=int)]
    times=[]
    for row in sample:
        result,_=model.predict(ROOT/row['path']);times.append(result['seconds'])
    result={'clips':len(times),'warmup_clips':1,'scope':'Decode, sample, nine CNN passes and classifier. Model initialization excluded.',
            'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
            'mean_seconds':float(np.mean(times)),'times_seconds':times,'clip_ids':[r['id'] for r in sample],
            'platform':platform.platform(),'processor':platform.processor(),'onnx_intra_op_threads':2}
    (ROOT/'results/latency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
