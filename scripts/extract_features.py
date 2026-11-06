"""Extract frozen representations, without fitting heads or scoring labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cricketshot.features import decode,decode_focused,sample_indices,windows,VideoEncoder,FrameEncoder,feature_binding


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',nargs='+',choices=['train','tune','calibration','policy','test'],default=['train','tune','calibration','policy']);args=parser.parse_args()
    cv2.setNumThreads(1);rows=[r for r in csv.DictReader((ROOT/'data/manifest.csv').open()) if r['split'] in args.split]
    rows.sort(key=lambda r:(['train','tune','calibration','policy','test'].index(r['split']),r['label'],r['clip_id']))
    cache=ROOT/'.build/features';cache.mkdir(parents=True,exist_ok=True);binding=feature_binding(ROOT);marker=cache/'binding.json'
    if marker.exists():assert json.loads(marker.read_text())['sha256']==binding,'Feature cache binding changed.'
    marker.write_text(json.dumps({'sha256':binding}))
    encoder=VideoEncoder(ROOT/'models/mc3_features.onnx');frame_encoder=FrameEncoder(ROOT/'models/mobilenet_features.onnx')
    started=time.perf_counter();new=0
    def load(r):
        path=cache/(r['label']+'_'+r['clip_id']+'.npz')
        if path.is_file():
            with np.load(path,allow_pickle=False) as z:
                assert str(z['binding'])==binding and str(z['input_sha256'])==r['sha256']
                assert z['video'].shape==(24,512) and z['focused'].shape==(24,1280)
            return r,path,None,None
        full,_=decode(ROOT/r['path']);focused=decode_focused(ROOT/r['path']);assert len(full)==len(focused)
        ids=sample_indices(len(full));return r,path,full[ids],focused[ids]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[]
        for i,r in enumerate(rows):
            pending.append(pool.submit(load,r))
            if len(pending)>=3 or i==len(rows)-1:
                while pending and (len(pending)>=3 or i==len(rows)-1):
                    row,path,full,focused=pending.pop(0).result()
                    if full is not None:
                        indices,motion=windows(full);video=encoder.encode(full);frames=frame_encoder.encode(focused)
                        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,video=video,focused=frames,indices=json.dumps({k:v.tolist() for k,v in indices.items()}),motion=json.dumps(motion),input_sha256=row['sha256'],binding=binding);temp.replace(path);new+=1
                    done=i-len(pending)+1
                    if done%100==0 or done==len(rows):print(f'Features {done}/{len(rows)}, newly extracted {new}, elapsed {(time.perf_counter()-started)/60:.1f} min',flush=True)
    print('Finished frozen-feature extraction',flush=True)


if __name__=='__main__':main()
