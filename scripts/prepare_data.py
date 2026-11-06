"""Audit all native classes before freezing six-family development/evaluation roles."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,re
import cv2,imagehash,numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Groups:
    def __init__(self,ids):self.p={i:i for i in ids}
    def root(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.root(a),self.root(b);self.p[max(a,b)]=min(a,b)


def inspect(path):
    label=path.relative_to(ROOT/'data/raw').parts[0];match=re.fullmatch(r'vid(\d+)_(\d+)',path.stem,re.I)
    if not match:raise ValueError('Unrecognised source identity: '+str(path))
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(96,54),interpolation=cv2.INTER_AREA))
            if len(frames)>1500:raise ValueError('Unexpectedly long dataset clip')
    finally:cap.release()
    if not frames or fps<=0:raise ValueError('Unreadable clip: '+str(path))
    sampled=np.stack([frames[i] for i in np.rint(np.linspace(0,len(frames)-1,16)).astype(int)])
    thumbs=sampled[[0,8,15]]
    return {'path':path.relative_to(ROOT).as_posix(),'label':label,'clip_id':path.stem,'source_id':int(match[1]),'sha256':sha(path),'decoded_sha256':hashlib.sha256(sampled.tobytes()).hexdigest(),'frames':len(frames),'fps':fps,'duration':len(frames)/fps,'phash':[str(imagehash.phash(Image.fromarray(t))) for t in thumbs]},thumbs


def main():
    cv2.setNumThreads(1);cfg=json.loads((ROOT/'configs/experiment.json').read_text());out=ROOT/'data'
    files=sorted(p for p in (out/'raw').rglob('*') if p.suffix.lower() in {'.avi','.mp4','.mov','.mkv'})
    expected={r['label']:r['video_files'] for r in json.loads((out/'provenance.json').read_text())['archives']}
    actual=dict(Counter(p.relative_to(out/'raw').parts[0] for p in files))
    if actual!=expected:raise ValueError('The raw archive inventory is incomplete or changed. Run fetch_data.py and check the pinned counts before preparing a split.')
    inspected=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,item in enumerate(pool.map(inspect,files),1):
            inspected.append(item)
            if i%500==0:print('Audited',i,'/',len(files),flush=True)
    rows=[r for r,t in inspected];thumbs=np.stack([t for r,t in inspected]);ids={r['source_id'] for r in rows};groups=Groups(ids)
    previous={int(r['source_id']):r for r in csv.DictReader((out/'previous_sources.csv').open())}
    inherited=defaultdict(list)
    for sid,r in previous.items():
        if sid in ids:inherited[r['previous_group']].append(sid)
    for members in inherited.values():
        for sid in members[1:]:groups.join(members[0],sid)
    aliases=defaultdict(list);alias_edges=[]
    for r in csv.DictReader((out/'source_videos.csv').open(encoding='utf-8-sig')):
        sid=int(r['Video No'])
        if sid not in ids:continue
        detail=re.sub(r'\|?\s*day\s*\d+','',r['Details (series, match no, year)'],flags=re.I)
        key=re.sub(r'\W+','',(r['Team vs Team']+detail).casefold());aliases[key].append(sid)
    for members in aliases.values():
        for sid in members[1:]:groups.join(members[0],sid);alias_edges.append([members[0],sid])
    for r in rows:r['target']=cfg['source_mapping'].get(r['label'],'review')
    seen={};duplicate={};conflicts=set();edges=[]
    def link(i,j,kind,mae=None):
        groups.join(rows[i]['source_id'],rows[j]['source_id']);duplicate.setdefault(j,i)
        edges.append({'a':i,'b':j,'type':kind,**({'mae_255':mae} if mae is not None else {})})
        if rows[i]['target']!=rows[j]['target']:conflicts.update((i,j))
    for i,r in enumerate(rows):
        for key in [('file',r['sha256']),('decoded',r['decoded_sha256'])]:
            if key in seen:link(seen[key],i,key[0])
            else:seen[key]=i
    hashes=np.array([[int(h,16) for h in r['phash']] for r in rows],dtype=np.uint64)
    lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8);durations=np.array([r['duration'] for r in rows])
    for i,r in enumerate(rows):
        distances=lut[np.bitwise_xor(hashes[i+1:],hashes[i]).view(np.uint8).reshape(-1,3,8)].sum(axis=2)
        candidates=np.flatnonzero((distances.sum(axis=1)<=8)&(distances.max(axis=1)<=4))+i+1
        for j in candidates:
            if abs(durations[i]-durations[j])/max(durations[i],durations[j])>.15:continue
            mae=float(np.abs(thumbs[i].astype(np.int16)-thumbs[j].astype(np.int16)).mean())
            if mae<=5:link(i,int(j),'near',mae)
    component_roles=defaultdict(set)
    for sid,r in previous.items():
        if sid in ids:component_roles[groups.root(sid)].add(r['previous_split'])
    bridged={g for g,roles in component_roles.items() if len(roles)>1}
    val=[]
    for i,r in enumerate(rows):
        r['group_id']=groups.root(r['source_id']);roles=component_roles[r['group_id']]
        r['excluded_reason']=('compilation_source' if r['source_id'] in cfg['excluded_source_ids'] else 'partition_bridge' if r['group_id'] in bridged else 'conflicting_target_duplicate' if i in conflicts else 'duplicate_clip' if i in duplicate else '')
        if r['excluded_reason']:r['split']='excluded'
        elif roles=={'train'}:r['split']='train'
        elif roles=={'validation'}:r['split']='validation';val.append(r)
        else:r['split']='test'
    assign=[r for r in val if r['label'] not in cfg['unseen_style_classes']]
    y=np.array([cfg['classes'].index(r['target']) if r['target']!='review' else 6 for r in assign]);g=np.array([r['group_id'] for r in assign]);role={}
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=cfg['seed'])
    for fold,(_,idx) in enumerate(cv.split(np.zeros(len(y)),y,g)):
        for i in idx:role[assign[i]['group_id']]=['tune','calibration','policy'][fold]
    for r in val:r['split']=role.get(r['group_id'],'reserved_unknown')
    for r in rows:
        if r['label'] in cfg['unseen_style_classes'] and r['split'] not in {'test','excluded'}:r['split']='reserved_unknown'
    fields=list(rows[0])
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({**r,'phash':json.dumps(r['phash'])} for r in rows)
    splits=['train','tune','calibration','policy','test','reserved_unknown','excluded']
    counts={s:dict(Counter(r['target'] for r in rows if r['split']==s)) for s in splits}
    role_groups={s:{r['group_id'] for r in rows if r['split']==s} for s in splits[:5]}
    for a in role_groups:
        for b in role_groups:
            if a!=b:assert not role_groups[a]&role_groups[b],(a,b)
    for s in splits[:5]:assert set(cfg['classes']).issubset(counts[s]),(s,counts[s])
    audit={'raw_clips':len(rows),'retained_clips':sum(r['split']!='excluded' for r in rows),'raw_class_counts':dict(Counter(r['label'] for r in rows)),'counts':counts,'native_counts':{s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in splits},'group_counts':{s:len(g) for s,g in role_groups.items()},'source_ids':len(ids),'bridged_components':sorted(bridged),'exclusions':dict(Counter(r['excluded_reason'] for r in rows if r['excluded_reason'])),'metadata_alias_edges':alias_edges,'duplicate_edges':edges,'protocol_sha256':sha(ROOT/'configs/experiment.json'),'manifest_sha256':sha(out/'manifest.csv'),'test_limit':'Reused old benchmark groups plus any newly available source components. Source groups are not independently verified matches.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k not in {'duplicate_edges','metadata_alias_edges'}},indent=2),flush=True)


if __name__=='__main__':main()
