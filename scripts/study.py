"""Shared study loading, train-only projections and transparent evaluation arithmetic."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cricketshot.features import describe,feature_binding
from cricketshot.policy import CLASSES,head_logits,softmax,calibrated_support


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(include_test=False,known_only=False):
    binding=feature_binding(ROOT)
    rows=[r for r in csv.DictReader((ROOT/'data/manifest.csv').open()) if r['split'] in (['train','tune','calibration','policy','test'] if include_test else ['train','tune','calibration','policy']) and (not known_only or r['target']!='review')]
    modes=read('configs/experiment.json')['representations']
    # This local cache contains frozen descriptors only, never predictions or fits.
    # Bind it to the exact audited row order and descriptor implementation.
    cache=ROOT/'.build/study_cache'/('all' if include_test else 'development')/('known' if known_only else 'mixed')
    signature=hashlib.sha256(json.dumps([binding,modes,[(r['path'],r['sha256'],r['split']) for r in rows]]).encode()).hexdigest()
    marker=cache/'binding.json';arrays=cache/'descriptors.npz'
    if marker.exists() and arrays.exists():
        recorded=json.loads(marker.read_text())
        if recorded['binding']==signature and recorded['sha256']==sha(arrays):
            with np.load(arrays,allow_pickle=False) as z:
                x={m:z[m].copy() for m in modes}
                return rows,x,z['y'].copy(),z['splits'].copy(),z['groups'].copy()
    values={m:[] for m in modes}
    for r in rows:
        path=ROOT/'.build/features'/(r['label']+'_'+r['clip_id']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            assert str(z['binding'])==binding and str(z['input_sha256'])==r['sha256']
            indices={k:np.asarray(v) for k,v in json.loads(str(z['indices'])).items()}
            for m in modes:values[m].append(describe(z['video'],z['focused'],indices,m))
    x={m:np.stack(v) for m,v in values.items()}
    y=np.array([CLASSES.index(r['target']) if r['target']!='review' else -1 for r in rows]);splits=np.array([r['split'] for r in rows]);groups=np.array([int(r['group_id']) for r in rows])
    cache.mkdir(parents=True,exist_ok=True)
    np.savez(arrays,**x,y=y,splits=splits,groups=groups)
    marker.write_text(json.dumps({'binding':signature,'sha256':sha(arrays),'content':'Frozen descriptors only; no model outcomes.'}))
    return rows,x,y,splits,groups


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);cm=np.bincount(y*6+p,minlength=36).reshape(6,6)
    tp=np.diag(cm);n=cm.sum(axis=1);pred=cm.sum(axis=0)
    precision=np.divide(tp,pred,out=np.zeros(6,float),where=pred>0);recall=np.divide(tp,n,out=np.zeros(6,float),where=n>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(6,float),where=(precision+recall)>0)
    return {'n':int(len(y)),'accuracy':float(tp.sum()/len(y)) if len(y) else None,'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'confusion_matrix':cm.tolist(),'per_class':{c:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(n[i])} for i,c in enumerate(CLASSES)}}


def fit_transform(x,mask,components,seed=604):
    scaler=StandardScaler().fit(x[mask]);mean=scaler.mean_.astype(np.float32);scale=scaler.scale_.astype(np.float32)
    z=np.clip((x-mean)/scale,-5,5).astype(np.float32);state={'mean':mean,'scale':scale,'input_clip':np.array(5,np.float32)}
    if components:
        pca=PCA(n_components=components,whiten=True,svd_solver='randomized',random_state=seed)
        with threadpool_limits(limits=4):pca.fit(z[mask])
        state.update(projection_center=pca.mean_.astype(np.float32),projection=(pca.components_/np.sqrt(np.maximum(pca.explained_variance_[:,None],1e-12))).astype(np.float32))
        z=(z-state['projection_center'])@state['projection'].T
    return state,z.astype(np.float32)


def candidate_probabilities(selection,x,out=ROOT/'results'):
    values={}
    for name,r in selection['heads'].items():
        with np.load(out/(name+'.npz'),allow_pickle=False) as z:h={k:z[k].copy() for k in z.files}
        with threadpool_limits(limits=4):values[name]=softmax(head_logits(x[r['representation']],h))
    for name,r in selection['ensembles'].items():values[name]=np.mean([values[h] for h in r['members']],axis=0)
    return values


def selective(y,support,threshold):
    above=support>threshold;accepted=above.sum(axis=1)==1;pred=support.argmax(axis=1);known=y>=0;unknown=~known
    correct=accepted&known&(pred==y)
    nknown=int(known.sum());nunknown=int(unknown.sum());auto_known=int((accepted&known).sum());auto_all=int(accepted.sum())
    return {'n_known':nknown,'n_out_of_scope':nunknown,'automatic_known':auto_known,'automatic_all':auto_all,'correct_automatic':int(correct.sum()),'known_coverage':float(auto_known/nknown) if nknown else None,'known_automatic_accuracy':float(correct.sum()/auto_known) if auto_known else None,'automatic_accuracy_all':float(correct.sum()/auto_all) if auto_all else None,'out_of_scope_review_rate':float((~accepted&unknown).sum()/nunknown) if nunknown else None,'out_of_scope_false_automatic':int((accepted&unknown).sum()),'zero_candidates':int((above.sum(axis=1)==0).sum()),'multiple_candidates':int((above.sum(axis=1)>1).sum()),'per_class':{c:{'n':int((y==i).sum()),'automatic':int((accepted&(y==i)).sum()),'coverage':float((accepted&(y==i)).sum()/(y==i).sum()) if (y==i).any() else None,'automatic_accuracy':float(correct[y==i].sum()/(accepted&(y==i)).sum()) if (accepted&(y==i)).any() else None} for i,c in enumerate(CLASSES)}}


def nll_ece(y,p,bins=15):
    confidence=p.max(axis=1);correct=p.argmax(axis=1)==y;ece=0.
    for a,b in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(confidence>=a)&(confidence<b if b<1 else confidence<=b)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return {'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),'ece_15_bins':float(ece),'brier':float(np.square(p-np.eye(6)[y]).sum(axis=1).mean())}
