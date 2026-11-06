"""Independent NumPy evidence checks and fresh inference, using runtime dependencies.

No sklearn/torch or raw dataset download is required. The default replay record
is private; --record explicitly refreshes the submitted runtime evidence.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cricketshot.policy import CLASSES


def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def arithmetic(y,p):
    matrix=np.array([[int(((y==i)&(p==j)).sum()) for j in range(6)] for i in range(6)])
    stats={};f1=[];recalls=[]
    for i,c in enumerate(CLASSES):
        tp=int(matrix[i,i]);actual=int(matrix[i].sum());predicted=int(matrix[:,i].sum());precision=tp/predicted if predicted else 0.;recall=tp/actual if actual else 0.;f=2*tp/(actual+predicted) if actual+predicted else 0.
        stats[c]={'precision':precision,'recall':recall,'f1':f,'support':actual};f1.append(f);recalls.append(recall)
    return {'n':len(y),'accuracy':float((y==p).mean()) if len(y) else None,'macro_f1':float(np.mean(f1)),'balanced_accuracy':float(np.mean(recalls)),'confusion_matrix':matrix.tolist(),'per_class':stats}


def triage(y,support,threshold):
    count=np.sum(support>threshold,axis=1);auto=count==1;pred=np.argmax(support,axis=1);known=y>=0;unknown=y<0;correct=auto&known&(pred==y)
    nk=int(known.sum());nu=int(unknown.sum());ak=int((auto&known).sum());aa=int(auto.sum());nc=int(correct.sum());classes={}
    for i,c in enumerate(CLASSES):
        n=int((y==i).sum());a=int((auto&(y==i)).sum());right=int((correct&(y==i)).sum());classes[c]={'n':n,'automatic':a,'coverage':a/n if n else None,'automatic_accuracy':right/a if a else None}
    return {'n_known':nk,'n_out_of_scope':nu,'automatic_known':ak,'automatic_all':aa,'correct_automatic':nc,'known_coverage':ak/nk if nk else None,'known_automatic_accuracy':nc/ak if ak else None,'automatic_accuracy_all':nc/aa if aa else None,'out_of_scope_review_rate':float((unknown&~auto).sum()/nu) if nu else None,'out_of_scope_false_automatic':int((unknown&auto).sum()),'zero_candidates':int((count==0).sum()),'multiple_candidates':int((count>1).sum()),'per_class':classes}


def quality(y,p):
    confidence=p.max(axis=1);correct=(p.argmax(axis=1)==y).astype(float);ece=0.
    # Equal-width bins, with the final bin including exactly one.
    indices=np.minimum((confidence*15).astype(int),14)
    for b in range(15):
        mask=indices==b
        if mask.any():ece+=mask.sum()/len(y)*abs(float(correct[mask].mean())-float(confidence[mask].mean()))
    truth=np.zeros_like(p);truth[np.arange(len(y)),y]=1
    return {'nll':float(np.mean(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,None)))),'ece_15_bins':float(ece),'brier':float(np.square(p-truth).sum(axis=1).mean())}


CHECKS=0
def same(actual,recorded,path=''):
    global CHECKS
    if isinstance(actual,dict):
        for k,v in actual.items():same(v,recorded[k],path+'/'+k)
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(recorded),path
        for i,(a,b) in enumerate(zip(actual,recorded)):same(a,b,path+'/'+str(i))
    elif isinstance(actual,(float,np.floating)):
        assert abs(float(actual)-float(recorded))<1e-10,(path,actual,recorded);CHECKS+=1
    else:assert actual==recorded,(path,actual,recorded);CHECKS+=1


def auc(labels,scores):
    order=np.argsort(scores,kind='stable');sorted_scores=scores[order];ranks=np.empty(len(scores),float);start=0
    while start<len(scores):
        stop=start+1
        while stop<len(scores) and sorted_scores[stop]==sorted_scores[start]:stop+=1
        ranks[order[start:stop]]=(start+1+stop)/2;start=stop
    positives=labels.astype(bool);n=int(positives.sum());m=len(labels)-n
    return float((ranks[positives].sum()-n*(n+1)/2)/(n*m))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--record',action='store_true');args=parser.parse_args()
    cfg=read('configs/experiment.json');meta=read('models/classifier.json');m=read('results/metrics.json');selection=read('results/selection.json');policy=read('results/policy.json');calibration=read('results/calibration.json');audit=read('data/audit.json')
    assert meta['classes']==m['classes']==selection['classes']==cfg['classes']==CLASSES
    assert not selection['test_accessed_during_selection'] and selection['scope_fitted']
    assert sha(ROOT/'configs/experiment.json')==meta['protocol_sha256']==selection['protocol_sha256']==audit['protocol_sha256']
    assert sha(ROOT/'data/manifest.csv')==meta['manifest_sha256']==selection['manifest_sha256']==audit['manifest_sha256']
    assert sha(ROOT/'results/selection.json')==calibration['selection_sha256']
    for name,expected in [('heads.npz',meta['heads_sha256']),('scope_head.npz',meta['scope_sha256']),('baseline.npz',meta['baseline_sha256']),('mc3_features.onnx',read('models/encoder.json')['onnx_sha256']),('mobilenet_features.onnx',read('models/focus_encoder.json')['onnx_sha256'])]:assert sha(ROOT/'models'/name)==expected,name
    assert sha(ROOT/'src/cricketshot/features.py')==meta['feature_source_sha256']
    assert sha(ROOT/'src/cricketshot/policy.py')==meta['decision_source_sha256']==policy['policy_code_sha256']
    assert sha(ROOT/'scripts/train.py')==selection['training_code_sha256']
    if 'class_followup' in selection:
        followup=selection['class_followup'];assert not followup['test_accessed']
        assert sha(ROOT/'configs/class_followup.json')==followup['protocol_sha256']
        assert sha(ROOT/'scripts/refine_classes.py')==followup['training_code_sha256']
    assert sha(ROOT/'scripts/calibrate.py')==calibration['calibration_code_sha256']
    assert sha(ROOT/'scripts/evaluate.py')==m['evaluation_code_sha256']
    rows=list(csv.DictReader((ROOT/'data/manifest.csv').open()));eligible=[r for r in rows if r['split'] in ['train','tune','calibration','policy','test']]
    assert len(rows)==audit['raw_clips']==sum(r['video_files'] for r in read('data/provenance.json')['archives'])
    group_sets={s:{r['group_id'] for r in eligible if r['split']==s} for s in ['train','tune','calibration','policy','test']}
    for a in group_sets:
        for b in group_sets:
            if a!=b:assert not group_sets[a]&group_sets[b]
    assert all(r['label'] not in cfg['unseen_style_classes'] for r in eligible if r['split']!='test')
    assert len({(r['label'],r['clip_id']) for r in eligible})==len(eligible)
    with np.load(ROOT/'results/predictions.npz',allow_pickle=False) as z:e={k:z[k].copy() for k in z.files}
    y=e['y_true'];splits=e['split'];groups=e['group'];known=y>=0
    assert len(y)==len(eligible)
    for i,r in enumerate(eligible):
        assert e['path'][i]==r['path'] and splits[i]==r['split'] and groups[i]==int(r['group_id'])
        assert y[i]==(CLASSES.index(r['target']) if r['target']!='review' else -1)
    for name in selection['candidates']:
        p=e[name+'__probabilities'];assert p.shape==(len(y),6) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
        for split in group_sets:
            mask=(splits==split)&known;same(arithmetic(y[mask],p[mask].argmax(axis=1)),m['models'][name][split],name+'/'+split)
        for split in ['train','tune']:same(m['models'][name][split],selection['candidates'][name][split],name+'/selection/'+split)
    chosen=max(selection['candidates'],key=lambda n:selection['candidates'][n]['tune']['macro_f1']);assert chosen==selection['deployment']==m['deployment']==meta['deployment']
    assert [r['name'] for r in meta['members']]==selection['candidates'][chosen]['members']
    same(meta['calibration'],calibration);same(policy['calibration'],calibration)
    trace=read('results/training_trace.json')
    for name,r in selection['heads'].items():
        if r['head_type']=='mlp':
            history=[h for h in trace if h['head']==name];best=max(h['tune_macro_f1'] for h in history);assert r['best_epoch']==min(h['epoch'] for h in history if h['tune_macro_f1']==best)
    assert selection['scope_C']==max(selection['scope_grid'],key=lambda r:r['tune_auroc'])['C']
    if 'scope_followup' in selection:
        followup=selection['scope_followup'];assert not followup['test_accessed']
        assert sha(ROOT/'configs/scope_experiment.json')==followup['protocol_sha256']
        assert sha(ROOT/'scripts/refine_scope.py')==followup['training_code_sha256']
        best_scope=max(followup['candidates'],key=lambda r:r['tune_auroc']);same(best_scope,followup['selected'])
        assert meta['scope_representation']==best_scope['representation']
        scope_trace=read('results/scope_trace.json')
        for candidate in followup['candidates']:
            if candidate['head_type']=='mlp':
                history=[r for r in scope_trace if r['head']==candidate['name']];best_auc=max(r['tune_auroc'] for r in history)
                assert candidate['best_epoch']==min(r['epoch'] for r in history if r['tune_auroc']==best_auc)
                same(best_auc,candidate['tune_auroc'])
    p=e['conditional_raw'];temperature=calibration['temperature'];logits=np.log(np.maximum(p,1e-12))/temperature;logits-=logits.max(axis=1,keepdims=True);scaled=np.exp(logits);scaled/=scaled.sum(axis=1,keepdims=True)
    # The saved logit affine transform is float32; the stable sigmoid then
    # evaluates in float64. Preserve that boundary in this independent formula.
    affine=(calibration['scope_slope']*e['scope_logits']+calibration['scope_intercept']).astype(np.float64)
    scope=np.exp(-np.logaddexp(0,-affine));support=scope[:,None]*scaled
    np.testing.assert_allclose(scaled,e['conditional_calibrated'],atol=1e-12);np.testing.assert_allclose(scope,e['scope_score'],atol=1e-12);np.testing.assert_allclose(support,e['support'],atol=1e-12)
    cmask=splits=='calibration';ck=cmask&known;same(int(cmask.sum()),calibration['n']);same(int(ck.sum()),calibration['known'])
    original_logp=np.log(np.maximum(p[ck],1e-12));temperature_gradient=float(np.mean((original_logp[np.arange(ck.sum()),y[ck]]-(scaled[ck]*original_logp).sum(axis=1))/temperature))
    assert .05<=temperature<=20
    if .051<temperature<19.99:assert abs(temperature_gradient)<1e-4
    else:assert temperature_gradient>=-1e-4 if temperature<=.051 else temperature_gradient<=1e-4
    residual=scope[cmask]-known[cmask].astype(int)
    assert abs(float(residual.mean()))<1e-5
    assert abs(float(np.mean(residual*e['scope_logits'][cmask]))+calibration['scope_slope']/(100*cmask.sum()))<1e-5
    threshold=policy['threshold'];assert threshold==m['threshold']==meta['threshold']
    for split in group_sets:same(triage(y[splits==split],support[splits==split],threshold),m['triage'][split],split+'/triage')
    mask=splits=='policy';grid=[];target=cfg['policy_targets']
    for recorded in policy['grid']:
        r=triage(y[mask],support[mask],recorded['threshold']);same(r,recorded,'policy/grid');r['threshold']=recorded['threshold'];r['eligible']=bool(r['automatic_known']>=50 and r['known_automatic_accuracy']>=target['known_automatic_accuracy'] and r['out_of_scope_review_rate']>=target['exposure_review_rate']);r['fallback_objective']=r['known_coverage']*(r['known_automatic_accuracy'] or 0)-(1-r['out_of_scope_review_rate']);same(r['eligible'],recorded['eligible']);same(r['fallback_objective'],recorded['fallback_objective']);grid.append(r)
    candidates=[r for r in grid if r['eligible']]
    best=max(candidates,key=lambda r:(r['known_coverage'],r['known_automatic_accuracy'],-r['threshold'])) if candidates else max(grid,key=lambda r:(r['fallback_objective'],r['known_automatic_accuracy'] or 0,-r['threshold']))
    if 'operating_point' in policy:
        operational=read('configs/operating_point.json');recorded=policy['operating_point']
        assert sha(ROOT/'configs/operating_point.json')==recorded['config_sha256']
        assert sha(ROOT/'scripts/select_operating_point.py')==recorded['code_sha256']
        assert best['threshold']==operational['cautious_reference_threshold']
        best=next(r for r in grid if r['threshold']==operational['threshold'])
        goals=operational['development_goals'];same(recorded['development_goals_met'],best['known_coverage']>=goals['known_coverage'] and best['known_automatic_accuracy']>=goals['known_automatic_accuracy'] and best['out_of_scope_review_rate']>=goals['exposure_review_rate'])
    assert best['threshold']==threshold
    same(best,policy['policy_metrics']);same(best['eligible'],policy['accuracy_and_rejection_targets_met']);same(best['known_coverage']>=target['known_coverage_goal'],policy['known_coverage_goal_met'])
    base_grid=[dict(triage(y[mask],scaled[mask],t),threshold=t) for t in cfg['policy_grid']];matched=min(base_grid,key=lambda r:(abs(r['known_coverage']-best['known_coverage']),-(r['known_automatic_accuracy'] or 0),r['threshold']));assert matched['threshold']==policy['confidence_only_matched_coverage']['threshold'];same(matched,policy['confidence_only_matched_coverage']['policy_metrics'])
    test=splits=='test';same(triage(y[test],scaled[test],threshold),m['confidence_only_same_threshold']);same(triage(y[test],scaled[test],matched['threshold']),m['confidence_only_matched_coverage']['test'])
    unseen=np.isin(e['native_label'],cfg['unseen_style_classes']);same(triage(y[test&unseen],support[test&unseen],threshold),m['unseen_style_triage']);same(triage(y[test&~unseen&~known],support[test&~unseen&~known],threshold),m['exposure_style_triage'])
    majority=int(np.bincount(y[(splits=='train')&known],minlength=6).argmax());same(arithmetic(y[test&known],np.full(int((test&known).sum()),majority)),m['training_majority_baseline'])
    for label in cfg['exposure_classes']+cfg['unseen_style_classes']:
        label_mask=test&(e['native_label']==label);same(triage(y[label_mask],support[label_mask],threshold),m['unknown_by_native_label'][label])
    for split in ['policy','test']:
        km=(splits==split)&known;actual={'raw':quality(y[km],p[km]),'scaled':quality(y[km],scaled[km])}
        if split=='policy':
            # Development and final evidence use different GEMM batch sizes.
            # Float32 roundoff can change losses by a few billionths. All
            # threshold counts/decisions above must still match exactly.
            for kind,key in [('raw','raw'),('scaled','temperature_scaled')]:
                for quantity,value in actual[kind].items():
                    np.testing.assert_allclose(value,policy['policy_calibration_quality'][key][quantity],rtol=0,atol=2e-7)
        else:same(actual['raw'],m['calibration_quality']['raw_test']);same(actual['scaled'],m['calibration_quality']['scaled_test'])
    same(auc(known[test],scope[test]),m['calibration_quality']['scope_test_auroc'])
    for split in ['train','tune','calibration','policy']:same(auc(known[splits==split],scope[splits==split]),policy['scope_auroc_by_role'][split])
    # Independent whole-group bootstrap, keeping the paired policies aligned.
    yy=y[test];gg=groups[test];ss=support[test];cc=scaled[test];ids=np.unique(gg);members=[np.flatnonzero(gg==g) for g in ids];rng=np.random.default_rng(m['bootstrap_seed']);draws=[]
    for _ in range(m['bootstrap_draws']):
        idx=np.concatenate([members[j] for j in rng.integers(0,len(ids),len(ids))]);km=yy[idx]>=0;r=triage(yy[idx],ss[idx],threshold);b=triage(yy[idx],cc[idx],matched['threshold']);raw=arithmetic(yy[idx][km],cc[idx][km].argmax(axis=1));draws.append([raw['accuracy'],raw['macro_f1'],r['known_coverage'],r['known_automatic_accuracy'] if r['known_automatic_accuracy'] is not None else np.nan,r['automatic_accuracy_all'] if r['automatic_accuracy_all'] is not None else np.nan,r['out_of_scope_review_rate'],r['out_of_scope_review_rate']-b['out_of_scope_review_rate']])
    keys=['raw_known_accuracy','raw_known_macro_f1','known_coverage','known_automatic_accuracy','automatic_accuracy_all','out_of_scope_review_rate','review_gain_over_confidence_matched_coverage'];a=np.asarray(draws)
    for i,k in enumerate(keys):same(np.nanquantile(a[:,i],[.025,.975]).tolist(),m['source_group_95ci'][k],k+'/ci')
    print(f'Passed {CHECKS} independent value checks, model selection, threshold grid, source separation and bootstrap intervals.',flush=True)
    if not args.replay:return
    from cricketshot.recognizer import Recognizer
    cv2.setNumThreads(1);recognizer=Recognizer();examples=read('demo/examples.json');lookup={r['path']:i for i,r in enumerate(eligible)};replays=[];times=[]
    for example in examples:
        path=ROOT/example['path'];assert sha(path)==example['sha256'];result,frames=recognizer.predict(path);i=lookup[example['source_path']]
        assert example['sha256']==eligible[i]['sha256'] and example['source_label']==eligible[i]['label']
        values=np.array([result['support_scores'][c] for c in CLASSES]);error=float(np.max(np.abs(values-support[i])));assert error<2e-4,(example['id'],error)
        expected=triage(y[i:i+1],support[i:i+1],threshold);assert (result['decision']=='automatic')==bool(expected['automatic_all'])
        if result['decision']=='automatic':assert result['class']==CLASSES[int(support[i].argmax())]
        else:assert result['class'] is None
        assert result['input_sha256']==sha(path);times.append(result['elapsed_seconds']);replays.append({'id':example['id'],'decision':result['decision'],'class':result['class'],'input_sha256':result['input_sha256'],'score_max_abs_error':error,'seconds':result['elapsed_seconds']});print('Fresh replay',example['id'],result['decision'],result['class'],flush=True)
    private=ROOT/'.build/runtime';private.mkdir(parents=True,exist_ok=True);pattern=private/'wiring_pattern.avi';writer=cv2.VideoWriter(str(pattern),cv2.VideoWriter_fourcc(*'FFV1'),24,(320,192));assert writer.isOpened()
    x=np.tile(np.arange(320,dtype=np.uint16),(192,1))
    for t in range(24):writer.write(np.stack([((x+t*7)%256).astype(np.uint8),((x//2+t*11)%256).astype(np.uint8),((x//3+t*5)%256).astype(np.uint8)],axis=-1))
    writer.release();pattern_sha=sha(pattern);wiring,_=recognizer.predict(pattern);assert wiring['input_sha256']==pattern_sha
    record={'n_clips':len(times),'median_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),'timing_scope':'Warm encoders: interval decode, motion, CNN features, scope and class heads. Excludes model initialisation, JPEG previews and browser rendering.','examples':replays,'wiring_input':{'description':'Generated moving gradient; wiring/provenance check only, no recognition ground truth','input_sha256':pattern_sha,'recorded_sha256':wiring['input_sha256']},'heads_sha256':meta['heads_sha256'],'independent_value_checks':CHECKS}
    destination=ROOT/'results/runtime_verification.json' if args.record else private/'latest_verification.json';destination.write_text(json.dumps(record,indent=2));print(json.dumps({'replayed':len(times),'median_seconds':record['median_seconds'],'wiring_hash_matches':True},indent=2))


if __name__=='__main__':main()
