"""Evaluate frozen models and review policies once, then export numeric deployment files."""
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from study import ROOT,read,sha,data,metrics,candidate_probabilities,selective,nll_ece
from cricketshot.policy import CLASSES,head_logits,calibrated_support
from export_model import export


def bootstrap(y,groups,conditional,support,threshold,baseline_threshold,seed=604,draws=2000):
    rng=np.random.default_rng(seed);ids=np.unique(groups);members={g:np.flatnonzero(groups==g) for g in ids};values=[]
    for _ in range(draws):
        idx=np.concatenate([members[g] for g in rng.choice(ids,len(ids),replace=True)]);known=y[idx]>=0
        main=selective(y[idx],support[idx],threshold);base=selective(y[idx],conditional[idx],baseline_threshold)
        raw=metrics(y[idx][known],conditional[idx][known].argmax(axis=1))
        values.append([raw['accuracy'],raw['macro_f1'],main['known_coverage'],main['known_automatic_accuracy'] if main['known_automatic_accuracy'] is not None else np.nan,main['automatic_accuracy_all'] if main['automatic_accuracy_all'] is not None else np.nan,main['out_of_scope_review_rate'],main['out_of_scope_review_rate']-base['out_of_scope_review_rate']])
    keys=['raw_known_accuracy','raw_known_macro_f1','known_coverage','known_automatic_accuracy','automatic_accuracy_all','out_of_scope_review_rate','review_gain_over_confidence_matched_coverage']
    a=np.asarray(values);return {k:[float(v) for v in np.nanquantile(a[:,i],[.025,.975])] for i,k in enumerate(keys)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='results');parser.add_argument('--deploy',action='store_true');args=parser.parse_args()
    out=ROOT/args.output_dir;selection=json.loads((out/'selection.json').read_text());fixed=json.loads((out/'policy.json').read_text());cfg=read('configs/experiment.json')
    if selection['test_accessed_during_selection'] or (out/'metrics.json').exists():raise RuntimeError('Frozen test evidence already exists; do not overwrite it.')
    assert selection['protocol_sha256']==sha(ROOT/'configs/experiment.json') and selection['manifest_sha256']==sha(ROOT/'data/manifest.csv')
    assert fixed['policy_code_sha256']==sha(ROOT/'src/cricketshot/policy.py')
    rows,x,y,splits,groups=data(include_test=True);values=candidate_probabilities(selection,x,out);p=values[selection['deployment']]
    with np.load(out/'scope_head.npz',allow_pickle=False) as z:scope_head={k:z[k].copy() for k in z.files}
    with threadpool_limits(limits=4):scope_logits=head_logits(x[selection.get('scope_representation','context_mean')],scope_head).ravel()
    support,conditional,scope=calibrated_support(p,scope_logits,fixed['calibration']);threshold=fixed['threshold'];test=splits=='test';known=y>=0
    models={}
    for name in selection['candidates']:
        models[name]={s:metrics(y[(splits==s)&known],values[name][(splits==s)&known].argmax(axis=1)) for s in ['train','tune','calibration','policy','test']}
    triage={s:selective(y[splits==s],support[splits==s],threshold) for s in ['train','tune','calibration','policy','test']}
    native=np.array([r['label'] for r in rows]);unseen=np.isin(native,cfg['unseen_style_classes'])
    unknown_details={label:selective(y[test&(native==label)],support[test&(native==label)],threshold) for label in cfg['exposure_classes']+cfg['unseen_style_classes']}
    baseline_t=fixed['confidence_only_matched_coverage']['threshold']
    intervals=bootstrap(y[test],groups[test],conditional[test],support[test],threshold,baseline_t,cfg['seed'])
    result={'classes':CLASSES,'deployment':selection['deployment'],'threshold':threshold,'models':models,'triage':triage,'unknown_by_native_label':unknown_details,'unseen_style_triage':selective(y[test&unseen],support[test&unseen],threshold),'exposure_style_triage':selective(y[test&~unseen&~known],support[test&~unseen&~known],threshold),
            'confidence_only_same_threshold':selective(y[test],conditional[test],threshold),'confidence_only_matched_coverage':{'threshold':baseline_t,'test':selective(y[test],conditional[test],baseline_t)},
            'calibration_quality':{'raw_test':nll_ece(y[test&known],p[test&known]),'scaled_test':nll_ece(y[test&known],conditional[test&known]),'scope_test_auroc':float(roc_auc_score(known[test].astype(int),scope[test]))},
            'source_group_95ci':intervals,'bootstrap_draws':2000,'bootstrap_seed':cfg['seed'],'test_group_count':int(len(np.unique(groups[test]))),'test_history':cfg['test_history'],'policy_targets':cfg['policy_targets'],'policy_target_status':{k:fixed[k] for k in ['accuracy_and_rejection_targets_met','known_coverage_goal_met']},'evaluation_code_sha256':sha(Path(__file__)),'evaluated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    counts=np.bincount(y[(splits=='train')&known],minlength=6);majority=int(counts.argmax());result['training_majority_baseline']=metrics(y[test&known],np.full(int((test&known).sum()),majority))
    evidence={'y_true':y,'split':splits,'group':groups,'native_label':native,'path':np.array([r['path'] for r in rows]),'scope_logits':scope_logits,'scope_score':scope,'conditional_raw':p,'conditional_calibrated':conditional,'support':support}
    for name in selection['candidates']:evidence[name+'__probabilities']=values[name]
    np.savez_compressed(out/'predictions.npz',**evidence)
    export(out,args.deploy);(out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'deployment':result['deployment'],'known_test':models[result['deployment']]['test'],'triage_test':triage['test'],'unseen_style_review':result['unseen_style_triage'],'intervals':intervals},indent=2),flush=True)


if __name__=='__main__':main()
