"""Calibrate on dedicated groups, then select review threshold on policy groups."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from study import ROOT,read,sha,data,candidate_probabilities,selective,nll_ece
from cricketshot.policy import softmax,head_logits,calibrated_support,sigmoid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='results');args=parser.parse_args()
    out=ROOT/args.output_dir;selection=json.loads((out/'selection.json').read_text());cfg=read('configs/experiment.json')
    assert selection['scope_fitted'],'Fit the scope head before calibration.'
    if selection['test_accessed_during_selection'] or (out/'metrics.json').exists() or (out/'policy.json').exists():raise RuntimeError('Do not overwrite a frozen/evaluated review policy.')
    rows,x,y,splits,groups=data();probabilities=candidate_probabilities(selection,x,out);p=probabilities[selection['deployment']]
    calibration=(splits=='calibration');known_calibration=calibration&(y>=0);policy=splits=='policy'
    assert not (set(groups[calibration])&set(groups[policy]))
    logp=np.log(np.maximum(p,1e-12))
    def objective(log_temperature):
        q=softmax(logp[known_calibration]/np.exp(log_temperature))
        return float(-np.log(np.maximum(q[np.arange(len(q)),y[known_calibration]],1e-12)).mean())
    fitted=minimize_scalar(objective,bounds=(np.log(.05),np.log(20)),method='bounded')
    if not fitted.success:raise RuntimeError('Temperature optimisation failed.')
    with np.load(out/'scope_head.npz',allow_pickle=False) as z:h={k:z[k].copy() for k in z.files}
    with threadpool_limits(limits=4):scope_logits=head_logits(x[selection.get('scope_representation','context_mean')],h).ravel()
    platt=LogisticRegression(C=100,max_iter=3000,tol=1e-8)
    with threadpool_limits(limits=4):platt.fit(scope_logits[calibration,None],(y[calibration]>=0).astype(int))
    record={'temperature':float(np.exp(fitted.x)),'scope_slope':float(platt.coef_[0,0]),'scope_intercept':float(platt.intercept_[0]),'role':'calibration','n':int(calibration.sum()),'known':int(known_calibration.sum()),'selection_sha256':sha(out/'selection.json'),'calibration_code_sha256':sha(Path(__file__))}
    support,conditional,scope=calibrated_support(p,scope_logits,record)
    grid=[];targets=cfg['policy_targets']
    for threshold in cfg['policy_grid']:
        measured=selective(y[policy],support[policy],threshold)
        measured['threshold']=threshold
        measured['eligible']=bool(measured['automatic_known']>=50 and measured['known_automatic_accuracy']>=targets['known_automatic_accuracy'] and measured['out_of_scope_review_rate']>=targets['exposure_review_rate'])
        measured['fallback_objective']=float(measured['known_coverage']*(measured['known_automatic_accuracy'] or 0)-(1-measured['out_of_scope_review_rate']))
        grid.append(measured)
    eligible=[r for r in grid if r['eligible']]
    chosen=max(eligible,key=lambda r:(r['known_coverage'],r['known_automatic_accuracy'],-r['threshold'])) if eligible else max(grid,key=lambda r:(r['fallback_objective'],r['known_automatic_accuracy'] or 0,-r['threshold']))
    fixed={'threshold':chosen['threshold'],'role':'policy','policy_metrics':chosen,'accuracy_and_rejection_targets_met':bool(eligible),'known_coverage_goal_met':chosen['known_coverage']>=targets['known_coverage_goal'],'targets':targets,'deployment':selection['deployment'],'grid':grid,'support_definition':cfg['support'],'decision_rule':cfg['review_rule'],'policy_code_sha256':sha(ROOT/'src/cricketshot/policy.py'),'calibration':record,'finalized_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    fixed['policy_calibration_quality']={'raw':nll_ece(y[policy&(y>=0)],p[policy&(y>=0)]),'temperature_scaled':nll_ece(y[policy&(y>=0)],conditional[policy&(y>=0)]),'scope_auroc':float(roc_auc_score((y[policy]>=0).astype(int),scope[policy]))}
    fixed['scope_auroc_by_role']={s:float(roc_auc_score((y[splits==s]>=0).astype(int),scope[splits==s])) for s in ['train','tune','calibration','policy']}
    # Hold the chosen threshold fixed while measuring the confidence-only ablation.
    fixed['confidence_only_same_threshold']=selective(y[policy],conditional[policy],chosen['threshold'])
    baseline_grid=[dict(selective(y[policy],conditional[policy],t),threshold=t) for t in cfg['policy_grid']]
    matched=min(baseline_grid,key=lambda r:(abs(r['known_coverage']-chosen['known_coverage']),-(r['known_automatic_accuracy'] or 0),r['threshold']))
    fixed['confidence_only_matched_coverage']={'threshold':matched['threshold'],'policy_metrics':matched,'selection':'Closest known-shot coverage to the main policy on policy groups only; ties prefer higher automatic known accuracy, then lower threshold.'}
    (out/'calibration.json').write_text(json.dumps(record,indent=2));(out/'policy.json').write_text(json.dumps(fixed,indent=2))
    print(json.dumps({k:fixed[k] for k in ['threshold','policy_metrics','accuracy_and_rejection_targets_met','known_coverage_goal_met','policy_calibration_quality']},indent=2),flush=True)


if __name__=='__main__':main()
