"""Freeze the requested coverage-oriented default before test evaluation."""
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from study import ROOT,read,sha,data,candidate_probabilities,selective
from cricketshot.policy import head_logits,calibrated_support


def main():
    out=ROOT/'results';cfg=read('configs/operating_point.json');fixed=read('results/policy.json');selection=read('results/selection.json')
    if (out/'metrics.json').exists() or fixed.get('operating_point'):raise RuntimeError('The operating point is already frozen or evaluated.')
    (out/'cautious_policy.json').write_text(json.dumps(fixed,indent=2))
    chosen=next(r for r in fixed['grid'] if r['threshold']==cfg['threshold'])
    fixed.update(threshold=cfg['threshold'],policy_metrics=chosen,accuracy_and_rejection_targets_met=chosen['eligible'],known_coverage_goal_met=chosen['known_coverage']>=fixed['targets']['known_coverage_goal'])
    fixed['operating_point']={**cfg,'config_sha256':sha(ROOT/'configs/operating_point.json'),'code_sha256':sha(ROOT/'scripts/select_operating_point.py')}
    g=cfg['development_goals'];fixed['operating_point']['development_goals_met']=bool(chosen['known_coverage']>=g['known_coverage'] and chosen['known_automatic_accuracy']>=g['known_automatic_accuracy'] and chosen['out_of_scope_review_rate']>=g['exposure_review_rate'])
    rows,x,y,splits,groups=data();p=candidate_probabilities(selection,x,out)[selection['deployment']]
    with np.load(out/'scope_head.npz',allow_pickle=False) as z:head={k:z[k].copy() for k in z.files}
    with threadpool_limits(limits=4):logits=head_logits(x[selection.get('scope_representation','context_mean')],head).ravel()
    support,conditional,scope=calibrated_support(p,logits,fixed['calibration']);mask=splits=='policy'
    fixed['confidence_only_same_threshold']=selective(y[mask],conditional[mask],fixed['threshold'])
    grid=[dict(selective(y[mask],conditional[mask],t),threshold=t) for t in read('configs/experiment.json')['policy_grid']]
    matched=min(grid,key=lambda r:(abs(r['known_coverage']-chosen['known_coverage']),-(r['known_automatic_accuracy'] or 0),r['threshold']))
    fixed['confidence_only_matched_coverage']={'threshold':matched['threshold'],'policy_metrics':matched,'selection':'Closest known-shot coverage to the main policy on policy groups only; ties prefer higher automatic known accuracy, then lower threshold.'}
    fixed['finalized_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(out/'policy.json').write_text(json.dumps(fixed,indent=2))
    print(json.dumps({'threshold':fixed['threshold'],'metrics':chosen,'development_goals_met':fixed['operating_point']['development_goals_met']},indent=2))


if __name__=='__main__':main()
