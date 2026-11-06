"""Development-only dimension follow-up; preserve all earlier fitted candidates."""
import json,time
from train import neural
from study import ROOT,read,sha,data,fit_transform,candidate_probabilities,metrics


def main():
    out=ROOT/'results';cfg=read('configs/class_followup.json');base=read('configs/experiment.json');selection=read('results/selection.json')
    if (out/'metrics.json').exists() or (out/'policy.json').exists() or selection.get('class_followup'):raise RuntimeError('A new study is required after a frozen policy or test.')
    rows,x,y,splits,groups=data();train=(splits=='train')&(y>=0);tune=(splits=='tune')&(y>=0);trace=read('results/training_trace.json');ensembles={}
    original=selection['deployment']
    for rep in cfg['representations']:
        state,z=fit_transform(x[rep],train,cfg['pca'],base['seed']);members=[]
        for seed in cfg['seeds']:
            name=f'{rep}_pca{cfg["pca"]}_h{cfg["hidden"]}_seed{seed}'
            record,history=neural(z,y,splits,state,base['neural'],cfg['hidden'],seed,name,out);trace.extend(history);selection['heads'][name]={**record,'representation':rep,'pca':cfg['pca']};members.append(name)
        name=f'{rep}_pca{cfg["pca"]}_ensemble';selection['ensembles'][name]={'members':members};ensembles[rep]=members
    name=f'full_motion_pca{cfg["pca"]}_ensemble';selection['ensembles'][name]={'members':sum(ensembles.values(),[])}
    p=candidate_probabilities(selection,x,out)
    for name in [f'{r}_pca{cfg["pca"]}_ensemble' for r in cfg['representations']]+[name]:
        selection['candidates'][name]={'members':selection['ensembles'][name]['members'],'train':metrics(y[train],p[name][train].argmax(axis=1)),'tune':metrics(y[tune],p[name][tune].argmax(axis=1))}
    selection['deployment']=max(selection['candidates'],key=lambda n:selection['candidates'][n]['tune']['macro_f1'])
    selection['class_followup']={'protocol_sha256':sha(ROOT/'configs/class_followup.json'),'training_code_sha256':sha(ROOT/'scripts/refine_classes.py'),'previous_deployment':original,'test_accessed':False}
    selection['finalized_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(out/'selection.json').write_text(json.dumps(selection,indent=2));(out/'training_trace.json').write_text(json.dumps(trace,indent=2))
    chosen=selection['candidates'][selection['deployment']];print(json.dumps({'selected':selection['deployment'],'train':chosen['train']['accuracy'],'tune':chosen['tune']['accuracy'],'tune_f1':chosen['tune']['macro_f1']},indent=2),flush=True)


if __name__=='__main__':main()
