"""Fit six-family heads and a scope head; test rows are never loaded here."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from study import ROOT,read,sha,data,metrics,fit_transform,candidate_probabilities
from cricketshot.policy import CLASSES,head_logits,softmax


def neural(x,y,splits,state,hp,hidden,seed,name,out):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);np.random.seed(seed)
    train=(splits=='train')&(y>=0);tune=(splits=='tune')&(y>=0);tx=torch.from_numpy(x[train]);ty=torch.from_numpy(y[train]);vx=torch.from_numpy(x[tune])
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],hidden),torch.nn.ReLU(),torch.nn.Dropout(hp['dropout']),torch.nn.Linear(hidden,6))
    optimizer=torch.optim.AdamW(model.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
    counts=np.bincount(y[train],minlength=6);weights=len(ty)/(6*counts)
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=hp['label_smoothing'])
    best=-1.;best_head=None;best_epoch=None;stale=0;trace=[]
    for epoch in range(1,hp['max_epochs']+1):
        model.train();loss_sum=0.
        for idx in torch.randperm(len(tx)).split(hp['batch_size']):
            optimizer.zero_grad(set_to_none=True);inputs=tx[idx]+torch.randn_like(tx[idx])*hp['feature_noise_std'];loss=lossfn(model(inputs),ty[idx]);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        model.eval()
        with torch.inference_mode():
            tr=model(tx).numpy();va=model(vx).numpy()
        tm=metrics(y[train],tr.argmax(axis=1));vm=metrics(y[tune],va.argmax(axis=1))
        trace.append({'head':name,'epoch':epoch,'training_cross_entropy':loss_sum/len(tx),'train_accuracy':tm['accuracy'],'train_macro_f1':tm['macro_f1'],'tune_accuracy':vm['accuracy'],'tune_macro_f1':vm['macro_f1']})
        if vm['macro_f1']>best+1e-12:
            best=vm['macro_f1'];best_epoch=epoch;stale=0;best_head={**state,'weight1':model[0].weight.detach().numpy().copy(),'bias1':model[0].bias.detach().numpy().copy(),'weight':model[3].weight.detach().numpy().copy(),'bias':model[3].bias.detach().numpy().copy()}
        else:stale+=1
        if stale>=hp['patience']:break
    np.savez_compressed(out/(name+'.npz'),**best_head)
    print('Neural head',name,'epoch',best_epoch,'tune F1',round(best,4),flush=True)
    return {'head_type':'mlp','seed':seed,'best_epoch':best_epoch,'hidden':hidden},trace


def fit_scope(selected,out,cfg):
    if selected.get('scope_fitted'):raise RuntimeError('Scope selection is already frozen.')
    rows,x,y,splits,groups=data();scope_train=splits=='train';scope_tune=splits=='tune';known=(y>=0).astype(int)
    scope_state,scope_x=fit_transform(x['context_mean'],scope_train,128,cfg['seed']);scope_records=[];best_auc=-1.;scope_head=None
    for c in cfg['linear_C']:
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,tol=1e-5)
        with threadpool_limits(limits=4):model.fit(scope_x[scope_train],known[scope_train]);logits=model.decision_function(scope_x[scope_tune])
        auc=float(roc_auc_score(known[scope_tune],logits));scope_records.append({'C':c,'tune_auroc':auc})
        if auc>best_auc+1e-12:best_auc=auc;scope_head={**scope_state,'weight':model.coef_.astype(np.float32),'bias':model.intercept_.astype(np.float32)};selected['scope_C']=c
    np.savez_compressed(out/'scope_head.npz',**scope_head);selected['scope_grid']=scope_records;selected['scope_fitted']=True
    selected['finalized_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(out/'selection.json').write_text(json.dumps(selected,indent=2))
    print('Scope tune AUROC',best_auc,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='results');parser.add_argument('--phase',choices=['all','classes','scope'],default='all');args=parser.parse_args();out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True);cfg=read('configs/experiment.json')
    if args.phase=='scope':
        if (out/'policy.json').exists() or (out/'metrics.json').exists():raise RuntimeError('Do not refit after calibration or test.')
        selected=json.loads((out/'selection.json').read_text());assert selected['protocol_sha256']==sha(ROOT/'configs/experiment.json');fit_scope(selected,out,cfg);return
    if (out/'selection.json').exists():raise RuntimeError('Use a new output directory rather than overwrite a frozen fit.')
    rows,x,y,splits,groups=data(known_only=True);train=(splits=='train')&(y>=0);tune=(splits=='tune')&(y>=0)
    selected={'protocol_sha256':sha(ROOT/'configs/experiment.json'),'manifest_sha256':sha(ROOT/'data/manifest.csv'),'classes':CLASSES,'heads':{},'ensembles':{},'candidates':{},'test_accessed_during_selection':False,'status':'selection_frozen','fit_roles':['train'],'selection_role':'tune'};traces=[];transforms={}
    def transform(rep,dim):
        key=(rep,dim)
        if key not in transforms:transforms[key]=fit_transform(x[rep],train,dim,cfg['seed'])
        return transforms[key]
    for rep in cfg['representations']:
        dim=0 if rep=='video_mean' else 128;state,z=transform(rep,dim)
        for c in cfg['linear_C']:
            name=rep+'_linear_C'+str(c).replace('.','p')
            model=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,tol=1e-5)
            with threadpool_limits(limits=4):model.fit(z[train],y[train]);pred=model.predict(z[tune])
            head={**state,'weight':model.coef_.astype(np.float32),'bias':model.intercept_.astype(np.float32)};np.savez_compressed(out/(name+'.npz'),**head)
            with threadpool_limits(limits=4):pred=head_logits(x[rep][tune],head).argmax(axis=1)
            m=metrics(y[tune],pred);selected['heads'][name]={'representation':rep,'head_type':'linear','C':c,'pca':dim};selected['candidates'][name]={'tune':m,'members':[name]}
            print('Linear',name,'tune accuracy',round(m['accuracy'],4),'F1',round(m['macro_f1'],4),flush=True)
    for hp in cfg['neural_configs']:
        ensembles={}
        for rep in cfg['neural_representations']:
            state,z=transform(rep,hp['pca']);members=[]
            for seed in cfg['neural_seeds']:
                name=f'{rep}_pca{hp["pca"]}_h{hp["hidden"]}_seed{seed}'
                record,trace=neural(z,y,splits,state,cfg['neural'],hp['hidden'],seed,name,out);traces.extend(trace)
                selected['heads'][name]={**record,'representation':rep,'pca':hp['pca']};members.append(name)
            name=f'{rep}_pca{hp["pca"]}_ensemble';selected['ensembles'][name]={'members':members};ensembles[rep]=members
        name=f'full_motion_pca{hp["pca"]}_ensemble';selected['ensembles'][name]={'members':sum(ensembles.values(),[])}
    probabilities=candidate_probabilities(selected,x,out)
    for name,r in selected['ensembles'].items():selected['candidates'][name]={'tune':metrics(y[tune],probabilities[name][tune].argmax(axis=1)),'members':r['members']}
    for name,r in selected['candidates'].items():r['train']=metrics(y[train],probabilities[name][train].argmax(axis=1))
    chosen=max(selected['candidates'],key=lambda n:selected['candidates'][n]['tune']['macro_f1']);selected['deployment']=chosen
    selected['scope_fitted']=False;selected['finalized_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());selected['training_code_sha256']=sha(Path(__file__))
    (out/'training_trace.json').write_text(json.dumps(traces,indent=2));(out/'selection.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps({'deployment':chosen,'train':selected['candidates'][chosen]['train']['accuracy'],'tune':selected['candidates'][chosen]['tune']['accuracy'],'tune_macro_f1':selected['candidates'][chosen]['tune']['macro_f1']},indent=2),flush=True)
    if args.phase=='all':fit_scope(selected,out,cfg)


if __name__=='__main__':main()
