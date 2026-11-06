"""Recorded development follow-up after the linear scope coverage shortfall."""
import json,shutil,time
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from study import ROOT,read,sha,data,fit_transform
from cricketshot.policy import head_logits


def main():
    out=ROOT/'results';cfg=read('configs/scope_experiment.json');selection=read('results/selection.json')
    if (out/'metrics.json').exists() or selection.get('scope_followup'):raise RuntimeError('Follow-up already fitted or test opened.')
    baseline=out/'scope_baseline';baseline.mkdir(exist_ok=False)
    for name in ['selection.json','scope_head.npz','calibration.json','policy.json']:shutil.copy2(out/name,baseline/name)
    rows,x,y,splits,groups=data();train=splits=='train';tune=splits=='tune';target=(y>=0).astype(np.float32)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    best=max(selection['scope_grid'],key=lambda r:r['tune_auroc'])['tune_auroc'];chosen={'name':'linear_baseline','representation':'context_mean','head_type':'linear','C':selection['scope_C'],'tune_auroc':best}
    with np.load(baseline/'scope_head.npz',allow_pickle=False) as z:best_head={k:z[k].copy() for k in z.files}
    records=[chosen.copy()];traces=[]
    for spec in cfg['representations']:
        state,z=fit_transform(x[spec['name']],train,spec['pca']);tx=torch.from_numpy(z[train]);vx=torch.from_numpy(z[tune]);ty=torch.from_numpy(target[train,None])
        positive=float((ty==1).sum());negative=len(ty)-positive;positive_weight=negative/positive
        for seed in cfg['seeds']:
            torch.manual_seed(seed);np.random.seed(seed);name=f"{spec['name']}_pca{spec['pca']}_h{spec['hidden']}_scope_seed{seed}"
            model=torch.nn.Sequential(torch.nn.Linear(spec['pca'],spec['hidden']),torch.nn.ReLU(),torch.nn.Dropout(cfg['dropout']),torch.nn.Linear(spec['hidden'],1))
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
            local_best=-1.;stale=0;best_epoch=None;local_head=None
            for epoch in range(1,cfg['max_epochs']+1):
                model.train();loss_sum=0.
                for idx in torch.randperm(len(tx)).split(cfg['batch_size']):
                    optimizer.zero_grad(set_to_none=True);labels=ty[idx]*(1-cfg['label_smoothing'])+cfg['label_smoothing']/2;loss=lossfn(model(tx[idx]+torch.randn_like(tx[idx])*cfg['feature_noise_std']),labels);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
                model.eval()
                with torch.inference_mode():tr=model(tx).numpy().ravel();va=model(vx).numpy().ravel()
                auc=float(roc_auc_score(target[tune],va));tr_auc=float(roc_auc_score(target[train],tr));traces.append({'head':name,'epoch':epoch,'training_loss':loss_sum/len(tx),'train_auroc':tr_auc,'tune_auroc':auc})
                if auc>local_best+1e-12:
                    local_best=auc;stale=0;best_epoch=epoch;local_head={**state,'weight1':model[0].weight.detach().numpy().copy(),'bias1':model[0].bias.detach().numpy().copy(),'weight':model[3].weight.detach().numpy().copy(),'bias':model[3].bias.detach().numpy().copy()}
                else:stale+=1
                if stale>=cfg['patience']:break
            np.savez_compressed(out/(name+'.npz'),**local_head)
            record={'name':name,'representation':spec['name'],'pca':spec['pca'],'hidden':spec['hidden'],'seed':seed,'head_type':'mlp','best_epoch':best_epoch,'tune_auroc':local_best};records.append(record)
            if local_best>best+1e-12:best=local_best;best_head=local_head;chosen=record
            print(name,'epoch',best_epoch,'tune AUROC',round(local_best,4),flush=True)
    np.savez_compressed(out/'scope_head.npz',**best_head)
    selection['scope_followup']={'protocol_sha256':sha(ROOT/'configs/scope_experiment.json'),'training_code_sha256':sha(ROOT/'scripts/refine_scope.py'),'reason':cfg['reason'],'candidates':records,'selected':chosen,'test_accessed':False}
    selection['scope_representation']=chosen['representation'];selection['finalized_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'selection.json').write_text(json.dumps(selection,indent=2));(out/'scope_trace.json').write_text(json.dumps(traces,indent=2))
    for name in ['calibration.json','policy.json']:(out/name).unlink()
    print(json.dumps({'chosen_scope':chosen,'baseline_preserved':'results/scope_baseline'},indent=2),flush=True)


if __name__=='__main__':main()
