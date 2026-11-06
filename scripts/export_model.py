"""Export the development-selected model without accessing test predictions."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from study import ROOT,sha
from cricketshot.policy import CLASSES


def export(out,deploy=False):
    out=Path(out);selection=json.loads((out/'selection.json').read_text());policy=json.loads((out/'policy.json').read_text())
    assert selection['scope_fitted'] and selection['deployment']==policy['deployment']
    models=out/'deployment';models.mkdir(exist_ok=True);members=selection['candidates'][selection['deployment']]['members'];bundle={}
    for name in members:
        with np.load(out/(name+'.npz'),allow_pickle=False) as z:bundle.update({name+'__'+k:z[k].copy() for k in z.files})
    np.savez_compressed(models/'heads.npz',**bundle)
    with np.load(out/'scope_head.npz',allow_pickle=False) as z:np.savez_compressed(models/'scope_head.npz',**{k:z[k].copy() for k in z.files})
    baseline_names=[n for n in selection['candidates'] if selection['heads'].get(n,{}).get('representation')=='video_mean'];baseline=max(baseline_names,key=lambda n:selection['candidates'][n]['tune']['macro_f1'])
    shutil.copy2(out/(baseline+'.npz'),models/'baseline.npz')
    metadata={'schema_version':4,'classes':CLASSES,'name':'Six-family selective recognizer','deployment':selection['deployment'],'members':[dict(name=n,**selection['heads'][n]) for n in members],'threshold':policy['threshold'],'calibration':policy['calibration'],'baseline':baseline,'baseline_sha256':sha(models/'baseline.npz'),'heads_sha256':sha(models/'heads.npz'),'scope_sha256':sha(models/'scope_head.npz'),'protocol_sha256':selection['protocol_sha256'],'manifest_sha256':selection['manifest_sha256'],'decision_source_sha256':policy['policy_code_sha256'],'feature_source_sha256':sha(ROOT/'src/cricketshot/features.py')}
    metadata['scope_representation']=selection.get('scope_representation','context_mean')
    metadata['scope_followup']=selection.get('scope_followup')
    (models/'classifier.json').write_text(json.dumps(metadata,indent=2))
    if deploy:
        for name in ['heads.npz','scope_head.npz','baseline.npz','classifier.json']:shutil.copy2(models/name,ROOT/'models'/name)
    return metadata


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='results');parser.add_argument('--deploy',action='store_true');args=parser.parse_args();m=export(ROOT/args.output_dir,args.deploy)
    print(json.dumps({'deployment':m['deployment'],'threshold':m['threshold'],'installed':args.deploy},indent=2))


if __name__=='__main__':main()
