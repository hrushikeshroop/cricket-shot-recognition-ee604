"""Figures, selected examples and latency for the four-class image experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import csv
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from cricketshot.images import ImageRecognizer,LABELS,load_image,image_views


def write_demo_manifest(by_id, ids, chosen, fail, dis):
    """Reuse shipped illustrations; keep only the disagreement source optional.

    JPEG illustrations differ from original decoded PNG inputs. Predictions
    below are recomputed for these actual demo bytes; benchmark predictions
    and model files are never changed by selecting demonstration examples.
    """
    model=ImageRecognizer();entries=[]
    items=[(index, f"slides/assets/class_{j}.jpg", "success", True) for j,index in enumerate(chosen)]
    if fail is not None:items.append((fail,"slides/assets/failure_original.jpg","confident_error",True))
    if dis is not None:items.append((dis,"demo/images/"+str(ids[dis])+".png","view_disagreement",False))
    for index,path,role,bundled in items:
        row=by_id[str(ids[index])];target=ROOT/path
        if not bundled:
            target.parent.mkdir(exist_ok=True,parents=True)
            load_image(ROOT/row['path']).save(target)
        result,_=model.predict(target)
        if role=='success' and result['label']!=row['label']:raise ValueError('Selected illustration no longer demonstrates success')
        if role=='confident_error' and result['label']==row['label']:raise ValueError('Selected failure no longer demonstrates an error')
        if role=='view_disagreement' and result['prediction_consistent']:raise ValueError('Selected disagreement no longer requests review')
        variant='Shipped JPEG illustration' if bundled else 'Original decoded image'
        description={'success':'Selected success','confident_error':'Confident failure','view_disagreement':'View disagreement'}[role]
        entries.append({'id':row['id'],'label':row['label'],'path':path,'split':'test',
                        'short_label':LABELS[row['label']], 'demo_role':role,'bundled':bundled,
                        'input_variant':variant,'example_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'expected_prediction':result['label'],
                        'note':description+'. Chosen after evaluation. '+variant+'; displayed scores are recomputed from this file.'})
    (ROOT/'demo/image_examples.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    return entries

def main():
    out=ROOT/'results/images/figures';out.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'slides/assets';assets.mkdir(parents=True,exist_ok=True)
    examples=ROOT/'demo/images';examples.mkdir(parents=True,exist_ok=True)
    r=json.loads((ROOT/'results/images/metrics.json').read_text());classes=r['classes']
    audit=json.loads((ROOT/'data/images/processed/audit.json').read_text())
    rows=list(csv.DictReader((ROOT/'data/images/processed/manifest.csv').open(encoding='utf-8',newline='')));by_id={x['id']:x for x in rows}
    with np.load(ROOT/'results/images/test_predictions.npz',allow_pickle=False) as a:
        y,p,ids,agreement,accepted=a['y'],a['p_contrast'],a['ids'],a['agreement'],a['accepted']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    chosen=[]
    for ci,c in enumerate(classes):
        candidates=np.flatnonzero((y==ci)&(p.argmax(1)==ci));candidates=candidates if len(candidates) else np.flatnonzero(y==ci)
        jpeg_candidates=[int(i) for i in candidates if Path(by_id[str(ids[i])]['path']).suffix.lower() in ('.jpg','.jpeg')]
        pool=np.array(jpeg_candidates,dtype=int) if jpeg_candidates else candidates
        chosen.append(int(pool[np.argmax(p[pool,ci])]))
    wrong=np.flatnonzero(p.argmax(1)!=y);fail=int(wrong[np.argmax(p[wrong].max(1))]) if len(wrong) else None
    low=int(np.argmin(p.max(1)));disagree=np.flatnonzero(~agreement);dis=int(disagree[0]) if len(disagree) else low
    # A four-image sheet of dataset labels. Each image is an actual held-out example.
    sheet=Image.new('RGB',(1200,720),'#f7f8f1');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(chosen):
        row=by_id[str(ids[i])];im=load_image(ROOT/row['path']);tile=ImageOps.contain(im,(570,300));x=(j%2)*600+15;yy=(j//2)*360+40
        sheet.paste(tile,(int(x+(570-tile.width)/2),yy));draw.text((x,yy-30),LABELS[row['label']],fill='#254832',font=font)
        im.save(assets/f'class_{j}.jpg',quality=95)
    sheet.save(out/'class_examples.jpg',quality=95)
    if fail is not None:
        im=load_image(ROOT/by_id[str(ids[fail])]['path']);views=image_views(im)
        sheet=Image.new('RGB',(1200,520),'white');d=ImageDraw.Draw(sheet)
        title=f'Dataset: {LABELS[classes[y[fail]]]} / Predicted: {LABELS[classes[p[fail].argmax()]]} / Score {p[fail].max()*100:.1f}%'
        d.text((15,10),title,fill='#254832',font=font)
        for j,v in enumerate(views):
            tile=ImageOps.contain(v,(570,445));sheet.paste(tile,(15+j*600,60))
        sheet.save(out/'failure.jpg',quality=95)
        im.save(assets/'failure_original.jpg',quality=95)
    write_demo_manifest(by_id,ids,chosen,fail,dis)
    cm=np.array(r['models']['contrast']['test']['confusion']);cmn=cm/cm.sum(1,keepdims=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    labels=['Drive','Glance/flick','Pull','Sweep']
    fig,ax=plt.subplots(figsize=(4.8,4.2));img=ax.imshow(cmn,vmin=0,vmax=1,cmap='Greens')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cmn[i,j]>.6 else '#233b2a',fontsize=11)
    ax.set_xticks(range(4),labels);ax.set_yticks(range(4),labels);ax.set_xlabel('Predicted label');ax.set_ylabel('Dataset label');fig.colorbar(img,ax=ax,shrink=.8,label='Within-class fraction');fig.tight_layout();fig.savefig(out/'confusion.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,2.7));x=np.arange(4);bottom=np.zeros(4)
    for split,color in [('train','#3c6b46'),('val','#9eae8c'),('test','#b8754b')]:
        vals=[audit['split_counts'][split][c] for c in classes];ax.bar(x,vals,bottom=bottom,label=split,color=color);bottom+=vals
    ax.set_xticks(x,labels);ax.set_ylabel('Retained images');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'distribution.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.8,3.0));conf=p.max(1);correct=p.argmax(1)==y
    for use_agreement,name,color in [(False,'Confidence only','#9eae8c'),(True,'Confidence + agreement','#3c6b46')]:
        xx=[];yy=[]
        for th in np.linspace(0,1,101):
            keep=(conf>=th)&(agreement if use_agreement else True)
            if keep.any():xx.append(keep.mean()*100);yy.append(correct[keep].mean()*100)
        ax.plot(xx,yy,label=name,color=color)
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_xlabel('Test coverage (%)');ax.set_ylabel('Accepted accuracy (%)');ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'review_curve.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,2)
    items=[(.1,'Image + central crop\n224 x 224 inputs'),(2.55,'Shared frozen encoder\n2 x 1280 features'),(5.0,'Concatenate vectors\n+ absolute difference'),(7.45,'Linear head + temperature\nagreement review gate')]
    for x,text in items:ax.add_patch(FancyBboxPatch((x,.55),2.1,.9,boxstyle='round,pad=.05',edgecolor='#3c6b46',facecolor='#edf2e5'));ax.text(x+1.05,1,text,ha='center',va='center',fontsize=8)
    for x in (2.25,4.7,7.15):ax.annotate('',xy=(x+.22,1),xytext=(x,1),arrowprops={'arrowstyle':'->','color':'#3c6b46'})
    fig.tight_layout(pad=.1);fig.savefig(out/'method.pdf',bbox_inches='tight');fig.savefig(out/'method.png',bbox_inches='tight');plt.close(fig)
    # End-to-end timing, one warmup, 50 uniformly indexed held-out images.
    model=ImageRecognizer();test=[by_id[str(i)] for i in ids];model.predict(ROOT/test[0]['path']);times=[]
    for i in np.linspace(0,len(test)-1,min(50,len(test)),dtype=int):
        result,_=model.predict(ROOT/test[i]['path']);times.append(result['seconds'])
    latency={'n':len(times),'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
             'scope':'Image decoding, two CNN views and all three inference heads. One warmup, model initialization excluded.',
             'times_seconds':times,'onnx_threads':2}
    (ROOT/'results/images/latency.json').write_text(json.dumps(latency,indent=2),encoding='utf-8')
    print('Image figures, examples and latency saved',flush=True)

if __name__=='__main__':main()
