"""Create report figures and honest demo examples directly from saved predictions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import csv
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from cricketshot.core import decode_video, DISPLAY

GREEN, ORANGE, GRAY = '#326847', '#bc7045', '#98a58e'

def main():
    out = ROOT / 'results/figures'
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / 'demo/clips').mkdir(parents=True, exist_ok=True)
    (ROOT / 'demo/thumbnails').mkdir(parents=True, exist_ok=True)
    r = json.loads((ROOT / 'results/metrics.json').read_text())
    audit = json.loads((ROOT / 'data/processed/audit.json').read_text())
    rows = list(csv.DictReader((ROOT / 'data/processed/manifest.csv').open(encoding='utf-8', newline='')))
    by_id = {row['id']: row for row in rows}
    with np.load(ROOT / 'results/test_predictions.npz', allow_pickle=False) as a:
        y, p, ids = a['y'], a['p_phase_delta'], a['ids']
    classes = r['classes']
    labels = [DISPLAY[c] for c in classes]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.dpi': 200})
    fig, ax = plt.subplots(figsize=(7, 2.9))
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for s, color in [('train',GREEN),('val',GRAY),('test',ORANGE)]:
        values = [audit['split_counts'][s].get(c, 0) for c in classes]
        ax.bar(x, values, bottom=bottom, label={'train':'Train','val':'Validation','test':'Test'}[s], color=color)
        bottom += values
    ax.set_xticks(x, labels, rotation=20, ha='right'); ax.set_ylabel('Clips'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/'class_distribution.png'); plt.close(fig)
    modes = ['middle','mean','phase','phase_delta']
    fig, ax = plt.subplots(figsize=(7, 3.0))
    x = np.arange(4); width=.32
    ax.bar(x-width/2, [r['models'][m]['test']['accuracy']*100 for m in modes],width,color=GREEN,label='Accuracy')
    ax.bar(x+width/2, [r['models'][m]['test']['macro_f1']*100 for m in modes],width,color=ORANGE,label='Macro F1')
    ax.set_xticks(x,['Middle frame','Mean','Phase','Phase + delta']);ax.set_ylim(0,100);ax.set_ylabel('Test score (%)');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'ablations.png');plt.close(fig)
    cm = np.array(r['models']['phase_delta']['test']['confusion'])
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4,4.6))
    im=ax.imshow(norm,cmap='Greens',vmin=0,vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.6 else '#24382a')
    ax.set_xticks(range(8),labels,rotation=45,ha='right');ax.set_yticks(range(8),labels)
    ax.set_xlabel('Predicted label');ax.set_ylabel('Dataset label')
    fig.colorbar(im,ax=ax,shrink=.75,label='Within-class fraction')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png');plt.close(fig)
    fig, ax = plt.subplots(figsize=(4,3))
    confidence=p.max(1);correct=p.argmax(1)==y
    thresholds=np.linspace(0,1,101);cov=[];acc=[]
    for th in thresholds:
        mask=confidence>=th
        if mask.any():cov.append(mask.mean()*100);acc.append(correct[mask].mean()*100)
    ax.plot(cov,acc,color=GREEN,linewidth=2)
    model=r['models']['phase_delta'];sel=model['test_selective']
    if sel['accepted']:
        ax.scatter(sel['coverage']*100,sel['accuracy']*100,color=ORANGE,zorder=3,label='Validation-chosen threshold')
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_xlabel('Test coverage (%)');ax.set_ylabel('Accuracy on accepted clips (%)')
    if sel['accepted']:ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();fig.savefig(out/'selective_curve.png');plt.close(fig)
    # Actual early/middle/late samples, with labels only from the upstream dataset.
    font_path='C:/Windows/Fonts/arial.ttf'
    font=ImageFont.truetype(font_path,18) if Path(font_path).exists() else ImageFont.load_default()
    small=ImageFont.truetype(font_path,13) if Path(font_path).exists() else ImageFont.load_default()
    chosen=[]
    for ci, c in enumerate(classes):
        candidates=np.flatnonzero(y==ci)
        # Deterministic high confidence per class, documented as selected examples.
        correct_candidates=candidates[p[candidates].argmax(1)==ci]
        use=correct_candidates if len(correct_candidates) else candidates
        i=int(use[np.argmax(p[use,ci])]); chosen.append(i)
    failures=np.flatnonzero(p.argmax(1)!=y)
    fail=int(failures[np.argmax(confidence[failures])]) if len(failures) else None
    uncertain=int(np.argmin(confidence))
    demo_choices=[]
    # Four visible examples include a failure and low confidence case when distinct.
    for i,tag in [(chosen[3],'Selected correct example'),(chosen[5],'Selected correct example'),
                  (fail,'Confident failure example'),(uncertain,'Low-confidence example')]:
        if i is not None and i not in [x[0] for x in demo_choices]:
            demo_choices.append((i,tag))
    entries=[]
    for n,(i,tag) in enumerate(demo_choices):
        row=by_id[str(ids[i])];src=ROOT/row['path'];suffix=src.suffix
        dest=ROOT/f'demo/clips/{row["id"]}{suffix}';shutil.copy2(src,dest)
        frames,_=decode_video(src)
        thumb=f'demo/thumbnails/{row["id"]}.jpg'
        Image.fromarray(frames[4]).save(ROOT/thumb,quality=90)
        entries.append({'id':row['id'],'label':row['label'],'path':dest.relative_to(ROOT).as_posix(),
                        'thumbnail':thumb,'note':tag+'. Chosen after evaluation for demonstration.',
                        'source':row['source'],'split':'test'})
    (ROOT/'demo/examples.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    for page in range(2):
        sheet=Image.new('RGB',(1200,820),'white');draw=ImageDraw.Draw(sheet)
        for k,i in enumerate(chosen[page*4:page*4+4]):
            row=by_id[str(ids[i])];frames,_=decode_video(ROOT/row['path']);yy=k*205
            draw.text((12,yy+7),f'{DISPLAY[row["label"]]} / {row["id"]} / test',fill='#254932',font=font)
            for phase,t in enumerate((1,4,7)):
                tile=ImageOps.contain(Image.fromarray(frames[t]),(370,155))
                sheet.paste(tile,(12+phase*395,yy+37))
        sheet.save(out/f'dataset_examples_{page+1}.jpg',quality=94)
    if fail is not None:
        row=by_id[str(ids[fail])];frames,_=decode_video(ROOT/row['path'])
        sheet=Image.new('RGB',(1200,285),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'Dataset: {DISPLAY[row["label"]]} / Predicted: {DISPLAY[classes[p[fail].argmax()]]} / Score {confidence[fail]*100:.1f}%',fill='#254932',font=font)
        for j,t in enumerate((1,4,7)):
            tile=ImageOps.contain(Image.fromarray(frames[t]),(380,220));sheet.paste(tile,(12+j*395,45))
        sheet.save(out/'failure_example.jpg',quality=94)
    # Original editable method figure for the report, using measured dimensions.
    fig, ax=plt.subplots(figsize=(7.2,1.9));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,2)
    blocks=[(.1,'Trimmed clip\n9 uniform frames'),(2.55,'Frozen MobileNetV2\n9 x 1280 features'),(5.0,'3 phase means\nlate - early + std'),(7.45,'Linear classifier\ncalibration + review')]
    for xx,label in blocks:
        ax.add_patch(FancyBboxPatch((xx,.55),2.1,.9,boxstyle='round,pad=0.07',edgecolor=GREEN,facecolor='#f2f6eb',linewidth=1))
        ax.text(xx+1.05,1,label,ha='center',va='center',fontsize=8)
    for xx in (2.25,4.7,7.15):
        ax.annotate('',xy=(xx+.22,1),xytext=(xx,1),arrowprops={'arrowstyle':'->','color':GREEN})
    fig.tight_layout(pad=.1);fig.savefig(out/'method_overview.pdf',bbox_inches='tight');fig.savefig(out/'method_overview.png',bbox_inches='tight');plt.close(fig)
    print('Figures and demo examples generated from measured predictions.')

if __name__=='__main__':
    main()
