"""Shared report utilities and optional supplementary video-pilot manuscript.

Metrics are read from saved experiment evidence, never entered by hand.
Run after benchmarks and tests; use --body-only to compile without code appendix.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import textwrap
import pymupdf as fitz
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from cricketshot.core import DISPLAY

def pct(x):return f'{x*100:.1f}'
def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')

def table(headers, rows, caption, label, spec=None):
    spec=spec or ('l'+'r'*(len(headers)-1))
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    return ('\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{'+spec+'}\n\\toprule\n'+
            ' & '.join(headers)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n')

def source_files():
    paths=[ROOT/'app.py',ROOT/'pyproject.toml',ROOT/'requirements.txt',ROOT/'requirements-dev.txt',
           ROOT/'requirements-lock.txt',ROOT/'run_demo.bat',ROOT/'setup.bat']
    for folder in ('src','scripts','tests','templates','static','configs'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.ps1','.mjs','.js','.css','.html','.json'})
    return sorted(set(p for p in paths if p.exists()),key=lambda p:p.relative_to(ROOT).as_posix())

def code_appendix(path):
    font='CodeFont';pdfmetrics.registerFont(TTFont(font,'C:/Windows/Fonts/consola.ttf'))
    c=canvas.Canvas(str(path),pagesize=(612,792))
    c.setTitle('Cricket Shot Recognition - Complete Project Source Appendix')
    page=0;yy=0;current='';inventory=[]
    def new_page():
        nonlocal page,yy
        if page:c.showPage()
        page+=1;c.setFillColorRGB(.13,.28,.18);c.setFont('Helvetica-Bold',10)
        c.drawString(36,759,'Appendix B / Complete project source')
        c.setFont('Helvetica',7);c.drawRightString(576,759,f'B-{page}')
        c.setStrokeColorRGB(.8,.85,.78);c.line(36,749,576,749)
        c.setFont(font,7);c.setFillColorRGB(.15,.18,.15);c.drawString(36,735,current[:120]);yy=720
    current='Original code and configuration. Line numbers identify source lines. Wrapped lines continue without a number.'
    new_page()
    for p in source_files():
        current=p.relative_to(ROOT).as_posix();data=p.read_bytes();text=data.decode('utf-8-sig')
        inventory.append({'path':current,'sha256':hashlib.sha256(data).hexdigest(),'lines':len(text.splitlines())})
        if yy<110:new_page()
        yy-=15;c.setFont('Helvetica-Bold',9);c.setFillColorRGB(.13,.28,.18);c.drawString(36,yy,current);yy-=16
        c.setFont(font,8);c.setFillColorRGB(.15,.18,.15)
        for n,line in enumerate(text.splitlines(),1):
            chunks=textwrap.wrap(line.expandtabs(4),width=112,replace_whitespace=False,drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or ['']
            for k,chunk in enumerate(chunks):
                if yy<38:
                    new_page();c.setFont(font,8)
                prefix=f'{n:4d}  ' if k==0 else '      '
                c.drawString(36,yy,prefix+chunk);yy-=10.8
        yy-=10
    c.save()
    (ROOT/'results/source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    return page

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--body-only',action='store_true');args=parser.parse_args()
    report=ROOT/'report';g=report/'generated';g.mkdir(exist_ok=True)
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    build=ROOT/'.build/latex/video';build.mkdir(parents=True,exist_ok=True)
    r=json.loads((ROOT/'results/metrics.json').read_text());audit=json.loads((ROOT/'data/processed/audit.json').read_text())
    latency=json.loads((ROOT/'results/latency.json').read_text());submission=json.loads((ROOT/'configs/submission.json').read_text())
    pilot=json.loads((ROOT/'results/pilot_fullframe/metrics.json').read_text())
    rows=list(csv.DictReader((ROOT/'data/processed/manifest.csv').open(encoding='utf-8',newline='')))
    classes=r['classes'];m=r['models']['phase_delta'];middle=r['models']['middle'];mean=r['models']['mean'];phase=r['models']['phase'];b=m['bootstrap']
    counts={s:sum(audit['split_counts'][s].values()) for s in ('train','val','test')}
    data_rows=[[DISPLAY[c]]+[audit['split_counts'][s].get(c,0) for s in ('train','val','test')]+[audit['class_counts'][c]] for c in classes]
    data_rows.append(['Total',counts['train'],counts['val'],counts['test'],audit['decoded_files']])
    (g/'data_table.tex').write_text(table(['Label','Train','Val.','Test','All'],data_rows,'Clip counts from the prepared manifest. All eight labels occur in each set, but validation has only one unorthodox clip.','tab:data'),encoding='utf-8')
    result_rows=[['Training majority','--',pct(r['majority_baseline']['accuracy']),pct(r['majority_baseline']['macro_f1'])]]
    names={'color':'Color histogram','middle':'Middle frame','mean':'Temporal mean','phase':'Ordered phases','phase_delta':'Phase + diff./std.'}
    for mode in names:
        info=r['models'][mode];result_rows.append([names[mode],str(info['C']),pct(info['test']['accuracy']),pct(info['test']['macro_f1'])])
    (g/'results_table.tex').write_text(table(['Representation','$C$','Acc. (\\%)','F1 (\\%)'],result_rows,'All learned heads select regularization by validation macro F1. Final visual models use the same fixed ROI and split. Color uses the full frame. F1 is macro-averaged.','tab:results'),encoding='utf-8')
    cal_rows=[]
    for label,key in [('Before calibration','test_uncalibrated'),('After calibration','test')]:
        mm=m[key];cal_rows.append([label,f'{mm["nll"]:.3f}',f'{mm["ece_10_bins"]:.3f}',f'{mm["brier_multiclass"]:.3f}'])
    (g/'calibration_table.tex').write_text(table(['Probabilities','NLL','ECE','Brier'],cal_rows,'Proposed model on test clips. ECE uses ten equal-width confidence bins. Brier is the unscaled mean sum of squared errors over eight classes. Lower is better.','tab:cal'),encoding='utf-8')
    robustness=[]
    for mode in ('middle','mean','phase_delta'):
        robustness.append([names[mode],pct(r['models'][mode]['test']['accuracy']),pct(r['robustness']['dark'][mode]['accuracy']),pct(r['robustness']['blur'][mode]['accuracy']),pct(r['models'][mode]['temporal_reversal']['accuracy'])])
    (g/'robustness_table.tex').write_text(table(['Representation','Clean','Dark','Blur','Reverse'],robustness,'Test accuracy in percent. Every corruption uses the complete unchanged test set. Reverse changes temporal order only.','tab:robust'),encoding='utf-8')
    delta=(m['test']['macro_f1']-middle['test']['macro_f1'])*100
    finding=(f'The proposed representation exceeds the middle-frame macro F1 by {delta:.1f} percentage points.' if delta>=0 else f'The middle-frame baseline exceeds the proposed representation in macro F1 by {-delta:.1f} percentage points.')
    best=max(r['models'],key=lambda k:r['models'][k]['test']['macro_f1'])
    discussion=(f'Table~\\ref{{tab:results}} shows that the proposed representation has {pct(m["test"]["accuracy"])}\\% accuracy and {pct(m["test"]["macro_f1"])}\\% macro F1. '+finding.replace('percentage points.','percentage points.').replace('%',r'\%')+
                f' The highest test macro F1 among these learned heads is {pct(r["models"][best]["test"]["macro_f1"])}\\%, from {names[best].lower()}. We report that result even when it does not favor the proposed descriptor. '
                f'Adding difference and spread to the ordered phase means changes test macro F1 by {(m["test"]["macro_f1"]-phase["test"]["macro_f1"])*100:+.1f} points. This is a joint ablation. The full-frame pilot and ROI experiment also differ in their input crop, so a crop improvement should not be attributed to temporal pooling alone.')
    recalls=sorted([(c,m['test']['per_class'][c]['recall'],m['test']['per_class'][c]['support']) for c in classes],key=lambda x:x[1])
    low,high=recalls[0],recalls[-1]
    class_text=(f'Figure~\\ref{{fig:confusion}} shows uneven class behavior. {DISPLAY[high[0]]} recall is {pct(high[1])}\\% on {int(high[2])} clips, while {DISPLAY[low[0]].lower()} recall is {pct(low[1])}\\% on {int(low[2])}. Defence has only three test clips, so one changed prediction moves its recall by 33.3 points. The single unorthodox validation clip also makes class-specific tuning unreliable.')
    sel=m['test_selective'];th=m['threshold_selection'];target='The validation target was met.' if th['target_met'] else 'No candidate met the stated validation target, so the 0.80 fallback was used.'
    selective_acc='undefined because no clip is accepted' if sel['accuracy'] is None else f'{pct(sel["accuracy"])}\\%'
    calibration=(f'The fitted temperature is {m["temperature"]:.3f} and the review threshold is {m["review_threshold"]:.2f}. {target} On test clips, the rule accepts {sel["accepted"]} of {m["test"]["n"]} ({pct(sel["coverage"])}\\% coverage), with accepted accuracy {selective_acc}. This operating result is an observation, not a promised 90\\% accuracy. Temperature scaling leaves the confusion matrix unchanged. A calibrated score is still not evidence that an unknown upload belongs to one of the eight classes.')
    reverse=m['temporal_reversal']['accuracy']-m['test']['accuracy']
    reverse_text=('The drop is consistent with some use of temporal order, but it does not identify which movements the model recognizes.' if reverse<0 else 'Reversal does not reduce accuracy here, which weakens the claim that this model has learned a reliable direction of stroke progression.')
    dark=r['robustness']['dark']['phase_delta']['accuracy'];blur=r['robustness']['blur']['phase_delta']['accuracy']
    robust_text=f'The proposed model scores {pct(dark)}\\% in the darkened condition and {pct(blur)}\\% under blur. Neither corruption result is used to retrain or select the model.'
    with np.load(ROOT/'results/test_predictions.npz',allow_pickle=False) as a:
        y,p,ids=a['y'],a['p_phase_delta'],a['ids']
    wrong=np.flatnonzero(p.argmax(1)!=y);i=int(wrong[np.argmax(p[wrong].max(1))]);row=next(x for x in rows if x['id']==str(ids[i]))
    failtext=f'Figure~\\ref{{fig:failure}} is clip \\texttt{{{esc(row["id"])}}}, whose upstream label is {DISPLAY[classes[y[i]]].lower()}. The model predicts {DISPLAY[classes[p[i].argmax()]].lower()} with score {pct(p[i].max())}\\%. We deliberately include this confident error in the demo rather than showing only successful clips.'
    url=submission.get('github_url')
    note='Project repository: \\url{'+url+'}.' if url else 'The group will choose and publish its GitHub repository after reviewing the project. The required repository URL is not yet available and must be inserted before submission.'
    values={'N':str(audit['decoded_files']),'ACC':pct(m['test']['accuracy']),'F1':pct(m['test']['macro_f1']),
            'MID_ACC':pct(middle['test']['accuracy']),'MID_F1':pct(middle['test']['macro_f1']),'ABSTRACT_FINDING':finding,
            'TEST_N':str(counts['test']),'TRAIN_N':str(counts['train']),'VAL_N':str(counts['val']),
            'LARGEST_N':str(max(audit['class_counts'].values())),'SMALLEST_N':str(min(audit['class_counts'].values())),
            'SOURCE_N':str(audit['sources']),'ORIGINAL_SOURCE_N':str(len(set(re.sub(r'_\d+_\d+$','',Path(x['original_name']).stem) for x in rows))),
            'GROUP_N':str(audit['union_groups']),'EXACT_DUP':str(audit['exact_duplicate_files']),'NEAR_DUP':str(len(audit['cross_source_near_duplicates'])),
            'DECODE_ERRORS':str(len(audit['errors'])),'TRAIN_G':str(audit['split_groups']['train']),'VAL_G':str(audit['split_groups']['val']),'TEST_G':str(audit['split_groups']['test']),
            'PILOT_ACC':pct(pilot['models']['phase_delta']['test']['accuracy']),'PILOT_F1':pct(pilot['models']['phase_delta']['test']['macro_f1']),
            'RESULTS_DISCUSSION':discussion,'CLASS_DISCUSSION':class_text,'ACC_LOW':pct(b['accuracy_95'][0]),'ACC_HIGH':pct(b['accuracy_95'][1]),
            'F1_LOW':pct(b['macro_f1_95'][0]),'F1_HIGH':pct(b['macro_f1_95'][1]),'CALIBRATION_DISCUSSION':calibration,
            'REVERSE_ACC':pct(m['temporal_reversal']['accuracy']),'REVERSE_DISCUSSION':reverse_text,'ROBUST_DISCUSSION':robust_text,
            'FAILURE_DISCUSSION':failtext,'LATENCY_N':str(latency['clips']),'MEDIAN_MS':f'{latency["median_seconds"]*1000:.0f}',
            'P95_MS':f'{latency["p95_seconds"]*1000:.0f}','SUBMISSION_NOTE':note,
            'CONCLUSION':finding+' The ablations show how much of the result survives changes in temporal representation and input quality. Because the ROI revision followed pilot test feedback, the results should guide a larger independent evaluation rather than serve as a final generalization claim.'}
    body=(report/'content_template.tex').read_text(encoding='utf-8')
    for key,value in values.items():body=body.replace('{{'+key+'}}',value)
    if re.search(r'\{\{[A-Z_]+\}\}',body):raise ValueError('Unresolved report variables')
    (g/'body.tex').write_text(body,encoding='utf-8')
    (g/'authors.tex').write_text('\\author{'+esc(submission['group_name'])+r'\\Indian Institute of Technology Kanpur\\EE604, October 2026}'+'\n',encoding='utf-8')
    repo=note+'\n\nDataset: \\url{https://www.kaggle.com/datasets/amittalmale/cricket-shots}. Pretrained encoder: \\url{https://github.com/onnx/models}. Libraries: \\url{https://github.com/scikit-learn/scikit-learn}, \\url{https://github.com/microsoft/onnxruntime}, \\url{https://github.com/opencv/opencv}, \\url{https://github.com/pallets/flask}. No other dataset or cricket-model repository is used for training. The archived CVPR 2022 style is from \\url{https://cvpr2022.thecvf.com/author-guidelines}.\n'
    (g/'repository.tex').write_text(repo,encoding='utf-8')
    for name in ('cvpr.sty','ieee_fullname.bst'):
        shutil.copy2(report/'template/latex'/name,report/name)
    compiler=ROOT/'tools/tectonic/tectonic.exe'
    subprocess.run([str(compiler),'--keep-logs','--outdir',str(build),str(report/'main.tex')],cwd=report,check=True)
    shutil.copy2(build/'main.pdf',out/'Video_Pilot_Report.pdf')
    if not args.body_only:
        pages=code_appendix(out/'CricketShot_Code_Appendix.pdf')
        complete=fitz.open(out/'Video_Pilot_Report.pdf')
        appendix=fitz.open(out/'CricketShot_Code_Appendix.pdf');complete.insert_pdf(appendix)
        complete.set_metadata({'title':'Cricket Shot Recognition - EE604 report with full source appendix','author':submission['group_name']})
        complete.save(out/'Video_Pilot_Report_Complete.pdf',garbage=4,deflate=True)
        print('Complete report',len(complete),'pages, source appendix',pages,'pages')
    print('Report generated from measured evidence.')

if __name__=='__main__':main()
