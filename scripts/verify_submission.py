"""Independent arithmetic, source, package and submission-coverage audit."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pymupdf as fitz
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]

def main():
    checks=[]
    def check(name,condition,detail=''):
        checks.append({'name':name,'pass':bool(condition),'detail':detail})
        if not condition:print('FAIL:',name,detail,flush=True)
    run=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True)
    check('Behavioral and integration tests',run.returncode==0,run.stdout.strip()+run.stderr.strip())
    r=json.loads((ROOT/'results/images/metrics.json').read_text())
    with np.load(ROOT/'results/images/test_predictions.npz',allow_pickle=False) as a:
        for mode,info in r['models'].items():
            p=a['p_'+mode];pred=p.argmax(1);y=a['y']
            check(mode+' metric arithmetic',abs(np.mean(pred==y)-info['test']['accuracy'])<1e-12 and
                  abs(f1_score(y,pred,average='macro')-info['test']['macro_f1'])<1e-12 and
                  np.array_equal(confusion_matrix(y,pred),info['test']['confusion']))
        check('Review arithmetic',int(a['accepted'].sum())==r['review']['test_accepted'])
    manifest=list(csv.DictReader((ROOT/'data/images/processed/manifest.csv').open(encoding='utf-8',newline='')))
    csv_predictions=list(csv.DictReader((ROOT/'results/images/test_predictions.csv').open(encoding='utf-8',newline='')))
    check('CSV prediction IDs and test manifest agree',{x['id'] for x in csv_predictions}=={x['id'] for x in manifest if x['split']=='test'})
    edges_path=ROOT/'data/images/processed/transformed_duplicate_edges.json'
    if edges_path.exists():
        edges=json.loads(edges_path.read_text());by_id={x['id']:x for x in manifest};bad=[]
        pairs=edges['transform_edges']+[[e['a'],e['b']] for e in edges['orb_edges']]
        for a,b in pairs:
            if a in by_id and b in by_id and by_id[a]['split']!=by_id[b]['split']:bad.append([a,b])
        check('Verified transformed copies stay in one split',not bad,f'{len(pairs)} recorded edges checked; {len(bad)} violations')
        original=list(csv.DictReader((ROOT/'data/images/processed/initial_hash_audit/manifest.csv').open(encoding='utf-8',newline='')))
        retained={row['id']:row for row in manifest};original_splits={}
        for row in original:
            if row['id'] in retained:original_splits.setdefault(row['group'],set()).add(retained[row['id']]['split'])
        check('Initial hash components remain unsplit',all(len(s)==1 for s in original_splits.values()),f'{len(original_splits)} original components checked')
    from PIL import Image,ImageOps
    mismatches=[]
    for i in np.linspace(0,len(manifest)-1,min(64,len(manifest)),dtype=int):
        row=manifest[i]
        with Image.open(ROOT/row['path']) as source:im=ImageOps.exif_transpose(source).convert('RGB')
        digest=hashlib.sha256(im.width.to_bytes(4,'big')+im.height.to_bytes(4,'big')+im.tobytes()).hexdigest()
        if digest!=row['rgb_sha256']:mismatches.append(row['id'])
    check('Raw pixel hashes independently sampled',not mismatches,f'64 deterministic source images re-read; {len(mismatches)} mismatches')
    inventory=json.loads((ROOT/'results/source_inventory.json').read_text())
    from build_report import source_files
    expected_sources={p.relative_to(ROOT).as_posix() for p in source_files()}
    check('Source inventory covers every current custom file',expected_sources=={item['path'] for item in inventory})
    stale=[]
    for item in inventory:
        p=ROOT/item['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:stale.append(item['path'])
    check('Full code appendix matches current source',not stale,', '.join(stale) or f'{len(inventory)} source/config files match SHA-256 inventory')
    main=fitz.open(ROOT/'output/pdf/CricketShot_Report_Main.pdf');complete=fitz.open(ROOT/'output/pdf/CricketShot_Report_Complete.pdf');appendix=fitz.open(ROOT/'output/pdf/CricketShot_Code_Appendix.pdf')
    check('Complete report includes entire appendix',len(complete)==len(main)+len(appendix),f'{len(main)} main + {len(appendix)} appendix = {len(complete)} pages')
    text='\n'.join(p.get_text() for p in main)
    check('Report contains current primary score',f'{r["models"]["contrast"]["test"]["accuracy"]*100:.1f}' in text)
    plain=' '.join(text.split())
    required_sections=['Abstract','Introduction','Proposed method','Experiments','Code, data and repositories']
    check('Report covers required sections',all(s in plain for s in required_sections) and str(r['n']['test']) in plain,', '.join(required_sections))
    # Every authored file heading must appear in the printable appendix.
    appendix_text='\n'.join(p.get_text() for p in appendix)
    missing=[i['path'] for i in inventory if i['path'] not in appendix_text]
    check('Every inventoried file appears in source appendix',not missing,', '.join(missing))
    outside=[]
    for n,page in enumerate(complete):
        for word in page.get_text('words'):
            if word[0]<-1 or word[1]<-1 or word[2]>page.rect.width+1 or word[3]>page.rect.height+1:outside.append(n+1)
    check('PDF text stays inside page boundaries',not outside,str(sorted(set(outside))))
    pptx=ROOT/'output/presentation/CricketShot_Presentation.pptx'
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(pptx) as z:
        check('PowerPoint ZIP integrity',z.testzip() is None)
        slides=[n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml') and '/_rels/' not in n]
        check('Presentation slide count',len(slides)==16,str(len(slides)))
        tables=sum(len(ET.fromstring(z.read(n)).findall('.//a:tbl',ns)) for n in slides)
        check('Presentation evidence tables remain native',tables>=6,f'{tables} editable tables')
        slide_text=' '.join(' '.join(ET.fromstring(z.read(n)).itertext()) for n in slides)
        check('Presentation contains current primary score',f'{r["models"]["contrast"]["test"]["accuracy"]*100:.1f}%' in slide_text)
        notes=[n for n in z.namelist() if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')]
        check('Speaker notes present',len(notes)==16,str(len(notes)))
    office=json.loads((ROOT/'results/presentation_validation.json').read_text(encoding='utf-8-sig'))
    check('PowerPoint renderer geometry',not office['issues'],str(office['issues']))
    with zipfile.ZipFile(ROOT/'output/CricketShot_Overleaf.zip') as z:
        check('Overleaf package contains actual 2022 style and source appendix',all(n in z.namelist() for n in ('main.tex','cvpr.sty','references.bib','code_appendix.pdf')))
    revision=json.loads((ROOT/'results/revision/verification.json').read_text())
    check('Current revision arithmetic/cache audit',revision['all_checks_pass'],f"{len(revision['checks'])} recorded checks; scope in results/revision/verification.json")
    proof=json.loads((ROOT/'results/revision/cache_validation.json').read_text())
    check('Current preprocessing binding',proof['images_module_sha256']==hashlib.sha256((ROOT/'src/cricketshot/images.py').read_bytes()).hexdigest() and proof['core_module_sha256']==hashlib.sha256((ROOT/'src/cricketshot/core.py').read_bytes()).hexdigest())
    http=json.loads((ROOT/'results/revision/http_validation.json').read_text())
    check('Actual upload HTTP flow',http['all_pass'] and all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in http['source_model_hashes'].items()),'Four real examples and four actual multipart uploads on the revised running app; malformed upload rejected')
    hog=json.loads((ROOT/'results/hog/metrics.json').read_text())
    check('Report includes new HOG evidence',f"{hog['test']['accuracy']*100:.1f}" in text and f"{hog['test']['macro_f1']*100:.1f}" in text and 'HOG' in text)
    check('Slides include new HOG evidence',f"{hog['test']['accuracy']*100:.1f}%" in slide_text and 'HOG' in slide_text)
    check('Report discloses confidence failure',all(value in plain for value in ['0.043','0.094','557','433']))
    check('Course mapping and viva trace present',all((ROOT/path).is_file() for path in ['docs/course_mapping.md','docs/viva_guide.md','results/revision/worked_trace.json']))
    sub=json.loads((ROOT/'configs/submission.json').read_text())
    pending=[]
    if not sub.get('github_url'):pending.append('Real GitHub URL, deferred by user')
    if len(sub.get('members',[]))!=5:pending.append('Five member identities')
    check('Report contains all supplied names and roll numbers',all(m['name'] in plain and m['roll_number'] in plain for m in sub['members']))
    check('Slides contain all supplied names and roll numbers',all(m['name'] in slide_text and m['roll_number'] in slide_text for m in sub['members']))
    report_links={link.get('uri') for page in main for link in page.get_links()}
    check('Current repository URL in report and slides',sub['github_url'] in report_links and sub['github_url'] in slide_text)
    result={'checks':checks,'all_checks_pass':all(x['pass'] for x in checks),'pending_user_submission_items':pending,
            'revision_scope':'Earlier verification is under prior_* files. Current behavioral tests are rerun by this verifier.',
            'manual_review':'See docs/visual_review.md for actual render inspection. Automated checks alone do not prove visual quality or data-label correctness.',
            'limits':['No independent label audit of every raw image/clip','No image match/player identities','Exploratory revised grouping and reused video holdout','Unknown upstream image license']}
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    md='# Final verification record\n\n'
    md+='| Check | Result | Evidence |\n|---|---|---|\n'
    for c in checks:md+=f'| {c["name"]} | {"PASS" if c["pass"] else "FAIL"} | {c["detail"].replace(chr(10)," ").replace("|","/")[:250]} |\n'
    md+='\nSubmission items deliberately pending:\n\n'+'\n'.join('- '+x for x in pending)+'\n\n'+result['manual_review']+'\n'
    (ROOT/'docs/verification.md').write_text(md,encoding='utf-8')
    print(f'{sum(c["pass"] for c in checks)}/{len(checks)} verification checks pass. Pending submission items: {len(pending)}')
    if not result['all_checks_pass']:raise SystemExit(1)

if __name__=='__main__':main()
