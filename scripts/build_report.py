"""Shared table formatting and complete source appendix for the report."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import hashlib
import json
import textwrap
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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
