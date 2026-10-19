"""Record only the installed packages needed by this project, not unrelated tools."""
from pathlib import Path
import importlib.metadata
import json
import platform
import subprocess
ROOT=Path(__file__).resolve().parents[1]
packages=['numpy','Pillow','onnxruntime','opencv-python-headless','Flask','scikit-learn','scipy','threadpoolctl',
          'onnx','ImageHash','matplotlib','requests','pytest','reportlab','PyMuPDF','blinker','click','itsdangerous',
          'Jinja2','MarkupSafe','Werkzeug','flatbuffers','protobuf','ml_dtypes','PyWavelets','joblib',
          'packaging','python-dateutil','six','contourpy','cycler','fonttools','kiwisolver','pyparsing',
          'certifi','charset-normalizer','idna','urllib3','Pygments','pluggy','iniconfig','colorama']
versions={p:importlib.metadata.version(p) for p in packages}
(ROOT/'requirements-lock.txt').write_text('# Exact versions used on Python '+platform.python_version()+' / Windows x64.\n'+
                                        '\n'.join(f'{p}=={v}' for p,v in sorted(versions.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
(ROOT/'results/environment.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'packages':versions},indent=2),encoding='utf-8')
print('Recorded',len(versions),'package versions')
