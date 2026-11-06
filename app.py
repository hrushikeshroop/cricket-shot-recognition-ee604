"""Local six-family demo: one supported prediction or an explicit review request."""
from pathlib import Path
import sys,base64,io,json,tempfile,threading
import numpy as np
from PIL import Image
from flask import Flask,jsonify,render_template,request,send_file
from werkzeug.exceptions import RequestEntityTooLarge
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from cricketshot.recognizer import Recognizer


def create_app(recognizer=None):
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=100*1024*1024;holder=[recognizer];lock=threading.Lock()
    def examples():return json.loads((ROOT/'demo/examples.json').read_text())
    def find(identifier):return next((e for e in examples() if e['id']==identifier),None)
    def infer(path,start=0.,end=None):
        with lock:
            if holder[0] is None:holder[0]=Recognizer()
            result,frames=holder[0].predict(path,start,end)
        result['previews']={}
        for name,sequence in frames.items():
            previews=[]
            for i in np.rint(np.linspace(0,len(sequence)-1,8)).astype(int):
                im=Image.fromarray(sequence[i]);im.thumbnail((240,160));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);previews.append('data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii'))
            result['previews'][name]=previews
        return result
    @app.context_processor
    def identity():return {'submission':json.loads((ROOT/'configs/submission.json').read_text())}
    @app.get('/')
    @app.get('/video')
    def index():return render_template('index.html')
    @app.get('/api/status')
    def status():return jsonify(ready=all((ROOT/'models'/f).is_file() for f in ['classifier.json','heads.npz','scope_head.npz','mc3_features.onnx','mobilenet_features.onnx']),version='4.0.0',classes=6)
    @app.get('/api/evidence')
    def evidence():return jsonify(json.loads((ROOT/'results/metrics.json').read_text()))
    @app.get('/api/examples')
    def example_list():return jsonify([{k:v for k,v in e.items() if k not in {'path','thumbnail'}} for e in examples()])
    @app.get('/api/example/<identifier>/thumbnail')
    def thumbnail(identifier):
        e=find(identifier);return send_file(ROOT/e['thumbnail'],mimetype='image/jpeg') if e else (jsonify(error='Example not found'),404)
    @app.post('/api/example/<identifier>')
    def example(identifier):
        e=find(identifier)
        if not e:return jsonify(error='Example not found'),404
        result=infer(ROOT/e['path']);result['example']={k:v for k,v in e.items() if k not in {'path','thumbnail'}};return jsonify(result)
    @app.post('/api/predict')
    def predict():
        upload=request.files.get('video')
        if not upload or not upload.filename:return jsonify(error='Choose a video first.'),400
        suffix=Path(upload.filename).suffix.lower()
        if suffix not in {'.avi','.mp4','.mov','.mkv','.webm'}:return jsonify(error='Use AVI, MP4, MOV, MKV or WebM.'),400
        start=float(request.form.get('start') or 0);end=float(request.form['end']) if request.form.get('end') else None
        with tempfile.TemporaryDirectory(prefix='cricketshot6_') as folder:
            path=Path(folder)/('clip'+suffix);upload.save(path);return jsonify(infer(path,start,end))
    @app.errorhandler(ValueError)
    def invalid(exc):return jsonify(error=str(exc)),400
    @app.errorhandler(FileNotFoundError)
    def missing(exc):return jsonify(error='Project files are missing. Follow README setup.'),503
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):return jsonify(error='Upload below 100 MB, then select a batting interval up to five seconds.'),413
    return app


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=6040);args=parser.parse_args()
    create_app().run(host='127.0.0.1',port=args.port,debug=False,threaded=True)
