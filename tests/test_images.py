from pathlib import Path
import csv
import io
import json
import sys
import numpy as np
import pytest
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cricketshot.images import ImageRecognizer,image_descriptor,image_views,load_image
from app import create_app

def test_contrast_descriptor_dimensions():
    x=np.random.default_rng(4).normal(size=(3,2,1280)).astype('float32')
    assert image_descriptor(x,'contrast').shape==(3,3840)
    assert image_descriptor(x,'concat').shape==(3,2560)
    assert np.allclose(np.linalg.norm(image_descriptor(x,'contrast'),axis=1),1)

def test_image_group_and_hash_disjointness():
    rows=list(csv.DictReader((ROOT/'data/images/processed/manifest.csv').open(encoding='utf-8',newline='')))
    for key in ('group','rgb_sha256'):
        sets={s:{r[key] for r in rows if r['split']==s} for s in ('train','val','test')}
        assert not sets['train']&sets['test'] and not sets['train']&sets['val'] and not sets['val']&sets['test']
    assert len(rows)==len({r['rgb_sha256'] for r in rows})

def test_image_export_and_metrics_parity():
    model=ImageRecognizer();r=json.loads((ROOT/'results/images/metrics.json').read_text())
    with np.load(ROOT/'data/images/processed/features.npz',allow_pickle=False) as a:
        test=a['features'][a['split']=='test']
    with np.load(ROOT/'results/images/test_predictions.npz',allow_pickle=False) as a:
        for mode in ('global','focus','concat','contrast'):
            p=model.predict_features(test,mode)
            assert np.allclose(p,a['p_'+mode],atol=1e-7)
            assert np.allclose(p.sum(1),1)
            assert abs(np.mean(p.argmax(1)==a['y'])-r['models'][mode]['test']['accuracy'])<1e-12
        keep=(a['p_contrast'].max(1)>=model.config['review_threshold'])&a['agreement']
        assert np.array_equal(keep,a['accepted'])

def test_fresh_image_inference_matches_cached():
    rows=list(csv.DictReader((ROOT/'data/images/processed/manifest.csv').open(encoding='utf-8',newline='')))
    row=next(r for r in rows if r['split']=='test')
    result,_=ImageRecognizer().predict(ROOT/row['path'])
    with np.load(ROOT/'results/images/test_predictions.npz',allow_pickle=False) as a:
        i=list(a['ids']).index(row['id']);assert np.allclose(list(result['probabilities'].values()),a['p_contrast'][i],atol=1e-6)

def test_invalid_images(tmp_path):
    p=tmp_path/'bad.jpg';p.write_bytes(b'not image')
    with pytest.raises(ValueError):load_image(p)
    p=tmp_path/'tiny.png';Image.new('RGB',(3,3)).save(p)
    with pytest.raises(ValueError):load_image(p)

def test_image_api_upload_and_examples():
    c=create_app().test_client();assert c.get('/').status_code==200
    assert c.post('/image-api/predict').status_code==400
    assert c.post('/image-api/predict',data={'image':(io.BytesIO(b'bad'),'bad.jpg')}).status_code==400
    assert c.post('/image-api/predict',data={'image':(io.BytesIO(b'bad'),'bad.svg')}).status_code==400
    e=c.get('/image-api/examples').json;assert len(e)>=3
    assert all(x['split']=='test' and 'path' not in x for x in e)
    p=c.post('/image-api/example/'+e[0]['id']);assert p.status_code==200 and len(p.json['previews'])==2
    assert c.get('/image-api/example/not-real/thumbnail').status_code==404
    # Exercise the real upload endpoint using an actual test example, not a mock.
    sample=json.loads((ROOT/'demo/image_examples.json').read_text())[0]
    with (ROOT/sample['path']).open('rb') as f:
        upload=c.post('/image-api/predict',data={'image':(io.BytesIO(f.read()),'sample.jpg')})
    assert upload.status_code==200
    assert abs(sum(upload.json['probabilities'].values())-1)<1e-8
