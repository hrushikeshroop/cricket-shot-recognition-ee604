"""Tests target leakage, inference parity, malformed inputs and measured evidence."""
from pathlib import Path
import csv
import io
import json
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cricketshot.core import descriptor, softmax, Recognizer, decode_video
from app import create_app

def test_probabilities_stable_and_normalized():
    p=softmax(np.array([[10000,10001,-10000],[0,0,0]]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(1),1)
    assert p[0].argmax()==1
    with pytest.raises(ValueError):softmax([1,2],0)

def test_temporal_descriptor_obeys_order():
    x=np.random.default_rng(4).normal(size=(2,9,1280)).astype('float32')
    assert descriptor(x,'phase_delta').shape==(2,6400)
    assert np.allclose(descriptor(x,'mean'),descriptor(x[:,::-1],'mean'),atol=1e-6)
    assert not np.allclose(descriptor(x,'phase_delta'),descriptor(x[:,::-1],'phase_delta'))

@pytest.fixture(scope='module')
def evidence():
    path=ROOT/'data/processed/manifest.csv'
    if not path.exists():pytest.skip('Dataset has not been prepared')
    return list(csv.DictReader(path.open(encoding='utf-8',newline='')))

def test_split_has_no_shared_sources_or_hashes(evidence):
    for key in ('source','group','sha256'):
        sets={s:{r[key] for r in evidence if r['split']==s} for s in ('train','val','test')}
        assert not sets['train'] & sets['test']
        assert not sets['train'] & sets['val']
        assert not sets['val'] & sets['test']
    classes={r['label'] for r in evidence}
    assert len(classes)==8
    for s in ('train','val','test'):
        assert {r['label'] for r in evidence if r['split']==s}==classes

def test_saved_metrics_recompute_from_predictions():
    r=json.loads((ROOT/'results/metrics.json').read_text())
    with np.load(ROOT/'results/test_predictions.npz',allow_pickle=False) as a:
        for mode,info in r['models'].items():
            p=a['p_'+mode]
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
            assert abs(np.mean(p.argmax(1)==a['y'])-info['test']['accuracy'])<1e-12
            assert np.sum(info['test']['confusion'])==len(a['y'])
        assert len(np.unique(a['ids']))==len(a['ids'])

def test_numeric_export_matches_training_predictions(evidence):
    model=Recognizer()
    with np.load(ROOT/'data/processed/features.npz',allow_pickle=False) as a:
        pp=model.predict_features(a['features'][a['split']=='test'])
    with np.load(ROOT/'results/test_predictions.npz',allow_pickle=False) as a:
        assert np.allclose(pp,a['p_phase_delta'],atol=1e-7)

def test_live_decode_and_feature_parity(evidence):
    row=next(r for r in evidence if r['split']=='test')
    result,frames=Recognizer().predict(ROOT/row['path'])
    with np.load(ROOT/'results/test_predictions.npz',allow_pickle=False) as a:
        i=list(a['ids']).index(row['id']);expected=a['p_phase_delta'][i]
    assert len(frames)==9
    assert np.allclose(list(result['probabilities'].values()),expected,atol=1e-6)

def test_invalid_video_rejected(tmp_path):
    bad=tmp_path/'bad.avi';bad.write_bytes(b'not a video')
    with pytest.raises(ValueError):decode_video(bad)

def test_local_api_errors_and_templates():
    client=create_app().test_client()
    assert client.get('/').status_code==200
    assert client.get('/api/status').json['ready']
    assert client.post('/api/predict').status_code==400
    bad=client.post('/api/predict',data={'video':(io.BytesIO(b'wrong'),'bad.exe')})
    assert bad.status_code==400
    invalid=client.post('/api/predict',data={'video':(io.BytesIO(b'wrong'),'bad.mp4')})
    assert invalid.status_code==400
    assert client.post('/api/example/../../app.py').status_code==404
    assert client.post('/api/example/not-real').status_code==404

def test_example_api_runs_real_inference():
    client=create_app().test_client()
    examples=client.get('/api/examples').json
    assert examples
    for e in examples:
        assert e['split']=='test' and 'path' not in e
    response=client.post('/api/example/'+examples[0]['id'])
    assert response.status_code==200
    data=response.json
    assert len(data['previews'])==9
    assert abs(sum(data['probabilities'].values())-1)<1e-8
    assert data['accepted']==(data['confidence']>=data['threshold'])
