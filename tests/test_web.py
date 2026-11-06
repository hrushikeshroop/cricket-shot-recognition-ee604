"""HTTP input handling and unresolved decisions remain separate from classifier accuracy."""
from pathlib import Path
import importlib.util,io,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('six_app',ROOT/'app.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from cricketshot.policy import CLASSES,decide
from cricketshot.recognizer import decode_interval


class FakeRecognizer:
    def predict(self,path,start=0.,end=None):
        assert Path(path).name=='clip.avi'
        return decide([.2,.18,.17,.15,.12,.1],.3),{'full':np.zeros((24,40,80,3),np.uint8),'focused':np.zeros((24,40,40,3),np.uint8)}


@pytest.fixture
def client():return module.create_app(FakeRecognizer()).test_client()


def test_home_has_six_families_and_a_human_review_panel(client):
    response=client.get('/');assert response.status_code==200
    assert b'SIX SHOT FAMILIES' in response.data and b'Record a human decision' in response.data
    status=client.get('/api/status').get_json();assert status['classes']==6 and status['version']=='4.0.0'


def test_missing_or_unsupported_upload_is_rejected(client):
    assert client.post('/api/predict').status_code==400
    assert client.post('/api/predict',data={'video':(io.BytesIO(b'x'),'clip.exe')},content_type='multipart/form-data').status_code==400


def test_uploaded_filename_does_not_control_path_or_force_a_label(client):
    r=client.post('/api/predict',data={'video':(io.BytesIO(b'test bytes'),'../../outside.avi')},content_type='multipart/form-data')
    assert r.status_code==200
    data=r.get_json();assert data['class'] is None and data['decision']=='human_review' and set(data['support_scores'])==set(CLASSES)


def test_upload_size_limit_is_enforced():
    app=module.create_app(FakeRecognizer());app.config['MAX_CONTENT_LENGTH']=100
    r=app.test_client().post('/api/predict',data={'video':(io.BytesIO(b'x'*200),'clip.avi')},content_type='multipart/form-data')
    assert r.status_code==413 and '100 MB' in r.get_json()['error']


def test_invalid_interval_number_is_a_user_error(client):
    r=client.post('/api/predict',data={'video':(io.BytesIO(b'x'),'clip.avi'),'start':'bad'},content_type='multipart/form-data')
    assert r.status_code==400


def test_corrupt_video_is_a_user_error():
    class Decoder:
        def predict(self,path,start=0.,end=None):return decode_interval(path,start,end)
    r=module.create_app(Decoder()).test_client().post('/api/predict',data={'video':(io.BytesIO(b'not video'),'clip.avi')},content_type='multipart/form-data')
    assert r.status_code==400
