"""Six-family recognition with a calibrated support threshold and review outcome."""
from pathlib import Path
import hashlib,json,time
import cv2,numpy as np
from .features import ROOT,BOX,VideoEncoder,FrameEncoder,sample_indices,windows,describe
from .policy import CLASSES,DISPLAY,head_logits,softmax,calibrated_support,decide


def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def decode_interval(path,start=0.,end=None):
    """Decode once, reproducing full/focused training pixels for a selected interval.

    A longer upload is allowed, but only one interval up to five seconds is
    classified. Sequential decoding preserves the frame-index provenance.
    """
    start=float(start);end=None if end is None else float(end)
    if not np.isfinite(start) or start<0 or start>120:raise ValueError('Start must be between 0 and 120 seconds.')
    if end is not None and (not np.isfinite(end) or end<=start or end-start>5.05 or end>120):raise ValueError('Select one batting interval of at most five seconds.')
    cap=cv2.VideoCapture(str(path));full=[];focused=[]
    if not cap.isOpened():cap.release();raise ValueError('Cannot decode this video. Try another video encoding.')
    fps=float(cap.get(cv2.CAP_PROP_FPS));source_count=int(max(0,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not np.isfinite(fps) or fps<=0 or fps>240:cap.release();raise ValueError('The video has an invalid or unsupported frame rate.')
    source_duration=source_count/fps if source_count else None
    if source_duration is not None and source_duration>120.05:cap.release();raise ValueError('Upload up to two minutes, then select a batting interval.')
    if end is None and source_duration is not None and source_duration-start>5.05:cap.release();raise ValueError('Choose a start and end around one shot, at most five seconds.')
    first=int(np.floor(start*fps+1e-8));last=None if end is None else int(np.ceil(end*fps-1e-8));shape=None;index=0
    try:
        while index<28800:
            if last is not None and index>=last:break
            ok,bgr=cap.read()
            if not ok:break
            if index<first:index+=1;continue
            if len(full)>=600 or bgr.shape[0]*bgr.shape[1]>3840*2160:raise ValueError('Use at most 600 selected frames and resolution up to 4K.')
            if shape is None:shape=bgr.shape
            if bgr.shape!=shape:raise ValueError('Changing frame geometry is not supported.')
            h,w=bgr.shape[:2];small=cv2.resize(bgr,(320,max(32,round(h*320/w))),interpolation=cv2.INTER_AREA)
            full.append(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
            x1,y1,x2,y2=BOX;rgb=cv2.cvtColor(bgr[round(y1*h):round(y2*h),round(x1*w):round(x2*w)],cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2];scale=min(1.,320/max(h,w));focused.append(cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA));index+=1
    finally:cap.release()
    if not full:raise ValueError('No decodable frames in the selected interval.')
    if len(full)/fps>5.05:raise ValueError('Select at most five seconds around one batting shot.')
    return np.stack(full),np.stack(focused),{'fps':fps,'selected_frames':len(full),'duration_seconds':len(full)/fps,'start_seconds':first/fps,'end_seconds':(first+len(full))/fps,'source_reported_frames':source_count,'source_reported_duration_seconds':source_duration},first


class Recognizer:
    def __init__(self,root=ROOT):
        cv2.setNumThreads(1)
        self.root=Path(root);models=self.root/'models';self.meta=json.loads((models/'classifier.json').read_text())
        if self.meta['schema_version']!=4 or self.meta['classes']!=CLASSES:raise ValueError('Classifier taxonomy is inconsistent.')
        for path,key in [('heads.npz','heads_sha256'),('scope_head.npz','scope_sha256')]:
            if file_sha(models/path)!=self.meta[key]:raise ValueError('Classifier weight hash mismatch.')
        with np.load(models/'heads.npz',allow_pickle=False) as z:self.heads={m['name']:{k.split('__',1)[1]:z[k].copy() for k in z.files if k.startswith(m['name']+'__')} for m in self.meta['members']}
        with np.load(models/'scope_head.npz',allow_pickle=False) as z:self.scope={k:z[k].copy() for k in z.files}
        for h in [*self.heads.values(),self.scope]:
            if np.any(h['scale']<=0) or not all(np.isfinite(a).all() for a in h.values()):raise ValueError('Invalid numerical head parameters.')
        self.encoder=VideoEncoder(models/'mc3_features.onnx');self.frame_encoder=FrameEncoder(models/'mobilenet_features.onnx')

    def predict(self,path,start=0.,end=None):
        began=time.perf_counter();full,focused,info,first=decode_interval(path,start,end);ids=sample_indices(len(full));full=full[ids];focused=focused[ids]
        indices,motion=windows(full);video_features=self.encoder.encode(full);frame_features=self.frame_encoder.encode(focused);representations={}
        values=[];member_predictions=[]
        for m in self.meta['members']:
            rep=m['representation']
            if rep not in representations:representations[rep]=describe(video_features,frame_features,indices,rep)
            probabilities=softmax(head_logits(representations[rep],self.heads[m['name']]))
            values.append(probabilities);member_predictions.append({'name':m['name'],'representation':rep,'class':CLASSES[int(probabilities.argmax())]})
        conditional=np.mean(values,axis=0);scope_rep=self.meta.get('scope_representation','context_mean')
        if scope_rep not in representations:representations[scope_rep]=describe(video_features,frame_features,indices,scope_rep)
        scope_logit=float(head_logits(representations[scope_rep],self.scope).ravel()[0])
        support,scaled,scope_score=calibrated_support(conditional,scope_logit,self.meta['calibration']);result=decide(support,self.meta['threshold'])
        result.update(display_name=DISPLAY[result['class']] if result['class'] else 'Human review',conditional_scores={c:float(s) for c,s in zip(CLASSES,scaled)},scope_score=float(scope_score),model=self.meta['name'],deployment=self.meta['deployment'],input_sha256=file_sha(path),heads_sha256=self.meta['heads_sha256'],video=info,encoder_sample_indices=(first+ids).tolist(),member_predictions=member_predictions,motion=motion,focus_box_xyxy_normalized=list(BOX),score_note='Calibrated support under the development data mixture. Scores do not guarantee correctness or unknown-shot detection.',elapsed_seconds=time.perf_counter()-began)
        return result,{'full':full,'focused':focused}
