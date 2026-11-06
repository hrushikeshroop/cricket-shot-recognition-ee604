"""Frozen video/frame features and camera-compensated temporal measurements.

Cropping is a fixed broadcast-layout prior, not a batter detector. Ordered bins
have no contact/phase annotations. The same code serves extraction and the demo.
"""
from pathlib import Path
import hashlib,inspect,json
import cv2,numpy as np,onnxruntime as ort
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2]
MEAN=np.array([.43216,.394666,.37645],dtype=np.float32)
STD=np.array([.22803,.22145,.216989],dtype=np.float32)
BOX=(.20,.15,.80,1.0)

def normalize(x):
    a = np.asarray(x,dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def decode(path):
    """Bounded sequential decoding; a single trimmed shot, at most five seconds."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot decode this video. Use an AVI, MP4 or MOV clip.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok,b = cap.read()
            if not ok:
                break
            if len(frames)>=600 or b.shape[0]*b.shape[1]>3840*2160:
                raise ValueError('Please use a trimmed shot of at most five seconds, up to 4K.')
            h,w = b.shape[:2]
            b = cv2.resize(b,(320,max(32,round(h*320/w))),interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames or not np.isfinite(fps) or fps<=0:
        raise ValueError('No decodable frames or valid frame rate.')
    if len(frames)/fps>5.05:
        raise ValueError('Please trim the video to one shot of at most five seconds.')
    if len({f.shape for f in frames})!=1:
        raise ValueError('Changing frame geometry is not supported.')
    return np.stack(frames), {'decoded_frames':len(frames),'fps':fps,'duration_seconds':len(frames)/fps}

def sample_indices(n,start=0,stop=None,count=24):
    stop = n if stop is None else stop
    if not 0<=start<stop<=n:
        raise ValueError('Invalid temporal window')
    return np.rint(np.linspace(start,stop-1,count)).astype(int)

def prepare(frames,indices):
    """Official MC3 resize(128,171), centre crop112, Kinetics normalisation.

    cv2 uses bilinear interpolation; the exported encoder sees precisely this
    preprocessing in both feature extraction and interactive inference.
    """
    a = np.stack([cv2.resize(frames[i],(171,128),interpolation=cv2.INTER_LINEAR)[8:120,30:142] for i in indices])
    a = (a.astype(np.float32)/255-MEAN)/STD
    return np.ascontiguousarray(a.transpose(3,0,1,2)[None])

def activity(frames):
    """Estimate residual motion after robust background affine alignment.

    Exclude the central batter area from camera fitting. Failed fits, hard cuts
    and tiny activity are reported and trigger a uniform-sampling fallback.
    """
    gray = [cv2.resize(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),(160,max(16,round(f.shape[0]/2))),interpolation=cv2.INTER_AREA) for f in frames]
    h,w = gray[0].shape
    region = (slice(round(.05*h),round(.85*h)),slice(round(.30*w),round(.70*w)))
    mask = np.full((h,w),255,np.uint8)
    mask[round(.02*h):round(.94*h),round(.24*w):round(.76*w)] = 0
    raw,comp,reliable,cuts = [],[],[],[]
    for prev,cur in zip(gray[:-1],gray[1:]):
        points = cv2.goodFeaturesToTrack(prev,160,.01,5,mask=mask)
        matrix,okfit = None,False
        if points is not None and len(points)>=8:
            nxt,status,_ = cv2.calcOpticalFlowPyrLK(prev,cur,points,None,winSize=(15,15),maxLevel=2)
            if nxt is not None:
                good = status.reshape(-1).astype(bool)
                a,b = points[good].reshape(-1,2),nxt[good].reshape(-1,2)
                if len(a)>=8:
                    # Seed makes OpenCV RANSAC repeatable across extractor/demo.
                    cv2.setRNGSeed(604)
                    matrix,inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=1000)
                    if matrix is not None and inliers is not None:
                        scale = float(np.hypot(matrix[0,0],matrix[0,1]))
                        okfit = float(inliers.mean())>=.5 and .9<=scale<=1.1
        flow = cv2.calcOpticalFlowFarneback(prev,cur,None,.5,2,11,2,5,1.1,0)
        mag = np.linalg.norm(flow[region],axis=-1)
        raw.append(float(np.quantile(mag,.8)))
        if okfit:
            aligned = cv2.warpAffine(prev,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            cut = float(np.mean(np.abs(aligned.astype(np.float32)-cur)))>45
            residual = cv2.calcOpticalFlowFarneback(aligned,cur,None,.5,2,11,2,5,1.1,0)
            comp.append(float(np.quantile(np.linalg.norm(residual[region],axis=-1),.8)) if not cut else 0.)
        else:
            cut = float(np.mean(np.abs(prev.astype(np.float32)-cur)))>45
            comp.append(0.)
        reliable.append(bool(okfit and not cut))
        cuts.append(bool(cut))
    return {'raw':raw,'compensated':comp,'reliable':reliable,'cuts':cuts}

def windows(frames):
    n = len(frames)
    motion = activity(frames)
    length = max(2,min(n,int(np.ceil(.75*n))))
    center = (n-length)//2
    uniform = np.arange(n)
    selected = {'uniform':uniform,'center':np.arange(center,center+length)}
    details = {}
    for key,scores in [('raw_motion',motion['raw']),('compensated_motion',motion['compensated'])]:
        a = np.asarray(scores,float)
        candidates = np.array([a[i:i+length-1].mean() for i in range(n-length+1)]) if len(a) else np.array([0.])
        start = int(candidates.argmax())
        reliable = key=='raw_motion' or (np.mean(motion['reliable'])>=.5 if len(a) else False)
        meaningful = bool(candidates.max()>=.05 and candidates.max()>1.05*(candidates.mean()+1e-8))
        fallback = not (reliable and meaningful)
        selected[key] = uniform.copy() if fallback else np.arange(start,start+length)
        details[key] = {'start':0 if fallback else start,'stop':n if fallback else start+length,
                        'fallback_uniform':fallback,'window_score':float(candidates.max()),
                        'reliable_fraction':float(np.mean(motion['reliable'])) if len(a) else 0.}
    return selected, {'activity':motion,'windows':details}

class VideoEncoder:
    def __init__(self,path=None,threads=4):
        model=Path(path or ROOT/'models/mc3_features.onnx')
        if not model.is_file():raise FileNotFoundError('Missing video encoder.')
        options = ort.SessionOptions()
        options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name

    def encode(self,frames):
        """Return (24,512) features with spatial pooling, before temporal pooling."""
        return self.session.run(None,{self.input:prepare(frames,np.arange(len(frames)))})[0][0].astype(np.float32)

    def views(self,frames):
        frames = frames[sample_indices(len(frames))]
        selected,details = windows(frames)
        sequence = self.encode(frames)
        features = {key:sequence[indices].mean(axis=0) for key,indices in selected.items()}
        features['sequence'] = sequence
        return features,details,selected

def decode_focused(path):
    """Crop original decoded pixels before reducing resolution for the frame CNN."""
    cap=cv2.VideoCapture(str(path));frames=[];fps=float(cap.get(cv2.CAP_PROP_FPS))
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if len(frames)>=600 or bgr.shape[0]*bgr.shape[1]>3840*2160:raise ValueError('Use a short clip up to 4K.')
            h,w=bgr.shape[:2];x1,y1,x2,y2=BOX
            rgb=cv2.cvtColor(bgr[round(y1*h):round(y2*h),round(x1*w):round(x2*w)],cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2];scale=min(1.,320/max(h,w))
            rgb=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:cap.release()
    if not frames or not np.isfinite(fps) or fps<=0 or len(frames)/fps>5.05:raise ValueError('Use one shot, at most five seconds.')
    return np.stack(frames)

def prepare_focused(frames):
    images=[np.asarray(ImageOps.fit(Image.fromarray(f),(224,224),method=Image.Resampling.BILINEAR),dtype=np.float32)/255 for f in frames]
    x=(np.stack(images)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(x.transpose(0,3,1,2))

class FrameEncoder:
    """Frozen ImageNet MobileNetV2, used on every selected focused frame."""
    def __init__(self,path=None):
        root=Path(__file__).resolve().parents[2]
        model=Path(path or root/'models/mobilenet_features.onnx')
        if not model.is_file():raise FileNotFoundError('Missing focused frame encoder.')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
    def encode(self,frames):
        x=prepare_focused(frames)
        return np.concatenate([self.session.run(None,{self.input:x[i:i+8]})[0].reshape(-1,1280) for i in range(0,len(x),8)]).astype(np.float32)

def describe(video,focused,indices,representation):
    """Pool fixed features; all fitted projection/scaling is in the learned head."""
    g=normalize(video.mean(axis=0))
    if representation=='video_mean':return g
    context=[g,normalize(focused.mean(axis=0))]
    if representation=='context_mean':return normalize(np.concatenate(context))
    if representation not in {'full_ordered','motion_ordered'}:raise ValueError('Unknown representation')
    chosen=indices['uniform' if representation=='full_ordered' else 'compensated_motion']
    bins=[normalize(focused[b].mean(axis=0)) for b in np.array_split(chosen,3)]
    return normalize(np.concatenate([*context,*bins]))


def feature_binding(root=ROOT):
    root=Path(root)
    parts=[(root/'data/manifest.csv').read_bytes(),Path(__file__).read_bytes()]
    for name in ['mc3_features.onnx','mobilenet_features.onnx']:
        with (root/'models'/name).open('rb') as f:parts.append(hashlib.file_digest(f,'sha256').digest())
    return hashlib.sha256(b''.join(parts)).hexdigest()
