"""Numeric classifiers, calibrated class support and explicit candidate-set review."""
import numpy as np


CLASSES=['drive','defence','flick','pull','square_cut','sweep']
DISPLAY={c:c.replace('_',' ').title() for c in CLASSES}


def head_logits(feature,head):
    x=(np.asarray(feature,dtype=np.float32)-head['mean'])/head['scale']
    if 'input_clip' in head:x=np.clip(x,-float(head['input_clip']),float(head['input_clip']))
    if 'projection' in head:x=(x-head['projection_center'])@head['projection'].T
    if 'weight1' in head:x=np.maximum(0,x@head['weight1'].T+head['bias1'])
    return x@head['weight'].T+head['bias']


def softmax(logits):
    z=np.asarray(logits,dtype=np.float64)
    z=z-z.max(axis=-1,keepdims=True);e=np.exp(z)
    return e/e.sum(axis=-1,keepdims=True)


def sigmoid(logits):
    z=np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0,-z))


def calibrated_support(conditional,scope_logits,calibration):
    """Joint support = calibrated scope score * conditional six-family score.

    This is a learned score under the calibration mixture, not a guarantee that
    an unusual video is recognised as unknown or that a human will be correct.
    """
    p=np.asarray(conditional,dtype=np.float64)
    t=float(calibration['temperature'])
    if t<=0 or not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid class scores')
    cp=softmax(np.log(np.maximum(p,1e-12))/t)
    scope=sigmoid(float(calibration['scope_slope'])*np.asarray(scope_logits)+float(calibration['scope_intercept']))
    return cp*np.expand_dims(scope,-1),cp,scope


def decide(support,threshold,classes=CLASSES):
    scores=np.asarray(support,dtype=np.float64)
    if scores.shape!=(len(classes),) or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):raise ValueError('Invalid support scores')
    if not 0<threshold<1:raise ValueError('Invalid support threshold')
    selected=np.flatnonzero(scores>threshold).tolist()
    reason='exactly_one_candidate' if len(selected)==1 else 'no_candidate' if not selected else 'multiple_candidates'
    return {'decision':'automatic' if len(selected)==1 else 'human_review',
            'class':classes[selected[0]] if len(selected)==1 else None,
            'candidate_classes':[classes[i] for i in selected],
            'reason':reason,'top_candidate':classes[int(scores.argmax())],
            'support_scores':{c:float(s) for c,s in zip(classes,scores)},'threshold':float(threshold)}
