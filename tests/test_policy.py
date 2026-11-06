"""Decision-boundary and numeric inference checks, independent of trained weights."""
import numpy as np
from cricketshot.policy import decide,calibrated_support,head_logits


def test_exactly_one_class_is_automatic():
    r=decide([.7,.1,.05,.05,.05,.05],.3)
    assert r['decision']=='automatic' and r['class']=='drive' and r['candidate_classes']==['drive']


def test_multiple_candidates_require_review():
    r=decide([.36,.35,.1,.06,.04,.04],.3)
    assert r['decision']=='human_review' and r['class'] is None and len(r['candidate_classes'])==2


def test_no_candidate_and_threshold_equality_require_review():
    r=decide([.3,.2,.2,.1,.1,.1],.3)
    assert r['decision']=='human_review' and r['candidate_classes']==[]


def test_scope_support_can_reject_a_confident_conditional_class():
    support,conditional,scope=calibrated_support(np.array([.99,.002,.002,.002,.002,.002]),-5,{'temperature':1,'scope_slope':1,'scope_intercept':0})
    assert conditional.argmax()==0 and conditional[0]>.98
    assert np.isclose(support.sum(),scope) and decide(support,.3)['class'] is None


def test_projection_and_relu_match_manual_arithmetic():
    h={'mean':np.array([1,2],np.float32),'scale':np.array([2,1],np.float32),'input_clip':np.array(5),'projection_center':np.array([0,0],np.float32),'projection':np.array([[1,2]],np.float32),'weight1':np.array([[2],[-1]],np.float32),'bias1':np.array([-1,1],np.float32),'weight':np.array([[1,3],[2,-1]],np.float32),'bias':np.array([.5,1],np.float32)}
    np.testing.assert_allclose(head_logits(np.array([3,3],np.float32),h),[5.5,11])
