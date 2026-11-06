"""Interval provenance and full/focused preprocessing equivalence on lossless video."""
import cv2,numpy as np,pytest
from cricketshot.recognizer import decode_interval
from cricketshot.features import decode,decode_focused


def make_clip(tmp_path,n=120,fps=30):
    path=tmp_path/'pattern.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),fps,(160,96))
    assert writer.isOpened()
    for i in range(n):writer.write(np.full((96,160,3),[i*3%256,i*5%256,i*7%256],np.uint8))
    writer.release();return path


def test_interval_matches_original_training_preprocessing(tmp_path):
    path=make_clip(tmp_path);full,_=decode(path);focused=decode_focused(path)
    selected,detail,info,first=decode_interval(path,.5,1.5)
    assert first==15 and info['selected_frames']==30 and info['start_seconds']==.5 and info['end_seconds']==1.5
    np.testing.assert_array_equal(selected,full[15:45]);np.testing.assert_array_equal(detail,focused[15:45])


def test_longer_upload_requires_one_selected_interval(tmp_path):
    path=make_clip(tmp_path,n=61,fps=10)
    with pytest.raises(ValueError,match='Choose a start and end'):decode_interval(path)
    full,focused,info,first=decode_interval(path,1,2)
    assert len(full)==len(focused)==10 and first==10 and info['duration_seconds']==1


@pytest.mark.parametrize('start,end',[(-1,1),(1,0),(0,6),(float('nan'),1),(0,float('inf'))])
def test_invalid_interval_is_rejected_before_decoding(start,end):
    with pytest.raises(ValueError):decode_interval('not_a_file.avi',start,end)
