"""Rebuild frozen feature graphs into a separate directory; preserve supplied models."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np,onnx,onnx.utils,onnxruntime as ort,requests,torch
from torchvision.models.video import mc3_18,MC3_18_Weights
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.build/torch_cache'))
FRAME_URL='https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx'
FRAME_SOURCE_SHA='c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='.build/exported_encoders');args=parser.parse_args();out=ROOT/args.output_dir
    if out.resolve()==(ROOT/'models').resolve():raise ValueError('Use a separate export directory; do not replace benchmark encoders.')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(604)
    backbone=mc3_18(weights=MC3_18_Weights.KINETICS400_V1).eval()
    class TemporalEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.features=torch.nn.Sequential(backbone.stem,backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4)
        def forward(self,x):return self.features(x).mean(dim=(-1,-2)).transpose(1,2)
    model=TemporalEncoder().eval();batch=torch.randn(1,3,24,112,112);video_path=out/'mc3_features.onnx'
    torch.onnx.export(model,batch,str(video_path),input_names=['video'],output_names=['features'],opset_version=17,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(video_path),sess_options=opts,providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=model(batch).numpy()
    actual=session.run(None,{'video':batch.numpy()})[0];error=float(np.max(np.abs(expected-actual)))
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
    cache=ROOT/'.build/encoder_sources';cache.mkdir(parents=True,exist_ok=True);frame_source=cache/'mobilenetv2-12.onnx'
    if not frame_source.exists():
        response=requests.get(FRAME_URL,timeout=120);response.raise_for_status();frame_source.write_bytes(response.content)
    if sha(frame_source)!=FRAME_SOURCE_SHA:raise ValueError('Frame-encoder source checksum changed.')
    graph=onnx.load(frame_source);pools=[n for n in graph.graph.node if n.op_type=='GlobalAveragePool']
    if len(pools)!=1:raise ValueError('Expected one global pooling node.')
    frame_path=out/'mobilenet_features.onnx';onnx.utils.extract_model(str(frame_source),str(frame_path),[graph.graph.input[0].name],[pools[0].output[0]])
    onnx.checker.check_model(onnx.load(frame_path))
    record={'video_source':'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth','video_sha256':sha(video_path),'video_max_abs_export_error':error,'frame_source':FRAME_URL,'frame_source_sha256':FRAME_SOURCE_SHA,'frame_sha256':sha(frame_path),'note':'These exports are separate from the supplied benchmark graphs. Reaudit/reextract before substituting a graph with different bytes.'}
    (out/'export_check.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
