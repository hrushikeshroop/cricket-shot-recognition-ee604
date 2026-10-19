# Encoder attribution and modification notice

`mobilenetv2_features.onnx` is derived from the ONNX Model Zoo MobileNetV2-12 graph, pretrained on ImageNet. Source: https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet . The upstream license is reproduced in APACHE-2.0.txt.

Modification: the classification output is replaced by the 1,280-dimensional pooled feature output. Pretrained convolution weights stay frozen. See src/cricketshot/core.py and the encoder-export code in scripts/prepare_data.py. The original source graph and its SHA-256 are recorded in data/provenance.json; fetch_assets.py retrieves the pinned graph. This notice does not assign the upstream weights to the student team.
