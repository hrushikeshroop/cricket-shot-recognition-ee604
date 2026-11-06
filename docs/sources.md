# Sources

- Dataset and author archive/spreadsheet links: https://github.com/Dihan69/CricShot10k . Pinned downloads and hashes: `data/provenance.json`.
- Common coaching shot families: New Zealand Cricket, https://www.nzc.nz/media/15726/sample-coaching-plan-skill-sets.pdf and https://play.nzc.nz/sites/default/files/2022-09/Development%20Coaching%20Course%20E-Manual.pdf . These do not provide a worldwide frequency ranking.
- Cricket CNN/GRU background: Sen et al. (2021), https://www.mdpi.com/1424-8220/21/8/2846 . Different data/protocols prevent direct accuracy ranking.
- MC3 video CNN: Tran et al., https://arxiv.org/abs/1711.11248 ; official weights/preprocessing: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html . Exact graph/weight provenance: `models/encoder.json`.
- MobileNetV2: Sandler et al., https://arxiv.org/abs/1801.04381 ; supplied ONNX model source: https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet . Exact graph/pool provenance: `models/focus_encoder.json`.
- Temporal segment representations: Wang et al., https://arxiv.org/abs/1608.00859 . Our ordered pooling is not a reproduction of their trained TSN system.
- Optical flow and tracking: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html . Parameters/fallback are in `features.py`.
- Temperature scaling: Guo et al., https://proceedings.mlr.press/v70/guo17a.html . Calibration alone does not establish generalisation to arbitrary unknowns.
- Selective classification: Geifman and El-Yaniv, https://arxiv.org/abs/1705.08500 . Our empirical threshold sweep does not implement a formal finite-sample error guarantee.
- Auxiliary outlier exposure: Hendrycks et al., https://arxiv.org/abs/1812.04606 . Our binary scope head uses a different loss and setting.
- Report template required in the course email: https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn . The report uses the archived CVPR 2022 style.
- Course foundation: supplied private EE604 Lectures 14 part II and 15, CNN/softmax material. Lecture PDFs are not redistributed. PCA, calibration and optical-flow methods are additional project study.
- Runtime/development libraries: https://numpy.org/ , https://opencv.org/ , https://onnxruntime.ai/ , https://pillow.readthedocs.io/ , https://flask.palletsprojects.com/ , https://pytorch.org/ , https://scikit-learn.org/ , https://scipy.org/ , https://www.pytest.org/ , https://github.com/wkentaro/gdown . Exact tested versions are pinned in the requirements files.

Demo clips/frames are real dataset material with native labels, source IDs and SHA256 values in `demo/examples.json`. They were selected after evaluation to illustrate successes, review and a failure, and do not replace the complete test results.
