# Sources checked on 19 October 2026

## Data and assets actually used

- **Primary image study:** [aneesh10/Cricket Shots Dataset, version 2](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset). The archive contains 4,724 images across drive, legglance-flick, pullshot and sweep. Downloaded metadata reports the license as Unknown. One excess exact duplicate collapses. Visual review revealed transformed copies, so the initial hash-only grouping was strengthened with quarter-turn/mirror hashes and ORB/RANSAC-verified neighbor pairs. The final 4,723 images occupy 363 connected visual groups. These groups are not verified match or player identities. Preliminary scores and the final audit are preserved separately.

- **Training/evaluation videos:** [amittalmale/CRICKET_SHOTS, version 1](https://www.kaggle.com/datasets/amittalmale/cricket-shots). The public Kaggle metadata declares CC0. The downloaded archive contains 562 AVI clips in eight shot folders. The uploader supplies no detailed dataset description. We do not claim to have annotated these clips or verified rights in the underlying broadcast recordings. Download metadata and checksums are in `data/source_metadata.json` and `data/provenance.json`.
- **Frozen ImageNet encoder:** [ONNX Model Zoo MobileNetV2](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet), `mobilenetv2-12.onnx`, upstream Apache 2.0. We expose its global average pooling output and train our own classifier heads. No pretrained cricket classifier is used.
- **Report template:** [official CVPR 2022 author guidelines](https://cvpr2022.thecvf.com/author-guidelines), [2022 author kit v1.1-1](https://cvpr2022.thecvf.com/sites/default/files/2021-10/cvpr2022-author_kit-v1_1-1.zip). The professor's [Overleaf link](https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn) currently displays a CVPR 2026 template. This project uses the archived 2022 `cvpr.sty` instead.
- All cricket images in the report, slides and demo are decoded from the downloaded dataset. None are synthetic. Example selection is documented and does not alter quantitative evaluation.

## Primary literature used for positioning

- Mark Sandler et al., [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), CVPR 2018. The visual backbone, not a novel component of this project.
- Limin Wang et al., [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), ECCV 2016. Sparse temporal sampling and video-level supervision predate this project. Our descriptor uses fixed phase means and differences with a frozen encoder rather than reproducing TSN training.
- Chuan Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017. Source for scalar temperature scaling. We fit the scalar by validation negative log likelihood.
- Yonatan Geifman and Ran El-Yaniv, [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500), NeurIPS 2017. Background for trading prediction coverage against errors. Our finite validation threshold is a heuristic and does not implement their risk guarantee.
- Sungwoo Kang, [Modern Deep Learning Approaches for Cricket Shot Classification: A Comprehensive Baseline Study](https://arxiv.org/html/2510.09187v1), arXiv preprint, 2025. Compares video sequence models; its proposed method uses EfficientNet-B0 with a bidirectional GRU and temporal attention. Our main method uses one image and linear spatial-view heads. We do not compare its published percentages to our four-class image split or eight-class video pilot. Its [linked baseline repository](https://github.com/hpicsk/CricShot10_Baselines) is a literature reference, not code imported into this project.
- A. Sen et al., [CricShotClassify: An Approach to Classifying Batting Shots from Cricket Videos Using a Convolutional Neural Network and Gated Recurrent Unit](https://www.mdpi.com/1424-8220/21/8/2846), Sensors 21(8), 2846, 2021, DOI 10.3390/s21082846. Uses ten-class videos, CNN frame features and a GRU, including frozen and partly fine-tuned VGG16 variants. Our main study uses single images with frozen MobileNetV2 and linear spatial-view heads. The authors' [CricShot10 repository](https://github.com/ascuet/CricShot10) is a literature/dataset reference; neither its data nor classifier code was imported.
- Sk. Wasiqul Hoque et al., *KUCricShot: A Dataset for Developing Detection Model of Cricket Shots*, ICCIT 2023. Citation and four-shot dataset description checked against the [authors' repository](https://github.com/AtanuCSE/CricketShotDetection). This dataset was not downloaded or used for experiments.

## Implementation references

- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [source repository](https://github.com/scikit-learn/scikit-learn).
- [ONNX Runtime](https://onnxruntime.ai/docs/) and [repository](https://github.com/microsoft/onnxruntime).
- [Flask documentation](https://flask.palletsprojects.com/) and [repository](https://github.com/pallets/flask).
- [OpenCV](https://docs.opencv.org/) and [repository](https://github.com/opencv/opencv).
- [Tectonic](https://github.com/tectonic-typesetting/tectonic), local LaTeX compiler downloaded into `tools/`.

## Attribution boundary

The implementation uses the libraries above. HOG, the pretrained backbone, logistic regression, calibration and confidence rejection are established methods. The main image study implements a global/crop absolute-difference representation, compares it with single views and concatenation, and exposes agreement in an inspectable offline demo. Its small observed gain does not establish reliable superiority or priority in the literature. The phase/difference/spread descriptor belongs to the separate exploratory video pilot.

## Course revision, checked 19 October 2026

- Supplied lecture PDFs are preserved in the full local research package under `course_notes/`. They are excluded from the public repository and smaller demo package; verified topic/page references are in [course_mapping.md](course_mapping.md).
- HOG primary method: [Dalal and Triggs, CVPR 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). Our documented NumPy variant is not an exact reproduction of the original detector.
- [Official scikit-learn logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), used to check the classifier behavior; no external cricket classifier code is imported.
- [Supplied Overleaf URL](https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn) currently displays the 2026 submission kit. Existing unmodified official 2022 style is preserved in this revision.

