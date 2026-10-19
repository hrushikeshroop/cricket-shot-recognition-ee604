# Revised presentation notes

Slides 1-15 form the main talk; slide 16 is backup. Adjust timing to the assigned slot.

## 1. Cricket Shot Recognition

Introduce one-image classification. HOG and frozen features connect directly to lectures 10 and 15. The two-view difference is a small extension to test, not a new backbone. Cover image: selected held-out source image, https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2.

## 2. Problem, labels and constraints

The user supplies the image. Drive and leg glance/flick are combined upstream classes. A full match is outside scope. Classification is multiclass with one target, not multilabel sigmoid prediction. Source: dataset v2; lectures 15 pages 16-19.

## 3. Previous cricket-recognition methods

Sen et al.: CricShotClassify, Sensors 21(8), 2846 (2021), https://www.mdpi.com/1424-8220/21/8/2846 . CNN frame features and GRU, including frozen and partly fine-tuned VGG16 variants. Kang: https://arxiv.org/abs/2510.09187 , a preprint comparing sequence models and proposing EfficientNet-B0 with bidirectional GRU and temporal attention. No code or datasets from either paper were imported. KUCricShot authors' repository https://github.com/AtanuCSE/CricketShotDetection describes a separate four-shot, 1278-example dataset, also unused. Lecture 8 part 2 p9 distinguishes local/global features. Lecture 14 part 2 pp2-3 and 27 covers extractor/classifier and softmax. We implement an image comparison, not a new backbone or a priority claim.

## 4. Images and fixed visual-group split

Data: https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2. Unknown license in downloaded metadata; no original annotations claimed. Fixed seed 604 folds; this is one holdout, not five-fold cross-validation. Test images are clustered copies and must not be described as 945 independent photographs.

## 5. Evaluation choices and their limits

One validation set was reused for regularization, calibration and threshold selection. This can overfit validation. We disclose it rather than create a new split and pretend the original test was untouched. Initial hash-only contrast result was 96.3%; stronger transformed-copy grouping yields 65.5%. Both results and original individual predictions remain available.

## 6. Implemented pipeline

Explain the two branches. HOG and CNN global inputs have identical aspect-preserving 224 square geometry. The fixed crop reduces some background without training a detector, but can miss an off-center batter. Keeping the global view preserves a second source of context, although square center fitting also removes peripheral pixels. MobileNetV2 is an efficient ImageNet backbone selected for CPU use; freezing limits cricket fitting to small heads on only 221 training groups. This is design reasoning, not proof that freezing or this model is optimal. CNN has two view passes; its heads share features. Source: actual implemented pipeline, https://arxiv.org/abs/1801.04381 , and docs/course_mapping.md.

## 7. HOG: gradients, cells and blocks

Trace gx and gy central differences, sqrt(gx²+gy²), atan2 modulo 180. A 50-degree gradient of magnitude 90 votes 45 to the 40-degree bin and 45 to 60. No spatial vote interpolation. Each 36-value block divides by sqrt(sum(h²)+1e-10). 13*13*36=6084. This is a documented HOG variant, not the exact signed-angle lecture example or original pedestrian detector. Lecture 10 pp 7-23; original method https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf .

## 8. CNN views and the contrast feature

Both vectors are L2-normalized before joining so their raw magnitudes do not set the relative block scale; the joined descriptor is normalized again. Equal norm does not imply equal usefulness. No normalization ablation was conducted. A linear head on concatenation can compute signed u-v but cannot generally compute abs(u-v); the explicit block supplies this nonlinear relation. This is the motivation to test, not proof of improvement. Contrast W has (4,3840) shape plus four biases, 15364 stored parameters, trained with weighted multiclass cross-entropy and L2 regularization. No bat-motion interpretation is assigned to coordinates. MobileNetV2: https://arxiv.org/abs/1801.04381 ; weights: https://github.com/onnx/models .

## 9. Recognition results

HOG fixed settings, validation-selected C and predictions are saved under results/hog. Do not try a new descriptor after seeing the 35.3% test score to improve a headline. CNN heads/predictions are retained unchanged. This comparison does not isolate color versus pretraining versus architecture; no general statement that all HOG methods are inferior. Evidence: results/revision/comparison.csv and saved individual probabilities.

## 10. Where contrast changes decisions

This descriptive analysis uses unchanged saved test predictions. Cases are first IDs in gain/loss categories, selected after evaluation. image_00696: label drive; global drive, crop pull, concat pull, contrast drive. image_00017: label drive; global drive, crop glance/flick, concat drive, contrast glance/flick. Agreement subsets do not identify a causal mechanism or validate a new selection rule. For uncertainty, resample the 70 visual groups, retaining all their images, in 2000 paired draws with seed17; image-weighted statistics. The interval omits training/split variability and unknown match correlation. Complete individual-method intervals remain in report and results/revision/comparison.json; case evidence is in view_diagnostics.json.

## 11. Class errors

Read an off-diagonal entry and explain precision versus recall. Glance/flick is difficult under this protocol. HOG predicts pull for 110 of 224 sweep images. Fixed geometry, similar stances and background are plausible issues, not proven causes for each error. Evidence: saved prediction arithmetic in comparison.json.

## 12. Scores, calibration and secondary review

NLL and ECE worsen. ECE uses ten equal-width confidence bins. The same validation set selected C, temperature and threshold; the 90% validation target did not transfer to test. The gate needs global/crop/contrast agreement; the heads are correlated. It accepts557/945 with433 correct. Its measured benefit over confidence alone is tiny. Retain it as visible triage for conflicting views; diagnostic heads use the same two encoder outputs, adding no encoder pass. HOG scores are unscaled and do not affect review. Sources: https://proceedings.mlr.press/v70/guo17a.html ; https://arxiv.org/abs/1705.08500 .

## 13. A confident error stays in the demo

Use image_04199, the preserved confident failure. Do not invent its exact cause or silently relabel it. The selected picture is an illustration, not an estimate of failure frequency. Real data source: aneesh10/cricket-shot-dataset v2. Follow with the view-disagreement image_00014.

## 14. Local image demonstration

Run setup.bat on a fresh machine, then run_demo.bat. Install and rehearse before going offline. Default URL http://127.0.0.1:6040. Five shipped JPEG illustrations cover all four successful class labels plus confident error image_04199. The full local package also has original disagreement image_00014; public clones can restore it optionally. Fresh scores are computed from actual input files; JPEG inputs may differ from original benchmark pixels. All examples were selected after evaluation. Show actual views and HOG, then export JSON. Screenshot source is the running app.

## 15. What we can defend

The implemented contribution is the controlled comparison and inspectable integration of two-view absolute differences and diagnostic agreement. Components are established; HOG itself is not new. The measured gain is uncertain, so do not claim a new architecture, detector or state-of-the-art result. The audit also corrects an optimistic initial result, and calibration fails to transfer. Mention possible missed copies and unverified labels. Course mapping and viva guide contain page references and a measured trace. Team identities and repository URL are on the cover.

## 16. Backup: video pilot and duplicate audit

Use only for questions. Equal 30.2% majority accuracy means weak video recognition, despite higher macro F1. Fixed crop followed pilot test feedback. Source: https://www.kaggle.com/datasets/amittalmale/cricket-shots . Image audit: pHash <=6 bits with thumbnail MAD, rotations/mirrors, then 15 feature neighbors within cosine .20. ORB binary matching, .72 ratio, >=16 matches, >=12 RANSAC inliers, >=65% inlier fraction, >=8% hull coverage. These are evaluation tools, not a novel shot classifier. Other sources: https://github.com/onnx/models ; https://github.com/scikit-learn/scikit-learn ; https://github.com/microsoft/onnxruntime ; https://github.com/opencv/opencv ; https://arxiv.org/abs/1608.00859 . Full URLs, saved evidence and complete code appendix accompany the report.

