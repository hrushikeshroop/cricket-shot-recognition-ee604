# Revised presentation notes

Slides 1-15 form the main talk; slide 16 is backup. Adjust timing to the assigned slot.

## 1. Cricket Shot Recognition

Introduce one-image classification. HOG and frozen features connect directly to lectures 10 and 15. The two-view difference is a small extension to test, not a new backbone. Cover image: selected held-out source image, https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2. Assistance and actual contributions must be attributed accurately.

## 2. Problem, labels and constraints

The user supplies the image. Drive and leg glance/flick are combined upstream classes. A full match is outside scope. Classification is multiclass with one target, not multilabel sigmoid prediction. Source: dataset v2; lectures 15 pages 16-19.

## 3. Course foundation and prior work

Lecture 8 part 2 page 9 distinguishes local/global features; our crop is a regional descriptor, not a keypoint. Lecture 14 part 2 pp 2-3 and 27 cover extractor/classifier and softmax. MobileNetV2 is our architecture choice, not claimed taught by name. Sources: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf ; https://arxiv.org/abs/1801.04381 ; https://github.com/AtanuCSE/CricketShotDetection ; https://arxiv.org/abs/2510.09187 . Do not compare published percentages across datasets.

## 4. Images and fixed visual-group split

Data: https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2. Unknown license in downloaded metadata; no original annotations claimed. Fixed seed 604 folds; this is one holdout, not five-fold cross-validation. Test images are clustered copies and must not be described as 945 independent photographs.

## 5. Evaluation choices and their limits

One validation set was reused for regularization, calibration and threshold selection. This can overfit validation. We disclose it rather than create a new split and pretend the original test was untouched. Initial hash-only contrast result was 96.3%; stronger transformed-copy grouping yields 65.5%. Both results and original individual predictions remain available.

## 6. Implemented pipeline

Explain the two branches. HOG and CNN global inputs have identical aspect-preserving 224 square geometry. CNN has two view passes; its global, crop and fused heads share features. HOG uses luminance, unsigned gradients, cells, blocks and per-image normalization. No StandardScaler or PCA learns from test. Source: actual implemented pipeline; lecture mapping in docs/course_mapping.md.

## 7. HOG: gradients, cells and blocks

Trace gx and gy central differences, sqrt(gx²+gy²), atan2 modulo 180. A 50-degree gradient of magnitude 90 votes 45 to the 40-degree bin and 45 to 60. No spatial vote interpolation. Each 36-value block divides by sqrt(sum(h²)+1e-10). 13*13*36=6084. This is a documented HOG variant, not the exact signed-angle lecture example or original pedestrian detector. Lecture 10 pp 7-23; original method https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf .

## 8. CNN views and the contrast feature

Both vectors are normalized before joining; the joined descriptor is normalized again. Contrast W has (4,3840) shape plus four biases. It learns weighted multiclass cross-entropy with L2 regularization. No bat-motion interpretation is assigned to feature coordinates. Actual inputs are displayed in the demo. MobileNetV2: https://arxiv.org/abs/1801.04381 ; ONNX weights: https://github.com/onnx/models .

## 9. Recognition results

HOG fixed settings, validation-selected C and predictions are saved under results/hog. Do not try a new descriptor after seeing the 35.3% test score to improve a headline. CNN heads/predictions are retained unchanged. This comparison does not isolate color versus pretraining versus architecture; no general statement that all HOG methods are inferior. Evidence: results/revision/comparison.csv and saved individual probabilities.

## 10. Uncertainty comes from 70 visual groups

Draw groups with replacement and keep all images within each drawn group. Reuse draws between methods for paired differences. Statistics are image-weighted in each resample. Intervals omit training/split variability and unknown match correlation. They are not a hypothesis-test certificate of equality. Contrast 619/945, concatenation 616/945. Data and calculations: scripts/analyze_revision.py.

## 11. Class errors

Read an off-diagonal entry and explain precision versus recall. Glance/flick is difficult under this protocol. HOG predicts pull for 110 of 224 sweep images. Fixed geometry, similar stances and background are plausible issues, not proven causes for each error. Evidence: saved prediction arithmetic in comparison.json.

## 12. Scores, calibration and secondary review

NLL and ECE worsen; do not say calibration improved. ECE uses ten equal-width confidence bins. The same validation set selected C, temperature and threshold. The gate needs global/crop/contrast agreement; the heads are correlated. HOG scores are unscaled and do not affect review. Sources: https://proceedings.mlr.press/v70/guo17a.html ; https://arxiv.org/abs/1705.08500 .

## 13. A confident error stays in the demo

Use image_04199, the preserved confident failure. Do not invent its exact cause or silently relabel it. The selected picture is an illustration, not an estimate of failure frequency. Real data source: aneesh10/cricket-shot-dataset v2. Follow with the view-disagreement image_00014.

## 14. Local image demonstration

Run setup.bat on a fresh machine, then run_demo.bat. Default URL http://127.0.0.1:6040. Revision verification may use 6060 to avoid another local service. Show success image_00210, confident error image_04199 and disagreement image_00014. Selected examples are honestly labeled. JSON preserves temperature-scaled score semantics. Screenshot must come from the revised running app, not a mockup.

## 15. What we can defend

Novelty remains a weakness, not a promise of marks. Do not claim a new architecture, detector or state-of-the-art result. Mention unknown upstream license, possible missed copies and unverified labels. Course mapping and viva guide contain actual page references and a tensor-shape trace. Team identities and the real repository URL are on the cover; assistance is attributed.

## 16. Backup: video pilot and duplicate audit

Use only for questions. Equal 30.2% majority accuracy means weak video recognition, despite higher macro F1. Fixed crop followed pilot test feedback. Source: https://www.kaggle.com/datasets/amittalmale/cricket-shots . Image audit: pHash <=6 bits with thumbnail MAD, rotations/mirrors, then 15 feature neighbors within cosine .20. ORB binary matching, .72 ratio, >=16 matches, >=12 RANSAC inliers, >=65% inlier fraction, >=8% hull coverage. These are evaluation tools, not a novel shot classifier. Other sources: https://github.com/onnx/models ; https://github.com/scikit-learn/scikit-learn ; https://github.com/microsoft/onnxruntime ; https://github.com/opencv/opencv ; https://arxiv.org/abs/1608.00859 . Full URLs, saved evidence and complete code appendix accompany the report.

