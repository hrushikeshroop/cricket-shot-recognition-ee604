# Understanding and defending the implementation

Use this with `course_mapping.md`, the source code and the measured trace in
`results/revision/worked_trace.json`. Reading answers alone is insufficient;
each member should run the small exercises at the end.

## What is the actual task?

One image goes in; one of drive, leg glance/flick, pull or sweep comes out. The
upstream labels combine some stroke types. This does not detect the batter,
locate contact, recognize arbitrary shots, measure technique or interpret a full
match. A central crop can cut off an off-center batter. Even the global view
undergoes aspect-preserving square center fitting at the final resizing step.

## Frozen parameters and trained parameters

The ONNX MobileNetV2 contains public ImageNet weights. They are used for forward
passes only. Its original 1,000-class output layer is removed from the feature
graph; pooled output has 1,280 values. We learn coefficients and biases of
multinomial logistic regression from **training cricket images**. For contrast,
the matrix has shape (4,3840), plus four biases: 15,364 stored parameters. The
global and crop diagnostic heads each have (4,1280) weights. The HOG head has
(4,6084) weights. HOG itself has no learned parameters.

## HOG, from pixels to a descriptor

`prepare_hog` shares the CNN global-view geometry and converts the resulting
224x224 RGB input to grayscale. `gradients` takes central horizontal/vertical
differences. Magnitude is sqrt(gx²+gy²); angle is atan2(gy,gx), modulo 180 degrees.
In `cell_histograms`, each pixel votes with its **magnitude** into two neighboring
orientation bins. It is not a count of pixels. With centers at 40 and 60 degrees,
a 50-degree gradient of magnitude 90 contributes 45 to each.

There are 14x14 cells of 16x16 pixels. A block contains 2x2 cells (36 values),
and moves one cell at a time, so 13x13 blocks overlap. Each block is divided by
sqrt(sum(h²)+epsilon²), epsilon=1e-5. Concatenate 169x36=6,084 values and normalize
the full vector to unit L2 length. This deterministic normalization has no fitted
training/validation/test statistics. There is no learned StandardScaler or PCA.
It is our documented HOG variant, not an exact reproduction of the original
human detector. Unsigned directions and fixed cells lose some direction/color
information; backgrounds still contribute gradients.

## CNN descriptors, normalization and the learned head

`load_image` checks size, applies EXIF orientation, and makes RGB pixels.
`image_views` bounds the image to 1024 pixels and creates a fixed central crop.
`prepare_frame(view='full')` fits either view to 224x224, scales RGB to [0,1],
applies ImageNet mean/std, transposes channels and adds a batch dimension.
Two separate calls to the frozen encoder produce (2,1280).

L2-normalize each vector to u and v. Global uses u; crop uses v; concatenation
uses normalize([u;v]); contrast uses normalize([u;v;abs(u-v)]). The absolute value
is nonlinear: a linear head could already compute u-v from concatenation, but
cannot generally compute abs(u-v). It is feature-wise difference between two
resized views, not a physical measure of bat movement or a local keypoint.

The linear head computes logits a=Wz+b and softmax(a/T). Training minimizes
weighted multiclass cross-entropy with L2 regularization. `C` is inverse
regularization strength; a larger value allows larger coefficients. Each head
uses C in {0.1,1,10,100}, selected by validation macro F1. Frozen features reduce
training work but may not represent cricket-specific differences well.

## Data split, duplicate leakage and the audit tools

Exact decoded RGB copies collapse before splitting. A perceptual hash compresses
coarse appearance into bits; Hamming distance counts differing bits. Small RGB
thumbnails check whether a similar hash also has similar colors. Rotations and
mirrors are checked because upstream files include transformed copies.

The advanced second stage proposes neighbors using frozen CNN similarity, then
checks ORB binary descriptors (corner detection plus a rotation-aware binary
patch description). Ratio filtering rejects ambiguous nearest descriptor
matches. A RANSAC homography fits a planar projective relation using subsets,
then counts consistent matches. Inlier fraction and spread constraints avoid
joining pictures because of a small logo. These tools can still miss related
frames or join similar photographs. A graph edge links a qualifying pair;
connected components form groups. See `strengthen_image_groups.py` for numeric
cutoffs. The matching does not learn cricket labels or supply match identities.

The original high 96.3% score preceded discovery of missed modified copies.
The revised split gives 65.5%; both results are retained. Grouping changed after
earlier evaluation, so this is exploratory. HOG uses the corrected manifest
unchanged: 2,834 train, 944 validation and 945 test images; 221/72/70 groups.
No group or decoded hash crosses sets. This is not an unseen-match experiment.

## Reading the results

A confusion matrix has true labels as rows and predicted labels as columns.
Recall is correct predictions divided by the row's support. Precision divides
correct predictions by all predictions of that class. F1 is their harmonic mean;
macro F1 averages the four class F1 values equally. Accuracy weights each image
equally. Contrast gets 619/945 correct; concatenation gets 616/945. That tiny
difference and one split do not establish superiority or strong algorithmic
novelty. HOG is an additional controlled comparison, not a source of a better
headline chosen after looking at test scores.

The uncertainty calculation resamples **70 visual groups**, keeping each sampled
group's images together. It uses the same draws for paired model differences.
The intervals condition on the fitted weights and split; they omit training
variability and cannot repair unidentified match correlations.

## Confidence, calibration and review

A model score is a relative softmax output among four known labels. It is not
the measured probability that an arbitrary upload is correct. Temperature is a
positive scalar fitted on validation NLL. It changes score sharpness but not
the top class. Here test ECE increased from about 0.043 to 0.094 and NLL worsened.
We keep the original setting and report the failure of calibration to transfer.

The same validation set chose C, temperatures and the review threshold. This
reuse can overfit it. The original rule requires max score at least 0.70 and
agreement of global, crop and contrast heads. The heads are correlated; agreement
does not prove correctness. On test, 557 images pass and 433 are correct:
58.9% coverage, 77.7% accepted accuracy. The 90% validation target is not a test
result or guarantee. HOG comparison scores are uncalibrated and do not alter the
review rule. The selected confident failure remains in the demo.

## What is novel, and what is inherited?

HOG, pretrained features, logistic regression, multi-view concatenation,
temperature scaling and geometric matching are established. The course-level
extension is the particular two-view absolute-difference study and its visible
agreement diagnostic. The measured gain is weak. The audit is useful evaluation
work, not a new recognition algorithm. Novelty remains a limitation; do not
claim first-in-literature status or disguise inherited tools. The separate video
pilot's 30.2% accuracy equals its majority baseline; its 21.3% macro F1 is better
than majority macro F1 but does not make it a dependable video recognizer.

## Small tasks members should actually perform

- Run setup, the tests and the image demo; upload your own cricket image and
  inspect the JSON. Explain the difference between the top label and review.
- Trace the provided success and confident failure through the functions above.
  Compare the stored tensor dimensions and logits in the worked trace.
- Run a synthetic HOG vote example and manually predict its two bin weights.
  Visualize a flat image and explain why its HOG vector is zero.
- Copy the training script to a scratch experiment, change C for **training and
  validation only**, and inspect coefficients and validation predictions. Do not
  replace the delivered model or choose settings from test feedback.
- Read a confusion matrix row and independently recompute one class's F1.
- Find a duplicate-group edge in the saved audit and verify both images share a
  split. Explain the difference between a visual group and a verified match ID.
- State a contribution only after personally doing or verifying it. Attribute
  public data, weights, papers, libraries and AI assistance accurately.
