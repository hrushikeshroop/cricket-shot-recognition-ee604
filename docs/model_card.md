# Model card: revised image project

Task: one-of-four image classification into drive, leg glance/flick, pull and sweep. The upstream labels are not independently re-annotated.

The main recognizer retains the original numeric CNN heads and temperatures. A frozen ImageNet MobileNetV2 supplies two 1,280-value embeddings. A fixed central crop is a regional view, not a detector. Normalized `[u;v;abs(u-v)]` feeds a 3,840-input linear softmax head. Global/crop diagnostic heads share the same two encoder passes. A class-balanced L2-regularized head learns from cricket training data; the CNN stays frozen.

The exploratory HOG baseline uses the same global-view input geometry and split. Parameters and cache binding are in `models/hog/classifier.json` and `data/images/processed/hog_features.json`. No feature scaler or CNN layer is trained for HOG. Its separate C is validation-selected, and its uncalibrated scores do not control review.

Contrast: 65.5% accuracy and 64.9% macro F1, 619/945 correct. Concatenation: 616/945. HOG: 35.3% and 34.5%, 334/945. Conditional uncertainty uses 70 visual groups, not independent images. See `results/revision/comparison.json`.

Temperature scaling worsened test calibration (ECE 0.043 to 0.094); settings remain unchanged. The 0.70 score/agreement gate accepts 557 images with 433 correct, 77.7% accepted accuracy. A high score or agreement is no guarantee. Unknown inputs can receive a known class confidently.

The corrected image evaluation is exploratory after duplicate-audit feedback; HOG development has seen prior dataset results. Validation reuse, missing match/player identities, possible missed copies, fixed geometry and unverified labels limit use. This is not a coaching, safety or production model. The separate video pilot is weak exploratory backup evidence.
