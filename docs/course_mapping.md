# Course ideas used in CricketShot

Page numbers below are **1-based PDF page numbers**, including the title page.
The supplied files are preserved in `course_notes/` in the full local research package; they are excluded from the public repository and smaller demo package. The notes were inspected as
text and the relevant diagram pages were rendered before this revision.

| Supplied lecture | PDF pages | Connection to the implementation |
|---|---|---|
| Lecture 8 part 2, `EE604_L8p2_2627_...pdf` | 4-5, 9 | Features can describe pixels, patches or whole images. Our global CNN vector describes a resized image; the second vector describes a fixed central region. The crop is a regional view, not a detected keypoint or a segmented batter. |
| Lecture 10, `EE604_L10_2627_...pdf` | 7-11 | HOG obtains horizontal/vertical derivatives, magnitude and angle, then accumulates **magnitude-weighted** orientation votes. See `gradients` and `cell_histograms` in `src/cricketshot/hog.py`. |
| Lecture 10 | 12-19 | Cells collect spatially arranged histograms; neighboring cells form overlapping blocks. Our 224x224 image has 14x14 cells of 16x16 pixels and 13x13 blocks of 2x2 cells. |
| Lecture 10 | 20-23 | Normalize blocks to reduce sensitivity to gradient strength and concatenate them into one image descriptor. Our implementation uses L2 with epsilon, then per-image L2 normalization for the linear classifier. |
| Lecture 14 part 2, `EE604_L14p2_2627_...pdf` | 2-3 | Classification separates a numeric feature extractor from a classifier. The project compares hand-crafted HOG with a frozen CNN, each followed by a learned linear classifier. |
| Lecture 14 part 2 | 16, 27 | Weights/bias map features to logits; softmax converts four logits to relative class scores. See `ImageRecognizer.predict_features` and `softmax`. |
| Lecture 15, `EE604_L15_2627_...pdf` | 11-14 | Pretrained networks can be frozen feature extractors followed by an ML classifier. This is the CNN branch's foundation. MobileNetV2 is our selected architecture, not an architecture claimed to have been taught by name. |
| Lecture 15 | 16-19 | One-of-four classification is multiclass, with one true target, softmax outputs and categorical cross-entropy. Our logistic regression uses this loss with L2 regularization and balanced class weights. It is not multilabel sigmoid classification. |

The lecture's HOG example uses signed angles and illustrative key angles. Our
explicit variant uses nine **unsigned** bins centered at 0,20,...,160 degrees,
linearly interpolates orientation votes, and does not interpolate across cell
boundaries. These are recorded engineering choices, not an exact reproduction
of the lecture's example or of the original Dalal-Triggs detector. No HOG
parameter was searched against the test set.

## Extensions beyond the lecture examples

* **Two-view absolute difference:** concatenate normalized global and regional
  CNN features with their coordinate-wise absolute difference. It gives a
  linear head an additional nonlinear relationship. It is not an attention or
  motion map. The observed gain over concatenation is only three test images.
* **Visual duplicate audit:** hashes and geometric matching group related
  images so modified copies are less likely to cross splits. This is evaluation
  work, not a new recognition algorithm. See the viva guide for the tools.
* **Temperature scaling and review:** retain the original validation choices
  and measure their limitations. Scaling worsened test calibration; the gate
  did not achieve its validation target on test. These are secondary diagnostics.
* **Group bootstrap:** resample entire visual groups to describe uncertainty,
  rather than treat 945 test images as independent original photographs.

This mapping supports explanation and viva preparation. It does not imply that
every technique used in the project was taught, or that course alignment makes
the algorithm novel.
