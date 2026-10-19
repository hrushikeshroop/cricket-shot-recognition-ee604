# Revision log - 19 October 2026

This is a separate runnable project under `revision/CricketShot`. The original workspace remains unchanged. Its source/results/document hashes were checked against `original_files_baseline.json`. The original large comparison ZIP was absent when revision began; the original project files and smaller portable ZIP were available. No claim is made to have read an absent archive.

## Changes and reasons

- **One main image task:** report, slides and home page use drive, leg glance/flick, pull and sweep. The eight-class video model is labeled exploratory backup, with original evidence retained.
- **Lecture grounding:** inspected supplied notes, rendered relevant diagram pages, and checked the suggested anchors. `course_mapping.md` gives 1-based PDF pages. HOG and pretrained-feature/classifier learning provide the foundation; MobileNetV2 and the audit/review extensions are identified honestly.
- **One controlled HOG experiment:** fixed a documented unsigned-gradient variant and reused B's own manifest, groups and split. Same classifier family and C grid; validation selects C=0.1. No scaler/PCA, fusion sweep, seed search or new test-based threshold choice.
- **Proportionate contribution:** explain `[u;v;abs(u-v)]` as a small course-level extension. The gain is three test images. Duplicate auditing is evaluation work. No first-in-literature or substantial-advantage claim remains.
- **Group uncertainty:** recomputed all five methods from individual predictions and used the same 2,000 whole-group draws for paired comparisons. The contrast/concatenation accuracy interval spans zero.
- **Confidence wording:** temperature scaling worsened test calibration, so the UI says model score. The original temperature and review rule remain unchanged. HOG has separate uncalibrated scores and does not alter review.
- **Readable code and understanding:** expanded compressed core, preprocessing, training and audit code, with shared extraction functions and purpose comments. Added a measured tensor/logit trace and viva exercises. No unexplained technique was renamed to disguise it.
- **Rebuilt documents:** four-page main paper plus one-page supplement, full current-code appendix, Overleaf bundle, and a 16-slide native deck with sources in notes. Detailed audit, stress and video evidence are supplementary/backup. All current custom source/configuration files are inventoried (53 files).
- **Demo:** retains the selected correct, confident-error and disagreement examples, adds a HOG comparison, and identifies score semantics in JSON. The video page explicitly shows its weak result.

## Numbers and what changed

| Quantity | Original evidence | Revision | Explanation |
|---|---|---|---|
| Contrast CNN accuracy / macro F1 | 65.5% / 64.9% | Unchanged | Retained model and predictions; 619/945 correct |
| Concatenation accuracy | 65.2% | Unchanged | 616/945 correct; only three fewer than contrast |
| HOG accuracy / macro F1 | Not measured | 35.3% / 34.5% | New exploratory comparison; 334/945 correct |
| HOG validation macro F1 | Not measured | 49.4% | C=0.1 selected without new test feedback |
| Paired accuracy difference interval | Not reported | -1.46 to +2.38 points | New paired group-bootstrap diagnostic; same test identities |
| Original preliminary contrast | 96.3% | Preserved historical evidence | Weaker duplicate grouping; not a headline result |
| Test calibration ECE | 0.043 unscaled / 0.094 scaled | Unchanged | Failure to improve is now stated plainly |
| Review | 557 accepted; 433 correct | Unchanged | 58.9% coverage; 77.7% accepted accuracy; no 90% test guarantee |
| Video accuracy / macro F1 | 30.2% / 21.3% | Unchanged | Separate exploratory data; accuracy equals majority baseline |
| Inference timing | 12 ms median, 23 ms p95 CNN-only | 23 ms median, 52 ms p95 CNN plus HOG | Different measured operation: two image decodes; excludes initialization, HTTP/preview overhead |
| Report | 6 reading pages + 57 code pages | 5 reading pages + 78 code pages | Revised narrative and readable expanded/current source; no metric change |
| Slides | 16 | 16 | Main talk slides 1-15; backup slide 16 |

## Verification scope

Current checks are recorded in `results/verification.json`, `results/revision/` and `docs/visual_review.md`. Earlier records have `prior_` names; they are not evidence of new tests. Current verification reran 19 behavioral/integration tests, recomputed metrics, compared numeric model inference with cached predictions, checked 64 fresh two-view CNN extractions and sampled HOG extractions, sent actual multipart HTTP uploads, and checked source inventory completeness. The 64 CNN comparisons have maximum absolute discrepancy zero.

Actual PowerPoint rendered all 16 editable slides and exported the PDF; geometry checks found no overflow. The Artifact Tool module/runtime was unavailable, so the native Office workflow was used. Tectonic compiled the report; final Overleaf source compiled to identical manuscript text, and the merge helper produced the full 83-page document. Rendering/inspection scope and fresh package validation are recorded separately.

## Remaining limits and submission work

The corrected grouping and new HOG study are exploratory. Visual groups do not identify every original photo or match. Labels were not all independently verified, the image license is unknown in downloaded metadata, and validation was reused for model selection/calibration/review. Novelty remains modest. None of this revision proves general performance or promises marks.

The five supplied team identities are now in `configs/submission.json`, the report author block, slide cover, README and demo footer. The public repository is https://github.com/hrushikeshroop/cricket-shot-recognition-ee604. Individual work statements are omitted as requested. Codex assistance remains attributed. Course submission has not been performed.

Fresh extracted-package inference also passed in an isolated environment installed only from requirements.txt; no development-site packages were available. That prepublication record is now preserved in results/prepublication_package_validation.json. The current results/package_validation.json records a fresh Git-index export with no training assets or example fixtures, tested in the same isolated inference environment.

## Team and public repository update

All five supplied names/roll numbers are included. The repository belongs to `hrushikeshroop`; no project files were uploaded to the empty repository initially created under `akshithg23`. Individual work statements are omitted at the user's request. The source appendix now has 82 pages, giving an 87-page complete report. Current validation passed 20 tests and 34 submission checks. Numeric models and experimental results were not changed by this publication update. Data-download instructions and an optional example-restoration script support the smaller public Git checkout.
