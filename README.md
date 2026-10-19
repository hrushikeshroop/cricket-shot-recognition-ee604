# CricketShot - EE604 course project

Repository: [https://github.com/hrushikeshroop/cricket-shot-recognition-ee604](https://github.com/hrushikeshroop/cricket-shot-recognition-ee604)

| Name | Roll number |
|---|---|
| Hrushikesh Roop Avvari | 230249 |
| Voora Rakesh | 231174 |
| Gokavarapu Manikanta Sri Akshith | 230418 |
| Lokesh Kumar Moond | 230606 |
| Sharath | 230303 |

Classify **one image** as drive, leg glance/flick, pull or sweep. This separate revision adds a lecture-grounded HOG baseline, readable ML code, paired group uncertainty, and a CNN/HOG comparison in the local demo. The original workspace remains outside this folder; original model predictions and historical experiments are retained here.

The unchanged contrast CNN gets **619/945 correct: 65.5% accuracy, 64.9% macro F1**. Concatenation gets 616/945. Their paired accuracy-difference interval spans -1.46 to +2.38 points, so there is no clear advantage. HOG gets **334/945 correct: 35.3% accuracy, 34.5% macro F1**, using the identical split. These are exploratory results, not unseen-match performance.

## Start the demo

On a new Windows machine with Python 3.11 or newer, run **setup.bat**, then **run_demo.bat**, and open **http://127.0.0.1:6040**. This revision excludes the original machine's Python environment. From a terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

No API key, database or cloud service is required. Inference works offline after dependencies and included weights are present. Uploads use temporary local storage. Stop the server with Ctrl+C. Use `python app.py --port 6060` if the default port is occupied.

Upload an image to start immediately. Optional selected success, confident error and disagreement examples can be restored with the download command below. The image page shows actual model inputs, CNN class scores, the review reason, and a separate HOG baseline prediction. JSON export records score semantics. HOG does not control review. The `/video` page is a clearly labeled exploratory backup.

Images may be JPEG, PNG, WebP or BMP, with at least 32 pixels per side, at most 24 megapixels and a 40 MB request limit. This is not a batter detector, full-match analyzer, technique assessor or reliable unknown-shot detector.

## Read and compare

| Location | What to inspect |
|---|---|
| `output/pdf/CricketShot_Report_Complete.pdf` | Current CVPR report, supplementary evidence and full custom-code appendix |
| `output/pdf/CricketShot_Report_Main.pdf` | Reading copy, with supplementary evidence |
| `output/CricketShot_Overleaf.zip` | Editable LaTeX, figures, current complete PDF and code appendix |
| `output/presentation/CricketShot_Presentation.pptx` | 16 editable slides with notes; slide 16 is backup |
| `docs/revision_log.md` | Changes, reasons, and which numbers changed |
| `docs/course_mapping.md` | Verified lecture/PDF-page references |
| `docs/viva_guide.md` | Explanations and exercises for members |
| `results/revision/worked_trace.json` | One actual image through the functions, shapes, logits and scores |
| `results/revision/comparison.csv` | Recomputed CNN/HOG results and visual-group intervals |
| `results/revision/` | Current cache, HTTP and arithmetic verification |
| `results/verification.json`, `docs/verification.md` | Current document/test checks |
| `results/prior_verification.json`, `docs/prior_*` | Earlier records; not evidence of new checks |
| `results/images_initial_hash/` | Preserved optimistic preliminary image results |
| `results/pilot_fullframe/`, `results/metrics.json` | Preserved full-frame and fixed-crop video evidence |
| `data/`, `models/`, `demo/` | Raw data, feature caches, trained models and selected examples |
| `course_notes/` | User-supplied lectures, preserved for local study |

## Reproduce the added comparison

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/train_hog.py
.\.venv\Scripts\python.exe scripts/analyze_revision.py
.\.venv\Scripts\python.exe scripts/revision_figures.py
.\.venv\Scripts\python.exe scripts/verify_revision.py
.\.venv\Scripts\python.exe -m pytest -q
```

The public repository includes the runtime encoder, all trained numeric heads, manifests, predictions, source code, report and slides. Raw datasets, feature caches, supplied lecture PDFs, environments and the large local comparison archive stay outside Git. Dataset sources and checksums are recorded under `data/` and `docs/sources.md`. The local Full ZIP includes these research assets.

To restore the four selected image examples without training:

```powershell
.\.venv\Scripts\python.exe -m pip install requests
.\.venv\Scripts\python.exe scripts/prepare_demo_examples.py --download
```

This downloads the pinned image archive (about 676 MB). Add `--include-video` to also restore the four backup video examples (a further roughly 142 MB dataset plus the original pretrained graph). With raw data already present, omit `--download`. Empty example lists are hidden; own-image uploads work immediately after setup. Full-data tests and training require the downloaded data and feature caches; the repository-only inference check is `python scripts/verify_runtime.py`. HOG cache metadata binds its manifest, parameters and implementation. Its fixed 224-square grayscale descriptor has 9 unsigned bins, 16x16 cells, 2x2-cell L2-normalized blocks with one-cell stride, and 6,084 globally normalized values. No learned scaling/PCA is used. Only C is selected on validation; there is no HOG+CNN fusion search.

`train_hog.py` will reuse a matching feature cache and overwrite only its own HOG evidence/model. The original CNN predictions are used unchanged by `analyze_revision.py`. Do not search test splits, seeds or thresholds to improve the result.

For a full reconstruction from downloads, run `fetch_assets.py`, `prepare_data.py`, `fetch_images.py`, `prepare_images.py`, `strengthen_image_groups.py`, then `train_images.py`. These legacy preparation/training commands overwrite their output files: use a new scratch copy and compare resulting manifests before replacing evidence. `prepare_images.py` alone creates the initial weaker grouping; the strengthening step is required before using the final protocol. The actual delivered manifest is the comparison's authority.

## Rebuild deliverables

```powershell
.\.venv\Scripts\python.exe scripts/build_revision_report.py
.\.venv\Scripts\python.exe scripts/prepare_revision_slides.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_slides.ps1
.\.venv\Scripts\python.exe scripts/verify_submission.py
.\.venv\Scripts\python.exe scripts/package_project.py --kind full
```

Report rebuilding requires Tectonic at `tools/tectonic/tectonic.exe` (available from https://tectonic-typesetting.github.io/); the local Full ZIP includes this compiler. Initial LaTeX package downloads may be needed on another host. Slide rebuilding requires desktop PowerPoint and a Windows logon session. The Artifact Tool module/runtime was unavailable during this revision; native PowerPoint COM authored and rendered the editable deck. Running the demo does not require Office or LaTeX.

The supplied Overleaf URL was inspected on 19 October 2026 and currently displays a 2026 kit despite its 2022 slug. The revision preserves the original unmodified 2022 style and its two-column format. In Overleaf, compile `main.tex`; after edits, export `main.pdf` and use `python merge_report.py main.pdf code_appendix.pdf complete_updated.pdf`. The current full report is already included. This PDF merge avoids the 2022 style's embedded eso-pic/pdfpages conflict.

After source changes, rebuild the full code appendix and verify its inventory. After changes to `configs/submission.json`, rebuild and review the author block and presentation cover. The full archive includes all relevant project assets; machine-specific environments, build/Python caches and duplicate handoff ZIPs are excluded.

## What the evidence does and does not support

The 945 test images belong to **70 visual groups**, not 945 independent original photographs. No verified match/player identities exist. Modified copies prompted a stronger audit after the original 96.3% score; the corrected 65.5% result and new HOG study are exploratory. One validation set was reused for CNN model selection, temperatures and review.

Temperature scaling worsened test ECE from 0.043 to 0.094 and NLL from 0.859 to 0.898. We preserve its original settings and make no calibration-improvement claim. Review accepts 557/945 images, with 433 correct (58.9% coverage, 77.7% accepted accuracy). The 90% validation target is not a test guarantee. All heads can agree on a confident error.

The separate eight-class video pilot gets 30.2% accuracy, equal to its training-majority baseline, and 21.3% macro F1. Its crop revision followed test feedback. It is not a dependable general video recognizer or a directly comparable image baseline.

HOG, pretrained CNN features, logistic regression, multi-view features and review are established techniques. The course-level contribution is an implemented two-view difference study with controlled ablations and an inspectable demo. Its measured gain is weak; algorithmic novelty remains a limitation. There is no promise of a particular mark.

## Before submission

The supplied course email sets **6 November 2026** for the report and **7-8 November** for presentation/demo examination; the exact session is still pending. Every member must submit their own report with an actual contribution statement.

The five supplied names and roll numbers are listed above and in the report and slide cover. The project repository is linked above. HelloIITK submission remains a group action. Codex assistance is recorded in `docs/contributions.md`. Data, weights, papers, libraries and template sources are attributed in the report, `docs/sources.md` and `THIRD_PARTY_NOTICES.md`. The image source's license is unknown in downloaded metadata.
