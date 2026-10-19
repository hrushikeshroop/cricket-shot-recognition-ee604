# Cricket Shot Recognition - EE604

Classify a cricket image as **drive, leg glance/flick, pull or sweep**. The project compares a HOG descriptor with frozen MobileNetV2 features and tests whether combining a full-image view, a fixed central crop and their absolute difference helps recognition.

Repository: https://github.com/hrushikeshroop/cricket-shot-recognition-ee604

| Name | Roll number |
|---|---|
| Hrushikesh Roop Avvari | 230249 |
| Voora Rakesh | 231174 |
| Gokavarapu Manikanta Sri Akshith | 230418 |
| Lokesh Kumar Moond | 230606 |
| Bollu Sharath chandra | 230303 |

## Results

All image methods use the same 945 test images in 70 visual groups. The contrast CNN scores **65.5% accuracy and 64.9% macro F1** (619 correct); concatenation scores 65.2% accuracy (616 correct). Their paired group-bootstrap accuracy-difference interval is -1.46 to +2.38 percentage points, so a reliable advantage is not established. The HOG baseline scores **35.3% accuracy and 34.5% macro F1** (334 correct).

These are exploratory results. Duplicate grouping was strengthened after preliminary evaluation; verified match and player identities are unavailable. Temperature scaling worsened test calibration. The review rule accepts 557 images with 433 correct; it is a diagnostic rather than a correctness guarantee. The separate eight-class video pilot scores 30.2% accuracy, equal to its majority baseline, and is available as a backup demonstration.

## Run the demo

Use Python 3.11 or newer. On Windows, run `setup.bat`, then `run_demo.bat`, and open http://127.0.0.1:6040. Alternatively:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Upload a JPEG, PNG, WebP or BMP image. The page shows both model input views, four CNN class scores, a review reason, the separate HOG prediction and JSON export. Images must be at least 32 pixels per side and at most 24 megapixels; requests are limited to 40 MB. Inference runs locally with the included model files. No database or API key is required. The model does not detect the batter, analyze a full match, grade technique or reliably reject unknown shots.

Five selected image illustrations are included: one successful example for each class and a confident error. They work immediately after setup, without downloading the training archive. Their predictions are computed when clicked; the JPEG illustrations can differ from the original benchmark inputs. All examples were selected after evaluation.

For an examination without internet access, install dependencies in advance and launch the app once. Check all five examples and keep the report and slide PDF locally. The full local package also includes the original disagreement case, `image_00014`.

The public repository excludes raw datasets, large feature caches, lecture PDFs, machine environments and temporary build files. To restore the optional original disagreement example from the pinned public archive without training:

```powershell
.\.venv\Scripts\python.exe -m pip install requests
.\.venv\Scripts\python.exe scripts/prepare_demo_examples.py --download
```

The optional image archive is about 676 MB. Add `--include-video` to restore the backup video examples too. With raw data already present, omit `--download`. The five bundled illustrations and own-image uploads work without this optional download. Source URLs, dataset versions and checksums are recorded under `data/` and in [the source register](docs/sources.md).

## Report, presentation and evidence

| Location | Contents |
|---|---|
| [Complete report](output/pdf/CricketShot_Report_Complete.pdf) | CVPR manuscript, supplementary experiments and complete custom-code appendix |
| [Reading report](output/pdf/CricketShot_Report_Main.pdf) | Manuscript and supplementary experiments |
| [Editable LaTeX package](output/CricketShot_Overleaf.zip) | Sources, figures, code appendix and complete PDF |
| [Presentation](output/presentation/CricketShot_Presentation.pptx) | 16 editable slides and speaker notes; slide 16 is backup |
| [Course mapping](docs/course_mapping.md) | Connections to the supplied lecture topics and page references |
| [Demo guide](docs/demo_guide.md), [method guide](docs/viva_guide.md) | Example sequence, equations and implementation explanations |
| [Data card](docs/data_card.md), [model card](docs/model_card.md) | Data protocol, model design and measured limitations |
| `results/images/`, `results/hog/`, `results/revision/` | Saved predictions, comparisons, confidence analysis and verification |
| `results/images_initial_hash/`, `results/pilot_fullframe/` | Earlier experiment evidence, retained to explain the evaluation changes |

The contrast feature corrects 17 concatenation errors but loses 14 correct decisions. The post-hoc breakdown and selected cases are in `results/revision/view_diagnostics.json`; they describe the small net gain without establishing a reliable improvement. The report compares two earlier cricket video methods with this image pipeline. [The method guide](docs/viva_guide.md) explains the design choices and their limits.

## Verification and reproduction

For an inference check using only the public repository and runtime dependencies:

```powershell
.\.venv\Scripts\python.exe scripts/verify_runtime.py
```

Full research tests require development dependencies, raw data and feature caches. On the full local dataset copy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/train_hog.py
.\.venv\Scripts\python.exe scripts/analyze_revision.py
.\.venv\Scripts\python.exe scripts/revision_figures.py
.\.venv\Scripts\python.exe scripts/verify_revision.py
.\.venv\Scripts\python.exe -m pytest -q
```

For a reconstruction from downloads, run `fetch_assets.py`, `prepare_data.py`, `fetch_images.py`, `prepare_images.py`, `strengthen_image_groups.py`, then `train_images.py`. Run these in a scratch copy: they replace generated data and model files. The strengthening step is required for the final visual-group protocol. Do not select new splits, seeds or thresholds using test results. HOG uses the same final manifest and selects only C on validation, without a fusion search.

Rebuild the documents with `scripts/build_revision_report.py`, `scripts/prepare_revision_slides.py` and `scripts/build_slides.ps1`, then run `scripts/verify_submission.py`. Report rebuilding needs Tectonic at `tools/tectonic/tectonic.exe`; slide rebuilding needs desktop PowerPoint on Windows. The demo needs neither. In Overleaf, compile `main.tex`; after editing, use the included `merge_report.py` to append the code PDF. The supplied unmodified CVPR 2022 style is retained.

Dataset, pretrained-model, paper, library and template sources are listed in [docs/sources.md](docs/sources.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The image dataset metadata reports an unknown license; this project does not grant rights to the source photographs.
