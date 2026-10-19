# Cricket Shot Recognition — EE604

Classify one cricket image as **drive, leg glance/flick, pull or sweep** using frozen MobileNetV2 features and a linear classifier. A HOG classifier provides a separate baseline. The tested extension combines full-image and central-crop features with their absolute difference.

| Team member | Roll number |
|---|---|
| Hrushikesh Roop Avvari | 230249 |
| Voora Rakesh | 231174 |
| Gokavarapu Manikanta Sri Akshith | 230418 |
| Lokesh Kumar Moond | 230606 |
| Bollu Sharath chandra | 230303 |

## Submission

- [Project report, including the full implementation appendix](submission/CricketShot_Report.pdf)
- [Presentation](submission/CricketShot_Presentation.pptx)

## Run the application

Use Python 3.11 or newer. On Windows, run `setup.bat`, then `run_demo.bat`, and open http://127.0.0.1:6040. Or run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Upload a JPEG, PNG, WebP or BMP image, or select one of the five bundled examples. They cover all four labels and a confident failure. The page shows the two input views, CNN scores, review reason, HOG result and downloadable prediction JSON. All examples were selected after evaluation. Inference is local and requires no database, cloud service or API key.

The model supports four labels. It does not detect the batter, locate contact or grade batting technique. Unknown shots can receive confident predictions. Inputs must be at least 32 pixels per side and at most 24 megapixels; requests are limited to 40 MB.

## Code and data

| Folder | Contents |
|---|---|
| `src/`, `app.py`, `templates/`, `static/` | Feature extraction, inference and application interface |
| `scripts/` | Dataset preparation, training, evaluation and runtime checks |
| `models/`, `configs/` | Required weights, classifier heads and experiment settings |
| `demo/` | Example images and labels |
| `data/` | Dataset source metadata, split manifests and grouping records |
| `results/` | Numerical results and saved predictions supporting the report |
| `tests/` | Descriptor, model, API and provenance checks |

Dataset and pretrained-weight links, versions and attribution are in [docs/sources.md](docs/sources.md). Raw datasets and feature caches are excluded from Git. The [data card](docs/data_card.md), [model card](docs/model_card.md) and [demo guide](docs/demo_guide.md) describe the protocol and use limits.

Five examples work without a dataset download. To restore the optional original disagreement example, install `requests` and run `scripts/prepare_demo_examples.py --download`; the pinned archive is about 676 MB. Add `--include-video` for the separate exploratory video examples.

## Results and checks

On 945 test images in 70 visual groups, contrast obtains **65.5% accuracy and 64.9% macro F1** (619 correct). Concatenation gets 616 correct. The paired group-bootstrap accuracy-difference interval is **−1.46 to +2.38 percentage points**, so a reliable gain is not established. HOG obtains **35.3% accuracy and 34.5% macro F1**. The study is exploratory because grouping was strengthened after preliminary evaluation; verified match/player identities are unavailable. Temperature scaling worsened test calibration. Agreement is a review diagnostic, not a correctness guarantee.

Check inference using only runtime dependencies:

```powershell
.\.venv\Scripts\python.exe scripts/verify_runtime.py
```

Training and full research tests require `requirements-dev.txt`, downloaded data and feature caches. The training code retains the recorded split and selects classifier regularization using validation data. Run training commands in a scratch copy because they replace generated models and results. Do not choose new settings from test feedback.

Source photographs retain their upstream rights; downloaded image metadata lists an unknown license. Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
