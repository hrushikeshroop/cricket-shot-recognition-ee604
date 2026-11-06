# Cricket Shot Recognition — EE604

Recognise one batting interval as **Drive, Defence, Flick, Pull, Square cut or Sweep**, or request **human review**. Drive combines cover and straight drives. A calibrated support score must be strictly above **0.40** for exactly one class; zero or multiple passing classes leave the automatic label unresolved.

## Run the demo

On Windows, run `setup.bat`, then `run_demo.bat`. Open http://127.0.0.1:6040. Python 3.14.5 was used for the supplied lock file. Manual Windows setup:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python app.py
```

On Linux/macOS, use `.venv/bin/python` instead of `.venv/Scripts/python`. Full data reproduction also requires libarchive's `bsdtar` for the RAR archives; Windows supplies a libarchive-based `tar`. GNU tar alone cannot extract these RAR files.

The shipped ONNX encoders and NumPy heads run on CPU. Upload AVI, MP4, MOV, MKV or WebM below **100 MiB** and **two minutes**. Select one batting interval up to **five seconds**; tighter trimming around the batter is preferable. This is a single-shot recogniser, not a full-match boundary detector. Resolution up to 4K and at most 600 selected frames are supported. Inference depends on the host CPU; measured warm-encoder median is 0.70 seconds for the supplied examples, excluding model loading and browser previews.

The interface shows six support scores, threshold marks, actual sampled frames and the reason for review. Support scores include taxonomy membership and need not sum to 100%. They do not guarantee correctness. A reviewer may download a local decision record; this neither sends their video elsewhere nor retrains the model.

## Measured results

These are **test** results after fitting/calibration/operating-point selection on separate source groups. This benchmark reuses earlier study sources; it is not an independently collected external-match test.

| Measure | Result |
| --- | ---: |
| Raw accuracy, all 1070 supported-shot clips | 75.8% |
| Raw macro F1 | 75.2% |
| Supported shots handled automatically | 70.0% |
| Accuracy among accepted supported shots | 83.8% |
| Outside shots sent to review (923 clips) | 77.8% |
| Accuracy across all automatic decisions, including outside-shot errors | 65.8% |

The default favours automatic coverage. The original stricter development setting is retained in `results/cautious_policy.json`; it is not the deployed threshold. Deferred cases are never counted as correct. Per-class workload, unseen-style rejection, confidence-only controls and source-group uncertainty are in the report and `results/metrics.json`.

## Submission files

- [Report with complete code appendix](submission/CricketShot_Report.pdf)
- [Presentation](submission/CricketShot_Presentation.pptx)
- `app.py`, `src/`, `static/`, `templates/`: working local demo
- `models/`: numeric deployment heads, frozen encoders and licences
- `scripts/`, `configs/`, `tests/`: data audit, fitting, evaluation and verification
- `data/`: author links/checksums, source metadata and the audited manifest
- `results/`: individual prediction evidence, aggregate metrics and fit/policy records
- `demo/`: real source-attributed success, review and failure examples

## Verify and reproduce

```text
.venv/Scripts/python scripts/verify.py --replay
```

Verification needs only runtime dependencies. It independently reconstructs metrics, selection, threshold counting and source-group intervals, checks hashes and freshly decodes every demo example. It does not train or modify the supplied benchmark evidence.

For development, activate a separate environment and install the official CPU wheels, then the development dependencies:

```text
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

Full source data need substantial local disk space and CPU feature extraction time. See [data and protocol](docs/data_card.md) and [model details](docs/model_card.md).

Rebuild in a **separate working copy**. Preserve its supplied evidence before fitting: create `.build`, then move its `results` directory to `.build/released_results`. Leave deployment models untouched unless intentionally testing a rebuilt model. Run:

```text
python scripts/fetch_data.py
python scripts/prepare_data.py
python scripts/extract_features.py
python scripts/train.py --phase classes
python scripts/train.py --phase scope
python scripts/calibrate.py
python scripts/refine_scope.py
python scripts/refine_classes.py
python scripts/calibrate.py
python scripts/select_operating_point.py
python scripts/extract_features.py --split test
python scripts/evaluate.py
```

`evaluate.py` writes a separate `results/deployment` bundle unless `--deploy` is explicitly passed. Fits and evaluation refuse to overwrite frozen evidence. All three development follow-ups were recorded before the current-run test evaluation. Hardware/library changes can change newly fitted heads; the shipped numeric weights are the exact reference. `export_encoders.py` builds comparison graphs separately and does not overwrite the supplied benchmark encoders.

Tests require development dependencies. On Windows, use a fresh workspace `--basetemp` path if the global temporary directory has restrictive permissions.

## Team and contributions

| Name | Roll number | Contribution |
| --- | --- | --- |
| Hrushikesh Roop Avvari | 230249 | Feature extraction, training and model comparisons |
| Voora Rakesh | 231174 | Confusion matrix, per-class metrics, result checks and slides |
| Gokavarapu Manikanta Sri Akshith | 230418 | Dataset preparation, evaluation and report |
| Lokesh Kumar Moond | 230606 | Upload/interval/review/JSON testing and demo preparation |
| Bollu Sharath chandra | 230303 | Human-review policy, interface and workflow tests |

Detailed individual statements are included in the report.

## Sources and limitations

Data: [CricShot10k author repository](https://github.com/Dihan69/CricShot10k), all fifteen archives. The authors specify non-commercial research use with citation; original video rights remain with the respective owners. Raw archives and the full video collection are not redistributed here. [Source links](docs/sources.md) and [third-party notices](THIRD_PARTY_NOTICES.md) identify data, encoders, papers, libraries and the course report template.

This is a course-level integration of focused/global temporal features, calibrated scope support and explicit candidate-set review. It is not a new CNN architecture. A substantial train–tuning gap remains. A hook may be conventional cricket but outside this taxonomy; an unfamiliar stroke may still be accepted incorrectly. Source metadata, upstream labels and fixed central crops limit generalisation. The six families are common coaching categories, not a verified worldwide frequency ranking.
