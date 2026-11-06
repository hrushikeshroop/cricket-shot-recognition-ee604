# Data and source protocol

Source: https://github.com/Dihan69/CricShot10k . `data/provenance.json` records the fifteen author archive URLs, exact SHA256 checksums and file counts. `data/source_videos.csv` preserves the source-video spreadsheet; `previous_sources.csv` preserves inherited grouping information. `manifest.csv` records every original clip, target mapping, source group, split, checksum and exclusion reason.

The actual archives contain **10091** clips; **9965** survive audit. Published summary counts differ; the project does not infer a cause. Six families use seven native categories: cover/straight drive → Drive, plus defence, flick, pull, square cut and sweep. Hook, late cut, down the wicket, lofted legside and lofted offside supply binary scope exposure. Reverse sweep, scoop and upper cut are withheld from every development role; their non-test clips remain unused.

| Role | Supported | Outside | Source groups |
| --- | ---: | ---: | ---: |
| train | 3210 | 2283 | 277 |
| tune | 357 | 258 | 33 |
| calibration | 359 | 263 | 33 |
| policy | 354 | 265 | 30 |
| test | 1070 | 923 | 109 |

Training alone fits coefficients, scaling and PCA. Tuning selects representations, neural epochs and scope settings. Calibration fits temperature/sigmoid parameters; policy groups select the operating point. Current-run test outcomes do not select these choices.

The audit excludes compilation sources, duplicate copies and conflicting-target duplicates. Source IDs, hashes, decoded duplicates, conservative three-frame pHashes and normalised source metadata form connected components. Components remain disjoint across roles. Near-duplicate rules and split details are explicit in `prepare_data.py` and `audit.json`.

The inherited test sources were evaluated in the preceding eight-class study. This is a **reused source-group benchmark**, not a new external collection. Source groups are not independently verified matches, and transformed replays or incomplete metadata can remain. Dataset labels have not been independently reannotated by batting experts. Pretraining overlap has not been exhaustively audited.

The authors request non-commercial research use and citation; they retain no blanket transfer of the original footage rights. Demo clips are a small source-attributed research subset. The download script checks pinned archive hashes and inventories, rejects archive traversal/links and does not require private browser cookies. The audit refuses incomplete category inventories. Full raw data and caches stay local.
