# Image data card

- Source: https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset, version 2.
- Downloaded files: 4724. Decoded: 4724. Failures: 0.
- Labels: drive, legglance-flick, pullshot, sweep. These are upstream labels, not new group annotations.
- License in downloaded metadata: Unknown. Rights in the underlying photographs/broadcasts have not been independently verified.
- Exact duplicate excess: 1. Exact duplicates collapse before splitting.
- Initial hash/RGB qualifying pairs: 2809. Stronger quarter-turn/mirror links: 948.
- Geometrically verified ORB/RANSAC links: 1797 from 19857 candidate pairs.
- Final visual groups: 363. Conflicting final components: 0.
- Retained images: 4723. Train/validation/test: 2834/944/945.
- Group counts: 221/72/70.
- Split: stratified group folds, seed 604, test fold 0, validation fold 1, other folds training.

The collection visibly contains transformed images. The initial hash-only audit missed rotated/mirrored variants. We strengthened grouping after visual QA and preliminary scoring, preserved the first results, and reran the comparison. The final evaluation is exploratory under the revised protocol. The audit does not guarantee every transformed copy is linked. No verified match/player identities or demographic annotations are available. Scores are conditional on this archive and its upstream labels.

See `data/images/provenance.json`, `source_metadata.json`, `processed/audit.json`, `processed/manifest.csv` and `transformed_duplicate_edges.json`. The recorded edge list permits direct split-boundary verification. The raw data are not included in source-control defaults.
