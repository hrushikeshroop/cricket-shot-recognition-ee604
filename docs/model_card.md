# Model and review policy

Six-family CPU video recogniser, schema 4. Selected classifier: `full_motion_pca512_ensemble` from 29 recorded candidates by tuning macro F1. Training accuracy 99.5%; tuning accuracy 82.1%. This remaining gap indicates overfitting; reducing labels alone does not fix it. The smaller original PCA256 head generalised less well on tuning but had a smaller gap. The chosen higher-dimensional ensemble improves tuning accuracy while fitting training almost perfectly.

Frozen MC3-18 Kinetics features describe the full view. Frozen ImageNet MobileNetV2 features describe the central crop from original pixels. Both use 24 sampled positions. Ordered focused bins and a compensated-activity interval provide explicit temporal descriptors. The activity measure has no contact/phase annotations. A fixed crop assumes an approximately central broadcast batter.

Training-only scaling, clipping and whitened PCA precede small ReLU heads. Dropout, weight decay, feature noise, label smoothing and tuning-based early stopping regularise fitting. The original 26-candidate study was followed by three PCA512 ensembles, selected without current-run test outcomes. `class_followup.json` records this development follow-up.

The scope head distinguishes membership in the six families from five auxiliary shot categories. An initial linear PCA128 context head reached 0.850 tuning AUROC but deferred too many supported shots. A recorded nonlinear follow-up selected `full_ordered_pca128_h32_scope_seed605` at 0.886 tuning AUROC. This is taxonomy membership, not an expert assessment of textbook technique.

Separate calibration groups fit one class temperature and a sigmoid for scope logits. Support is `scope_score * conditional_class_score`, so six supports sum to the scope score. The scores depend on the calibration mixture and are not universal correctness probabilities. The scope exposure loss differs from the published deep outlier-exposure method.

At the deployed common threshold **0.40**, exactly one strictly passing support yields an automatic class. None or multiple yield `human_review` and `class: null`. The top candidate is separate. The coverage-oriented point was chosen from development results before opening current-run test scores; original stricter accuracy/rejection goals and the cautious reference are retained. No per-clip threshold change or filename-specific exception is used.

Known automatic coverage: 70.0%; accepted-known accuracy: 83.8%; outside-shot review: 77.8%. Mixed automatic accuracy: 65.8%. Review cases are not counted as correct. See `metrics.json` for source-group bootstrap intervals, every class, withheld-style results and confidence-only ablations. The review component is useful only to the degree supported by those comparisons; it does not guarantee every unusual shot is rejected.

The runtime validates numeric parameters and deployment hashes; no pickle is loaded. Real-clip replay checks preprocessing/score agreement and actual decisions. Training and runtime full/focused interval pixels match on lossless tests. Verification provenance retains the generated wiring pattern's own SHA separately from example hashes.

Novelty is the specified course-level combination of focused/global temporal features, calibrated scope support and an explicit candidate-set review workflow, with coverage, errors, unseen styles and matched-coverage controls measured. MC3, MobileNet, PCA, calibration, selective classification and outlier exposure are established ideas. No new backbone or universal novelty claim is made.
