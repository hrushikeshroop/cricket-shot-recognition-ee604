"""Synchronize the handoff notes and model/data cards with final saved evidence."""
from pathlib import Path
import json
import re
ROOT=Path(__file__).resolve().parents[1]
def p(x):return f'{100*x:.1f}%'

def main():
    r=json.loads((ROOT/'results/images/metrics.json').read_text());a=json.loads((ROOT/'data/images/processed/audit.json').read_text())
    initial=json.loads((ROOT/'results/images_initial_hash/metrics.json').read_text());m=r['models']['contrast'];review=r['review']
    latex_safe='undefined' if review['test_accuracy'] is None else p(review['test_accuracy'])
    summary=(f'The proposed image model scores **{p(m["test"]["accuracy"])} accuracy and {p(m["test"]["macro_f1"])} macro F1 on {r["n"]["test"]} test images in {a["split_groups"]["test"]} transformed-photo groups**. '
             f'The global baseline scores {p(r["models"]["global"]["test"]["accuracy"])} accuracy, and concatenation scores {p(r["models"]["concat"]["test"]["accuracy"])}. Differences between heads are modest. '
             f'The review rule accepts {review["test_accepted"]}/{r["n"]["test"]} images with {latex_safe} accepted accuracy. This is an observation, not a guarantee.\n\n'
             f'**The initial hash-only contrast score was {p(initial["models"]["contrast"]["test"]["accuracy"])}. Visual QA exposed transformed copies, and a stronger audit reduced the collection to {a["visual_groups"]} connected photo groups.** '
             'The final results use that revised grouping and are explicitly exploratory. Image match/player identities remain unavailable; related copies may still escape the audit. The earlier high percentage is preserved as a preliminary result, not used as the headline score.')
    readme=(ROOT/'README.md').read_text()
    readme=re.sub(r'The proposed image model scores.*?\n\nThe video model',summary+'\n\nThe video model',readme,flags=re.S)
    if 'scripts/strengthen_image_groups.py' not in readme:
        readme=readme.replace('.\\.venv\\Scripts\\python.exe scripts/train_images.py', '.\\.venv\\Scripts\\python.exe scripts/strengthen_image_groups.py\n.\\.venv\\Scripts\\python.exe scripts/train_images.py')
    readme=readme.replace('It does not invent match IDs.','The next audit links quarter turns, mirrors and ORB/RANSAC-verified nearest neighbors before the final split. It does not invent match IDs.')
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    viva=(ROOT/'docs/novelty_and_viva.md').read_text()
    answer=(f'**Did the proposed feature improve the score?** On the revised split, contrast scores {p(m["test"]["accuracy"])} accuracy and concatenation scores {p(r["models"]["concat"]["test"]["accuracy"])}. '
            f'Global-only scores {p(r["models"]["global"]["test"]["accuracy"])}. These are modest one-split differences; concatenation has the best validation macro F1. '
            'The more substantial observation is how grouping transformed copies reduces the optimistic initial score. No general superiority is established.')
    viva=re.sub(r'\*\*Did the proposed feature improve the score\?\*\*.*?\n\n',answer+'\n\n',viva,flags=re.S)
    viva=re.sub(r'On this test set, \d+/\d+ images pass and accepted accuracy is [\d.]+%.',f'On this test set, {review["test_accepted"]}/{r["n"]["test"]} images pass and accepted accuracy is {latex_safe}.',viva)
    viva=viva.replace('The image test set does not make these choices.','The revised image test set does not select these classifier parameters. The grouping revision followed preliminary scores and visual QA, so the final evaluation is exploratory.')
    (ROOT/'docs/novelty_and_viva.md').write_text(viva,encoding='utf-8')
    card=f'''# Image data card

- Source: https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset, version 2.
- Downloaded files: {a['files_found']}. Decoded: {a['decoded_labeled']}. Failures: {len(a['errors'])}.
- Labels: drive, legglance-flick, pullshot, sweep. These are upstream labels, not new group annotations.
- License in downloaded metadata: Unknown. Rights in the underlying photographs/broadcasts have not been independently verified.
- Exact duplicate excess: {a['exact_duplicate_excess']}. Exact duplicates collapse before splitting.
- Initial hash/RGB qualifying pairs: {a['near_duplicate_pairs']}. Stronger quarter-turn/mirror links: {a['transform_hash_edges']}.
- Geometrically verified ORB/RANSAC links: {a['orb_verified_edges']} from {a['orb_candidates']} candidate pairs.
- Final visual groups: {a['visual_groups']}. Conflicting final components: {a['final_conflicting_components']}.
- Retained images: {a['exact_representatives_kept']}. Train/validation/test: {r['n']['train']}/{r['n']['val']}/{r['n']['test']}.
- Group counts: {a['split_groups']['train']}/{a['split_groups']['val']}/{a['split_groups']['test']}.
- Split: stratified group folds, seed 604, test fold 0, validation fold 1, other folds training.

The collection visibly contains transformed images. The initial hash-only audit missed rotated/mirrored variants. We strengthened grouping after visual QA and preliminary scoring, preserved the first results, and reran the comparison. The final evaluation is exploratory under the revised protocol. The audit does not guarantee every transformed copy is linked. No verified match/player identities or demographic annotations are available. Scores are conditional on this archive and its upstream labels.

See `data/images/provenance.json`, `source_metadata.json`, `processed/audit.json`, `processed/manifest.csv` and `transformed_duplicate_edges.json`. The recorded edge list permits direct split-boundary verification. The raw data are not included in source-control defaults.
'''
    (ROOT/'docs/data_card.md').write_text(card,encoding='utf-8')
    model=f'''# Model card

Primary task: four-class still-image recognition. Intended use: an inspectable local course demonstration and reproducible experiments. Input: one image with a visible batter playing a supported shot. Output: four calibrated scores and a review decision.

The shared encoder is frozen ImageNet MobileNetV2 from the ONNX Model Zoo. Each 224x224 view gives 1,280 pooled features. The proposed normalized descriptor concatenates global, central-crop and absolute-difference features (3,840 values). Its balanced multinomial logistic head has 15,364 parameters. Two single-view heads share those same two encoder passes and provide an agreement check. A concatenation-only head is retained for ablation.

Measured recognition: {p(m['test']['accuracy'])} accuracy, {p(m['test']['macro_f1'])} macro F1 on {r['n']['test']} images in {a['split_groups']['test']} visual groups. Temperature: {m['temperature']:.6f}. Review threshold: {review['threshold']:.2f}. The rule also requires all three top labels to agree. Test coverage: {p(review['test_coverage'])}; accepted accuracy: {latex_safe}. Review is a validation heuristic, not a certified risk bound or an unknown-class detector.

The initial hash-only contrast score ({p(initial['models']['contrast']['test']['accuracy'])}) was optimistic in the presence of transformed copies. Final scores follow a stronger audit and revised split after visual QA. No unseen-match, unseen-player, coaching, ball tracking or technique-quality claim is supported. Unfamiliar cameras, occlusion, cropped batters, unknown shots and non-cricket pictures can yield incorrect or confident predictions. The full raw annotation set has not been independently verified.

The separate eight-class video model uses nine sampled frames, phase means, late-minus-early difference and spread. It scores 30.2% accuracy and 21.3% macro F1 under source grouping, and its ROI revision reused a pilot holdout. It is an exploratory extension, not a validated production video recognizer.

Numeric classifier files avoid unpickling Python objects. The local server binds only to localhost, serializes inference and discards temporary uploads. Run the tests and review `results/verification.json` before using a modified model or preprocessing pipeline.
'''
    (ROOT/'docs/model_card.md').write_text(model,encoding='utf-8')
    plan='''# Implementation and scope record

The original plan started with a source-grouped eight-class video classifier. The full-frame and fixed-crop experiments were implemented and measured. Their low scores motivated a separate four-class image demonstration. Both tasks and their different limitations remain visible in the report.

The image plan compared global, crop, concatenation and absolute-difference features, with temperature scaling and an agreement review gate. All were implemented. Visual QA then exposed transformed copies that escaped the first hash rule. A stronger transform/geometric audit revised the grouping, and every image head and stress test was rerun. Preliminary predictions remain preserved.

| Planned requirement | Implemented evidence |
|---|---|
| Real data and reproducibility | Pinned downloads, metadata, SHA-256, audited manifests |
| Recognition method and novelty hypothesis | Shared encoders, two views, contrast block, temporal pilot |
| Controlled baselines | Same image groups, frozen features and C grid for all heads |
| Error and robustness analysis | Class metrics, confusion matrix, darkness, blur, mirror, video reversal |
| Confidence and review | Validation temperatures, threshold and three-head agreement |
| Usable local interface | Uploads, selected examples, actual model inputs, scores, JSON export |
| Course report and entire code | Actual CVPR 2022 style, full line-numbered source appendix |
| Presentation | Native editable PowerPoint tables and diagram, source notes and PDF fallback |
| Verification | Behavioral tests, split/edge checks, score recomputation, source inventory, native render review |
| Handoff organization | README, model/data cards, viva guide, demo guide and submission checklist |

Not implemented or not established: learned batter detection, pose/ball tracking, contact localization, coaching quality, unknown-class certification, independent match/player evaluation, new human annotations, a GitHub publication or HelloIITK submission. The user deferred the repository decision; names and real contributions remain for the five members.
'''
    (ROOT/'docs/project_plan.md').write_text(plan,encoding='utf-8')
    print('Documentation synchronized with final evidence')

if __name__=='__main__':main()
