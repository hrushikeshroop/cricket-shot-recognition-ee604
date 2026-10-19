> Archived original plan. Current work is documented in revision_plan.md and revision_log.md.

# Implementation and scope record

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
