# Revision plan and evidence

The original workspace is retained outside `revision/CricketShot`. All edits and
new experiments are in this separate project. `original_files_baseline.json`
records the original source/results/documents copied into this revision.
The original large ZIP was absent at revision start; the original project files,
portable ZIP, report, slides and archived verification records were available.

1. Read supplied lectures and verify the suggested page references. Record the
   course mapping and the distinction between course foundations and extensions.
2. Preserve the image manifest, test identities, CNN weights, original predictions,
   calibration and review choices. Preserve both preliminary image evidence and
   video pilot evidence. Do not search alternative seeds or splits.
3. Fix one HOG configuration, compare it with all four existing CNN descriptors,
   select only C on validation, and label this additional experiment exploratory.
4. Recompute comparisons from individual predictions, resampling all 70 visual
   groups together for paired uncertainty. Explain the three-correct-image
   contrast-versus-concatenation difference without a substantial-gain claim.
5. Expand compressed ML code; retain shared training/inference preprocessing.
   Compare HOG in the image demo and replace the unsupported calibrated-score
   wording. Keep correct, confident-error and disagreement examples.
6. Rebuild the report, full current-code appendix, Overleaf source, and 16-slide
   deck. Move detailed audit/video/stress evidence to supplementary material.
7. Run current tests, saved-prediction arithmetic, cache/export parity and upload
   checks. Render and inspect the rebuilt documents and UI. Archive all relevant
   assets, with a new manifest and checksum verification.

This plan predates the user's later authorization to publish the public GitHub
repository. Team details and the repository URL are now in configs/submission.json.
Course submission remains a group action. Novelty is a limited
course-level feature study; the revision does not invent a new architecture or
promise a particular mark.
