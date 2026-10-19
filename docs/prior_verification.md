# Final verification record

| Check | Result | Evidence |
|---|---|---|
| Behavioral and integration tests | PASS | ...............                                                          [100%] 15 passed in 1.56s |
| global metric arithmetic | PASS |  |
| focus metric arithmetic | PASS |  |
| concat metric arithmetic | PASS |  |
| contrast metric arithmetic | PASS |  |
| Review arithmetic | PASS |  |
| CSV prediction IDs and test manifest agree | PASS |  |
| Verified transformed copies stay in one split | PASS | 2745 recorded edges checked; 0 violations |
| Initial hash components remain unsplit | PASS | 3108 original components checked |
| Raw pixel hashes independently sampled | PASS | 64 deterministic source images re-read; 0 mismatches |
| Full code appendix matches current source | PASS | 43 source/config files match SHA-256 inventory |
| Complete report includes entire appendix | PASS | 6 main + 57 appendix = 63 pages |
| Report contains current primary score | PASS |  |
| Report covers required sections | PASS | Abstract, Introduction, Proposed method, Experiments, Code, data and repositories |
| Every inventoried file appears in source appendix | PASS |  |
| PDF text stays inside page boundaries | PASS | [] |
| PowerPoint ZIP integrity | PASS |  |
| Presentation slide count | PASS | 16 |
| Presentation evidence tables remain native | PASS | 7 editable tables |
| Presentation contains current primary score | PASS |  |
| Speaker notes present | PASS | 16 |
| PowerPoint renderer geometry | PASS | [] |
| Overleaf package contains actual 2022 style and source appendix | PASS |  |

Submission items deliberately pending:

The portable ZIP was extracted into a fresh folder and exercised in a separate Python process. All four image examples and one video example returned successful inference using only the packaged models and examples. Raw training data, environments and caches are excluded. Tested code/model hashes are recorded in `results/package_validation.json`.

- Real GitHub URL, deferred by user
- Five members identities and truthful individual contribution statements

See docs/visual_review.md for actual render inspection. Automated checks alone do not prove visual quality or data-label correctness.
