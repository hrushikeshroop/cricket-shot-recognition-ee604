# Current revision: visual and document review

## Public repository update, 19 October 2026

The five supplied names and roll numbers were checked on the rendered report's first page and the PowerPoint cover. Both show the repository under `hrushikeshroop`. The actual PDF hyperlink target is checked automatically, since a long visible URL wraps across lines.

The current report has five reading pages and 82 source-appendix pages (87 total). All five reading pages were inspected after the identity update; all 82 appendix pages were reviewed on four contact sheets, with readable source samples. The changed code remains inside the page bounds. The 16-slide deck was exported by PowerPoint again, with zero reported geometry issues; the changed cover was inspected. The earlier full-deck review below describes the unchanged slide layouts.

Current validation passed 20 tests and 34 submission checks. Actual HTTP checks reran five GET routes, all four image example/upload pairs and malformed-image rejection. Fresh home, success, failure and narrow UI screenshots were produced; the home and success renders were inspected again. Team footer content was also checked through the rendered HTML.

The current Overleaf package was extracted and compiled again: its five-page manuscript text matches the reading report, and its merge helper produced 87 pages. The current ZIP hash is recorded in `results/revision/overleaf_validation.json`.

`results/package_validation.json` now records a fresh export of exactly the Git index in an isolated environment with only `requirements.txt` installed. The export has no raw data, feature caches or selected example fixtures. An external real image passed the multipart-upload CNN/HOG path, and invalid bytes were rejected. The optional example list was empty, without broken example buttons. The earlier extracted Demo ZIP record is preserved as `results/prepublication_package_validation.json`; its older source hashes are historical.

## Earlier course-revision review

The following checks predate the team/account publication update above.

- The five-page reading report was rendered from the current PDF. All main pages and the supplementary page were inspected at readable size. The pipeline and HOG figures were enlarged, a confident-error caption was made self-contained, and a near-empty last page was removed by tightening the supplement and its links. The original CVPR 2022 style stays unmodified.
- All source-appendix pages were rendered; 13 contact sheets check pagination and wrapping, and full-size samples (pages 1, 31, 71 and 78) check code legibility. This is not a line-by-line semantic proof of every printed source line. Inventory coverage, current hashes and text bounds are also checked automatically.
- All 16 final slides were inspected individually from actual PowerPoint exports. Tables remain native/editable; method/gradient/UI images are embedded. PowerPoint reports zero text-height or outside-slide geometry issues. A successful automatic check alone is not treated as visual proof.
- The revised browser UI was rendered with fresh timestamps at 1440 pixels wide (home, success and confident failure) and 600 pixels wide (stacked layout). The HOG comparison, corrected model-score wording and explicit selected-example labels are visible. The 600-pixel view is not a claim of testing every phone width.
- Real local HTTP requests exercised all four examples and four multipart uploads. Browser screenshots exercised the actual JavaScript example/result flow. A native file-picker interaction was not automated. Invalid image bytes were rejected.
- The Overleaf ZIP was extracted into a fresh build folder and compiled. Its five-page manuscript text matches the local report. The included merge helper was run and produced 83 pages. This verifies the source package locally; no external Overleaf account or publication was used.

Initial sandboxed Chrome rendering failed; Windows process handles also caused a wait failure after a screenshot was written. The final renderer uses new isolated profiles and requires freshly written PNGs. Old screenshots were not accepted as current evidence. Private failed attempts remain excluded from deliverables.

The compact ZIP was extracted into a fresh folder, with a fresh Python 3.14.5 environment and system site packages disabled. Only requirements.txt was installed. All four image example and multipart-upload pairs, a malformed image rejection and one video example passed. Actual runtime packages and tested payload hashes are recorded in results/package_validation.json and results/revision/inference_environment.txt. The full archive contains those same runtime files plus the research assets.
