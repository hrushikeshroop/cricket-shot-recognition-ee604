# Verification of report and presentation

The reading report has 6 pages; the source appendix has 83 pages, giving 89 pages in the complete report. The appendix inventory covers 54 current executable source/configuration files with SHA-256 checks. All file headings and PDF text boundaries pass the document checks.

The report was rendered after the revision. All six reading pages were inspected, including the literature table, method overview, decision-change analysis and references. The appendix was rendered in full, scanned as contact sheets and sampled at full size. The unmodified CVPR 2022 style is retained.

PowerPoint exported 16 editable slides and a PDF with zero text-height or outside-slide issues. The changed literature, method, decision-change, review and demonstration slides were inspected at full size; the whole deck was scanned for unintended changes. A visible overlap under the literature table was corrected and rendered again. Team identities, notes and native evidence tables remain present.

Current checks passed: 23 behavioral/integration tests, 34 technical submission checks and 150 research/cache checks. The 64 fresh two-view CNN feature comparisons had maximum absolute difference zero. Actual local HTTP requests checked five GET routes, six image example/multipart-upload pairs and malformed-image rejection. Contribution statements are absent, so the verifier separately reports course submission readiness as false; passing technical checks does not satisfy that remaining requirement.

The Overleaf ZIP was extracted and compiled: its six-page manuscript text matches the reading report, and its merge helper produced 89 pages. The package hash is recorded in `results/revision/overleaf_validation.json`.

A fresh Git-index export ran in an isolated Python environment with system site packages disabled and only runtime requirements installed. It contained no raw datasets or feature caches. All five bundled illustrations passed checksum, example/upload parity and expected-behaviour checks, covering all four successful labels and the confident error. An external photograph also passed multipart CNN/HOG inference; invalid bytes were rejected. Runtime versions and source/model hashes are in `results/package_validation.json`.

Current screenshots came from the running app in Chrome at desktop widths and a true 430-pixel mobile viewport. All six local tiles and model-input images loaded, and no horizontal overflow was measured. The image used on slide 14 is the revised local interface.

These checks verify the stated behaviours and files. They do not establish that every upstream label is correct, that grouping detects every related image or that contrast reliably improves recognition.
