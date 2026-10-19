# Verification of report and presentation

The current reading report has 5 pages; the source appendix has 79 pages, giving 84 pages in the complete report. The appendix inventory covers 54 current executable source/configuration files with SHA-256 checks. All file headings and PDF text boundaries pass the document checks.

The report was rendered after the latest edit. The current report author block and presentation cover were inspected after the team-name update. The longer name fits correctly. Previous render checks covered the main text, figures and appendix pagination. The unmodified CVPR 2022 style is retained.

PowerPoint exported 16 editable slides and a PDF; its layout check reported zero text-height or outside-slide issues. Notes, native tables, team names and the repository link were checked. The updated cover fits within its existing layout.

Current checks passed: 20 behavioral tests, 34 submission checks and 150 research/cache checks. The 64 fresh two-view CNN feature comparisons had maximum absolute difference zero. Actual local HTTP requests checked five GET routes, four image example/multipart-upload pairs and malformed-image rejection.

The Overleaf ZIP was extracted and compiled: its five-page manuscript text matches the reading report, and its merge helper produced 84 pages. Exact package hashes are recorded in results/revision/overleaf_validation.json.

A fresh Git-index export ran in an isolated Python environment installed only from requirements.txt, without raw datasets, feature caches or selected example files. A real external image passed CNN/HOG inference through a multipart upload; invalid bytes were rejected. Runtime versions and tested source/model hashes are in results/package_validation.json.

These checks verify the stated behaviors and files. They do not establish that every upstream label is correct or that visual grouping detects every related image.
