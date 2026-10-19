# Demonstration sequence

1. Start `run_demo.bat`, open http://127.0.0.1:6040 and state the four supported image labels.
2. Use `image_00210`, a selected success. Show actual global/crop inputs and distinguish frozen extraction from trained classification. Compare its HOG result.
3. Use `image_04199`, the preserved confident error. It is upstream-labeled sweep but the CNN predicts pull. A confident answer and agreement can be wrong; do not invent an exact cause.
4. Use `image_00014`, the view-disagreement case. Explain why review is requested and why HOG does not control it.
5. Upload a real image you may use. Inspect all four scores, then download JSON. Scores are relative outputs, not guaranteed correctness probabilities.
6. If asked, show `/video` as backup: 30.2% accuracy equals majority accuracy. Keep the different dataset/labels and reused pilot holdout explicit.

Examples were selected after evaluation and are marked accordingly. Keep the deck PDF and UI screenshots available as fallback. For questions, use the measured trace and viva guide. Current verification uses port 6060 if 6040 belongs to another running copy.
