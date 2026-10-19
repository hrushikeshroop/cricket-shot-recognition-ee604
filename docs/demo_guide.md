# Demonstration sequence

1. Start `run_demo.bat`, open http://127.0.0.1:6040 and state the four supported image labels.
2. Use the four success tiles: `image_00210` (drive), `image_01510` (glance/flick), `image_02934` (pull), and `image_03677` (sweep). Start with drive, show the actual global/crop inputs, and distinguish frozen extraction from trained classification. Compare its HOG result.
3. Use the **Confident error** tile, `image_04199`. It is upstream-labeled sweep but the CNN predicts pull. A confident answer and agreement can be wrong; do not invent an exact cause.
4. In the full local package, use `image_00014`, the **View disagreement** tile. Explain why review is requested and why HOG does not control it. It is optional in a public clone; restore it before the examination with the README command. The five other tiles are bundled and need no dataset download.
5. Upload a real image you may use. Inspect all four scores, then download JSON. Scores are relative outputs, not guaranteed correctness probabilities.
6. If asked, show `/video` as backup: 30.2% accuracy equals majority accuracy. Keep the different dataset/labels and reused pilot holdout explicit; its examples also require local preparation.

The five bundled JPEG inputs are in `demo/images/`. Their current predictions are computed from these actual files, not replayed benchmark outputs. The optional disagreement input is the original decoded PNG. `demo/image_examples.json` records each variant, SHA-256, upstream label, expected prediction and selection role. Examples were selected after evaluation; they illustrate behaviour and do not estimate accuracy.

Install dependencies and run the complete sequence before going offline. The submitted report explains the method and includes the implementation; `results/revision/worked_trace.json` records one measured prediction trace. The default app port is 6040.
