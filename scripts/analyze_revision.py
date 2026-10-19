"""Recompute comparisons and paired uncertainty from saved predictions only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import hashlib
import json
import numpy as np
from train_evaluate import metrics


def confusion_statistics(matrix):
    """Accept (..., 4, 4) counts and compute image-weighted accuracy/macro F1."""
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    accuracy = diagonal.sum(-1) / matrix.sum(axis=(-2, -1))
    denominator = matrix.sum(-1) + matrix.sum(-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                   where=denominator > 0).mean(-1)
    return np.stack([accuracy, f1], axis=-1)


def main():
    out = ROOT / "results/revision"
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "results/images/metrics.json").read_text())
    rows = list(csv.DictReader((ROOT / "data/images/processed/manifest.csv").open(encoding="utf-8", newline="")))
    test_rows = [r for r in rows if r["split"] == "test"]
    with np.load(ROOT / "results/images/test_predictions.npz", allow_pickle=False) as arrays:
        y, ids, groups = [arrays[k].copy() for k in ("y", "ids", "groups")]
        predictions = {mode: arrays["p_" + mode].copy() for mode in ("global", "focus", "concat", "contrast")}
        accepted = arrays["accepted"].copy()
    with np.load(ROOT / "results/hog/test_predictions.npz", allow_pickle=False) as arrays:
        assert np.array_equal(ids, arrays["ids"])
        assert np.array_equal(y, arrays["y"])
        predictions["hog"] = arrays["p"].copy()
    assert ids.tolist() == [r["id"] for r in test_rows]
    assert len(np.unique(groups)) == 70 and len(y) == 945
    classes = old["classes"]
    unique = np.unique(groups)
    modes = list(predictions)
    group_matrices = np.zeros((len(modes), len(unique), 4, 4), dtype=int)
    for mi, mode in enumerate(modes):
        pred = predictions[mode].argmax(1)
        for gi, group in enumerate(unique):
            index = np.flatnonzero(groups == group)
            np.add.at(group_matrices[mi, gi], (y[index], pred[index]), 1)
    rng = np.random.default_rng(17)
    # Resample whole visual groups. Every image of a drawn group is retained.
    # Use the identical draws for all methods to form paired differences.
    draws = np.stack([np.bincount(rng.choice(len(unique), size=len(unique), replace=True),
                                  minlength=len(unique)) for _ in range(2000)])
    boot_matrices = np.einsum("bg,mgij->mbij", draws, group_matrices)
    boot_values = confusion_statistics(boot_matrices)
    comparisons = {}
    for mi, mode in enumerate(modes):
        measured = metrics(y, predictions[mode], classes)
        if mode != "hog":
            assert abs(measured["accuracy"] - old["models"][mode]["test"]["accuracy"]) < 1e-12
            assert abs(measured["macro_f1"] - old["models"][mode]["test"]["macro_f1"]) < 1e-12
        interval = np.percentile(boot_values[mi], [2.5, 97.5], axis=0)
        comparisons[mode] = {"test": measured, "correct": int((predictions[mode].argmax(1) == y).sum()),
                             "accuracy_95": interval[:, 0].tolist(),
                             "macro_f1_95": interval[:, 1].tolist()}
    ci, cj = modes.index("contrast"), modes.index("concat")
    difference = boot_values[ci] - boot_values[cj]
    paired_interval = np.percentile(difference, [2.5, 97.5], axis=0)
    paired = {"comparison": "contrast minus concatenation", "correct_difference": comparisons["contrast"]["correct"] - comparisons["concat"]["correct"],
              "accuracy_difference": comparisons["contrast"]["test"]["accuracy"] - comparisons["concat"]["test"]["accuracy"],
              "accuracy_difference_95": paired_interval[:, 0].tolist(),
              "macro_f1_difference_95": paired_interval[:, 1].tolist()}
    info = {"status": "Exploratory; prior results already known; no new test-based model choice",
            "models": comparisons, "paired_contrast_concat": paired,
            "uncertainty": {"unit": "visual duplicate group", "groups": len(unique), "draws": 2000, "seed": 17,
                            "method": "percentile cluster bootstrap; image-weighted statistics within whole-group resamples",
                            "limitations": "Conditions on fixed training, split and imperfect visual groups; omits training variability and unknown match correlation."},
            "review": {"accepted": int(accepted.sum()),
                       "correct": int(((predictions["contrast"].argmax(1) == y) & accepted).sum()),
                       "coverage": float(accepted.mean()),
                       "accepted_accuracy": float((predictions["contrast"].argmax(1)[accepted] == y[accepted]).mean()),
                       "threshold_unchanged": old["review"]["threshold"]},
            "calibration": {"uncalibrated": old["models"]["contrast"]["test_uncalibrated"],
                            "scaled": old["models"]["contrast"]["test"],
                            "conclusion": "Test NLL and ECE worsened. Retain original temperature; no improvement claim."},
            "manifest_sha256": hashlib.sha256((ROOT / "data/images/processed/manifest.csv").read_bytes()).hexdigest()}
    (out / "comparison.json").write_text(json.dumps(info, indent=2))
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "correct", "n", "accuracy", "macro_f1", "accuracy_ci_low", "accuracy_ci_high", "f1_ci_low", "f1_ci_high"])
        for mode, result in comparisons.items():
            writer.writerow([mode, result["correct"], len(y), result["test"]["accuracy"], result["test"]["macro_f1"],
                             *result["accuracy_95"], *result["macro_f1_95"]])
    print(json.dumps({"paired": paired, "review": info["review"], "hog": comparisons["hog"]}, indent=2))


if __name__ == "__main__":
    main()
