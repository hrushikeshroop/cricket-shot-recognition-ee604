"""Fit classifier heads, freeze validation choices and save measured test evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import sklearn
from PIL import Image, ImageEnhance, ImageFilter
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from threadpoolctl import threadpool_limits
from cricketshot.core import Encoder, decode_video, descriptor, softmax


def ece(y, p):
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def metrics(y, p, classes):
    pred = p.argmax(axis=1)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=np.arange(len(classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "nll": float(log_loss(y, p, labels=np.arange(len(classes)))),
        "ece_10_bins": ece(y, p),
        "brier_multiclass": float(
            np.mean(np.sum((p - np.eye(len(classes))[y]) ** 2, axis=1))
        ),
        "confusion": confusion_matrix(y, pred, labels=np.arange(len(classes))).tolist(),
        "per_class": classification_report(
            y,
            pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def source_bootstrap(y, p, groups, classes, repeats=2000):
    rng = np.random.default_rng(17)
    unique = np.unique(groups)
    values = []
    for _ in range(repeats):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
        values.append(
            [
                np.mean(p[idx].argmax(1) == y[idx]),
                f1_score(
                    y[idx],
                    p[idx].argmax(1),
                    labels=np.arange(len(classes)),
                    average="macro",
                    zero_division=0,
                ),
            ]
        )
    low, high = np.percentile(values, [2.5, 97.5], axis=0)
    return {
        "method": "Percentile cluster bootstrap over held-out source groups, 2000 draws, seed 17",
        "groups": len(unique),
        "accuracy_95": [float(low[0]), float(high[0])],
        "macro_f1_95": [float(low[1]), float(high[1])],
        "caution": "Few source groups. Intervals describe this test set and omit training variability.",
    }


def select_threshold(y, p):
    choices = []
    for threshold in np.arange(0.30, 0.951, 0.05):
        keep = p.max(axis=1) >= threshold
        if keep.sum() >= max(10, int(np.ceil(0.20 * len(y)))):
            acc = float(np.mean(p[keep].argmax(axis=1) == y[keep]))
            if acc >= 0.90:
                choices.append(
                    (float(threshold), float(keep.mean()), acc, int(keep.sum()))
                )
    if choices:
        th, coverage, acc, n = choices[0]
        return th, {
            "target_met": True,
            "validation_coverage": coverage,
            "validation_accuracy": acc,
            "validation_accepted": n,
        }
    th = 0.80
    keep = p.max(axis=1) >= th
    return th, {
        "target_met": False,
        "reason": "No candidate met 90% accuracy with at least 20% coverage and 10 clips. Fallback 0.80.",
        "validation_coverage": float(keep.mean()),
        "validation_accepted": int(keep.sum()),
        "validation_accuracy": (
            float(np.mean(p[keep].argmax(1) == y[keep])) if keep.any() else None
        ),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["full", "batter_roi"], default="batter_roi")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    cfg["view"] = args.view
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(exist_ok=True, parents=True)
    model_dir = ROOT / args.model_dir
    model_dir.mkdir(exist_ok=True, parents=True)
    data_path = ROOT / (
        "data/processed/features_fullframe.npz"
        if args.view == "full"
        and (ROOT / "data/processed/features_fullframe.npz").exists()
        else "data/processed/features.npz"
    )
    with np.load(data_path, allow_pickle=False) as a:
        features, colors = a["features"], a["color"]
        labels, split, groups, ids = a["labels"], a["split"], a["groups"], a["ids"]
    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(s) for s in labels])
    tr, va, te = [split == s for s in ("train", "val", "test")]
    test_y = y[te]
    results = {
        "classes": classes,
        "seed": 17,
        "configuration": cfg,
        "models": {},
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "feature_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    predictions = {}
    coefficients = {}
    with threadpool_limits(limits=2):
        for mode in ["color"] + cfg["descriptors"]:
            t = time.perf_counter()
            z = colors if mode == "color" else descriptor(features, mode)
            trials, heads = [], []
            for c in cfg["C_candidates"]:
                head = LogisticRegression(
                    C=c, solver="lbfgs", class_weight="balanced", max_iter=2000
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    head.fit(z[tr], y[tr])
                val_f1 = f1_score(
                    y[va],
                    head.predict(z[va]),
                    labels=np.arange(len(classes)),
                    average="macro",
                    zero_division=0,
                )
                trials.append(
                    {
                        "C": c,
                        "validation_macro_f1": float(val_f1),
                        "iterations": int(head.n_iter_.max()),
                        "warnings": [str(w.message) for w in caught],
                    }
                )
                heads.append(head)
            best = int(np.argmax([r["validation_macro_f1"] for r in trials]))
            head = heads[best]
            val_logits = head.decision_function(z[va])
            fit = minimize_scalar(
                lambda lt: log_loss(
                    y[va],
                    softmax(val_logits, np.exp(lt)),
                    labels=np.arange(len(classes)),
                ),
                bounds=np.log(cfg["temperature_bounds"]),
                method="bounded",
            )
            temp = float(np.exp(fit.x))
            val_p = softmax(val_logits, temp)
            test_logits = head.decision_function(z[te])
            uncalibrated, calibrated = softmax(test_logits), softmax(test_logits, temp)
            threshold, threshold_info = select_threshold(y[va], val_p)
            accepted = calibrated.max(1) >= threshold
            info = {
                "C": cfg["C_candidates"][best],
                "dimension": z.shape[1],
                "temperature": temp,
                "trials": trials,
                "validation": metrics(y[va], val_p, classes),
                "test": metrics(test_y, calibrated, classes),
                "test_uncalibrated": metrics(test_y, uncalibrated, classes),
                "review_threshold": threshold,
                "threshold_selection": threshold_info,
                "test_selective": {
                    "coverage": float(accepted.mean()),
                    "accepted": int(accepted.sum()),
                    "accuracy": (
                        float(
                            np.mean(calibrated[accepted].argmax(1) == test_y[accepted])
                        )
                        if accepted.any()
                        else None
                    ),
                },
                "fit_and_evaluate_seconds": time.perf_counter() - t,
            }
            if mode != "color":
                reversed_p = softmax(
                    descriptor(features[te, ::-1], mode) @ head.coef_.T
                    + head.intercept_,
                    temp,
                )
                info["temporal_reversal"] = metrics(test_y, reversed_p, classes)
                info["reversal_max_probability_change"] = float(
                    np.max(np.abs(reversed_p - calibrated))
                )
            results["models"][mode] = info
            predictions[mode] = calibrated
            coefficients[mode] = (head.coef_, head.intercept_)
            print(
                mode,
                "C",
                info["C"],
                "val F1",
                round(info["validation"]["macro_f1"], 4),
                "test accuracy",
                round(info["test"]["accuracy"], 4),
                "test F1",
                round(info["test"]["macro_f1"], 4),
                flush=True,
            )
    majority = np.bincount(y[tr]).argmax()
    results["majority_baseline"] = {
        "class": classes[majority],
        "accuracy": float(np.mean(test_y == majority)),
        "macro_f1": float(
            f1_score(
                test_y,
                np.full(len(test_y), majority),
                labels=np.arange(len(classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }
    mode = cfg["deployment_descriptor"]
    main_model = results["models"][mode]
    main_model["bootstrap"] = source_bootstrap(
        test_y, predictions[mode], groups[te], classes
    )
    coef, intercept = coefficients[mode]
    np.savez_compressed(model_dir / "classifier.npz", coef=coef, intercept=intercept)
    model_config = {
        "mode": mode,
        "view": cfg.get("view", "batter_roi"),
        "classes": classes,
        "temperature": main_model["temperature"],
        "review_threshold": main_model["review_threshold"],
        "C": main_model["C"],
        "frames": 9,
        "train_clips": int(tr.sum()),
        "validation_clips": int(va.sum()),
        "test_clips": int(te.sum()),
        "threshold_selection": main_model["threshold_selection"],
        "notes": "Review threshold is a validation heuristic, not a certified error bound or open-set detector.",
    }
    (model_dir / "classifier.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    # Corrupt test frames without fitting any parameter on the test set.
    rows = list(
        csv.DictReader(
            (ROOT / "data/processed/manifest.csv").open(encoding="utf-8", newline="")
        )
    )
    test_rows = [r for r in rows if r["split"] == "test"]
    encoder = Encoder(view=args.view)
    corrupt_features = {"dark": [], "blur": []}
    for i, row in enumerate(test_rows):
        frames, _ = decode_video(ROOT / row["path"])
        for name in corrupt_features:
            transformed = [
                (
                    np.asarray(ImageEnhance.Brightness(Image.fromarray(f)).enhance(0.6))
                    if name == "dark"
                    else np.asarray(
                        Image.fromarray(f).filter(ImageFilter.GaussianBlur(radius=2))
                    )
                )
                for f in frames
            ]
            corrupt_features[name].append(encoder.encode(transformed))
        if (i + 1) % 25 == 0:
            print(f"Robustness {i+1}/{len(test_rows)}", flush=True)
    results["robustness"] = {}
    for corruption, feats in corrupt_features.items():
        results["robustness"][corruption] = {}
        for name in ("middle", "mean", "phase_delta"):
            c, b = coefficients[name]
            pp = softmax(
                descriptor(np.stack(feats), name) @ c.T + b,
                results["models"][name]["temperature"],
            )
            results["robustness"][corruption][name] = metrics(test_y, pp, classes)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        y=test_y,
        ids=ids[te],
        groups=groups[te],
        **{f"p_{k}": v for k, v in predictions.items()},
    )
    with (result_dir / "test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "true", "predicted", "confidence", "accepted"]
            + [f"p_{c}" for c in classes]
        )
        for i, row in enumerate(test_rows):
            pp = predictions[mode][i]
            w.writerow(
                [
                    row["id"],
                    row["label"],
                    classes[pp.argmax()],
                    float(pp.max()),
                    bool(pp.max() >= main_model["review_threshold"]),
                ]
                + pp.tolist()
            )
    (result_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("Saved measured evidence to results/metrics.json", flush=True)


if __name__ == "__main__":
    main()
