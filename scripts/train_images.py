"""Fit four image heads under the documented, revised visual-grouping protocol."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import json
import time
import warnings
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits
from cricketshot.core import Encoder, softmax
from cricketshot.images import load_image, image_views, image_descriptor
from train_evaluate import metrics, source_bootstrap


def main():
    cfg = json.loads((ROOT / "configs/image_experiment.json").read_text())
    out = ROOT / "results/images"
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models/image"
    model_dir.mkdir(exist_ok=True)
    with np.load(ROOT / "data/images/processed/features.npz", allow_pickle=False) as a:
        features, labels, split, groups, ids = [
            a[k] for k in ("features", "labels", "split", "groups", "ids")
        ]
    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(c) for c in labels])
    tr, va, te = [split == s for s in ("train", "val", "test")]
    for s in ("train", "val", "test"):
        if set(labels[split == s]) != set(classes):
            raise ValueError("Class absent from " + s)
    results = {
        "classes": classes,
        "config": cfg,
        "models": {},
        "n": {s: int((split == s).sum()) for s in ("train", "val", "test")},
    }
    heads = {}
    preds = {}
    val_predictions = {}
    temperatures = {}
    with threadpool_limits(limits=2):
        for mode in cfg["modes"]:
            z = image_descriptor(features, mode)
            trials = []
            fitted = []
            for c in cfg["C_candidates"]:
                head = LogisticRegression(
                    C=c, class_weight="balanced", solver="lbfgs", max_iter=2000
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    head.fit(z[tr], y[tr])
                score = f1_score(y[va], head.predict(z[va]), average="macro")
                trials.append(
                    {
                        "C": c,
                        "validation_macro_f1": float(score),
                        "iterations": int(head.n_iter_.max()),
                        "warnings": [str(w.message) for w in caught],
                    }
                )
                fitted.append(head)
            best = int(np.argmax([t["validation_macro_f1"] for t in trials]))
            head = fitted[best]
            vl = head.decision_function(z[va])
            tl = head.decision_function(z[te])
            fit = minimize_scalar(
                lambda t: log_loss(y[va], softmax(vl, np.exp(t)), labels=np.arange(4)),
                bounds=np.log([0.1, 10]),
                method="bounded",
            )
            temp = float(np.exp(fit.x))
            p = softmax(tl, temp)
            v = softmax(vl, temp)
            heads[mode + "_coef"] = head.coef_
            heads[mode + "_intercept"] = head.intercept_
            preds[mode] = p
            val_predictions[mode] = v
            temperatures[mode] = temp
            results["models"][mode] = {
                "C": cfg["C_candidates"][best],
                "dimension": z.shape[1],
                "temperature": temp,
                "trials": trials,
                "validation": metrics(y[va], v, classes),
                "test": metrics(y[te], p, classes),
                "test_uncalibrated": metrics(y[te], softmax(tl), classes),
            }
            print(
                mode,
                "val F1",
                round(results["models"][mode]["validation"]["macro_f1"], 4),
                "test accuracy",
                round(results["models"][mode]["test"]["accuracy"], 4),
                "test F1",
                round(results["models"][mode]["test"]["macro_f1"], 4),
                flush=True,
            )
    vp = val_predictions["contrast"]
    agree_val = (
        val_predictions["global"].argmax(1) == val_predictions["focus"].argmax(1)
    ) & (val_predictions["global"].argmax(1) == vp.argmax(1))
    options = []
    for threshold in np.arange(0.5, 0.951, 0.05):
        keep = (vp.max(1) >= threshold) & agree_val
        if (
            keep.sum() >= max(20, np.ceil(0.2 * len(vp)))
            and np.mean(vp[keep].argmax(1) == y[va][keep]) >= 0.9
        ):
            options.append(float(threshold))
    threshold = options[0] if options else 0.8
    p = preds["contrast"]
    agreement = (preds["global"].argmax(1) == preds["focus"].argmax(1)) & (
        preds["global"].argmax(1) == p.argmax(1)
    )
    keep = (p.max(1) >= threshold) & agreement
    conf_only = p.max(1) >= threshold
    results["review"] = {
        "threshold": threshold,
        "target_met": bool(options),
        "validation_accepted": int(((vp.max(1) >= threshold) & agree_val).sum()),
        "test_accepted": int(keep.sum()),
        "test_coverage": float(keep.mean()),
        "test_accuracy": (
            float(np.mean(p[keep].argmax(1) == y[te][keep])) if keep.any() else None
        ),
        "agreement_fraction": float(agreement.mean()),
        "confidence_only_coverage": float(conf_only.mean()),
        "confidence_only_accuracy": (
            float(np.mean(p[conf_only].argmax(1) == y[te][conf_only]))
            if conf_only.any()
            else None
        ),
    }
    results["models"]["contrast"]["bootstrap"] = source_bootstrap(
        y[te], p, groups[te], classes
    )
    results["models"]["contrast"]["bootstrap"][
        "method"
    ] = "Percentile cluster bootstrap over visual duplicate groups, 2000 draws, seed 17"
    results["models"]["contrast"]["bootstrap"][
        "caution"
    ] = "Visual groups are not matches. Intervals omit training/split variability and may be optimistic if related frames remain across splits."
    majority = np.bincount(y[tr]).argmax()
    results["majority"] = {
        "class": classes[majority],
        "accuracy": float(np.mean(y[te] == majority)),
        "macro_f1": float(
            f1_score(
                y[te],
                np.full(te.sum(), majority),
                labels=np.arange(4),
                average="macro",
                zero_division=0,
            )
        ),
    }
    np.savez_compressed(model_dir / "heads.npz", **heads)
    config = {
        "classes": classes,
        "mode": "contrast",
        "temperatures": temperatures,
        "review_threshold": threshold,
        "review_rule": "confidence >= threshold AND global/focus/fused head top labels all agree",
        "review_target_met": bool(options),
        "crop_bounds": [0.15, 0.05, 0.85, 0.95],
        "note": "No guarantee of unknown-class rejection or match-independent accuracy.",
    }
    (model_dir / "classifier.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out / "test_predictions.npz",
        y=y[te],
        ids=ids[te],
        groups=groups[te],
        agreement=agreement,
        accepted=keep,
        **{"p_" + k: v for k, v in preds.items()}
    )
    rows = list(
        csv.DictReader(
            (ROOT / "data/images/processed/manifest.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    test_rows = [r for r in rows if r["split"] == "test"]
    encoder = Encoder(view="full")
    stress = {"dark": [], "blur": [], "flip": []}
    for i, row in enumerate(test_rows):
        im = load_image(ROOT / row["path"])
        for name in stress:
            changed = (
                ImageEnhance.Brightness(im).enhance(0.6)
                if name == "dark"
                else (
                    im.filter(ImageFilter.GaussianBlur(2))
                    if name == "blur"
                    else ImageOps.mirror(im)
                )
            )
            stress[name].append(
                encoder.encode([np.asarray(v) for v in image_views(changed)])
            )
        if (i + 1) % 200 == 0:
            print("Image stress tests", i + 1, "/", len(test_rows), flush=True)
    results["stress"] = {}
    for condition, feats in stress.items():
        results["stress"][condition] = {}
        for mode in ("global", "contrast"):
            pp = softmax(
                image_descriptor(np.stack(feats), mode) @ heads[mode + "_coef"].T
                + heads[mode + "_intercept"],
                temperatures[mode],
            )
            results["stress"][condition][mode] = metrics(y[te], pp, classes)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "true", "predicted", "confidence", "views_agree", "accepted"]
            + ["p_" + c for c in classes]
        )
        for i, row in enumerate(test_rows):
            writer.writerow(
                [
                    row["id"],
                    row["label"],
                    classes[p[i].argmax()],
                    float(p[i].max()),
                    bool(agreement[i]),
                    bool(keep[i]),
                ]
                + p[i].tolist()
            )
    print("Image evidence saved", flush=True)


if __name__ == "__main__":
    main()
