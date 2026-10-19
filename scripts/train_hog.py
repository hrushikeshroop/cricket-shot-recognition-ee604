"""One exploratory HOG comparison on the unchanged Option B image split."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import hashlib
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from cricketshot.hog import PARAMETERS, hog_descriptor
from cricketshot.images import load_image
from cricketshot.core import softmax
from train_evaluate import metrics


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    out = ROOT / "results/hog"
    out.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "models/hog"
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/hog_experiment.json").read_text())
    manifest = ROOT / "data/images/processed/manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    metadata = {"parameters": PARAMETERS, "manifest_sha256": file_hash(manifest),
                "implementation_sha256": file_hash(ROOT / "src/cricketshot/hog.py"),
                "protocol_sha256": file_hash(ROOT / "configs/hog_experiment.json")}
    # Preserve the intended protocol before extracting features or fitting heads.
    (out / "protocol_before_fit.json").write_text(json.dumps({**config, **metadata}, indent=2))
    cache = ROOT / "data/images/processed/hog_features.npz"
    cache_meta = cache.with_suffix(".json")
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()) == metadata:
        with np.load(cache, allow_pickle=False) as arrays:
            x = arrays["features"].copy()
            assert arrays["ids"].tolist() == [r["id"] for r in rows]
    else:
        features = []
        for i, row in enumerate(rows, 1):
            features.append(hog_descriptor(load_image(ROOT / row["path"])))
            if i % 500 == 0:
                print(f"HOG features {i}/{len(rows)}", flush=True)
        x = np.stack(features)
        np.savez_compressed(cache, features=x, ids=np.array([r["id"] for r in rows]))
        cache_meta.write_text(json.dumps(metadata, indent=2))
    classes = sorted({r["label"] for r in rows})
    y = np.array([classes.index(r["label"]) for r in rows])
    masks = {s: np.array([r["split"] == s for r in rows]) for s in ("train", "val", "test")}
    trials = []
    heads = []
    with threadpool_limits(limits=2):
        for c in config["C_candidates"]:
            head = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=2000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                head.fit(x[masks["train"]], y[masks["train"]])
            score = f1_score(y[masks["val"]], head.predict(x[masks["val"]]), average="macro")
            trials.append({"C": c, "validation_macro_f1": float(score),
                           "iterations": int(head.n_iter_.max()),
                           "warnings": [str(w.message) for w in caught]})
            heads.append(head)
            print(f"HOG C={c}, validation macro F1={score:.6f}", flush=True)
        best = int(np.argmax([t["validation_macro_f1"] for t in trials]))
        head = heads[best]
        # Test is scored only after the validation choice is frozen.
        (out / "selection_before_test.json").write_text(json.dumps({"trials": trials, "selected_C": trials[best]["C"]}, indent=2))
        probabilities = {s: softmax(head.decision_function(x[masks[s]])) for s in ("val", "test")}
    np.savez_compressed(folder / "head.npz", coef=head.coef_, intercept=head.intercept_)
    model_config = {"classes": classes, "parameters": PARAMETERS, "C": trials[best]["C"],
                    "metadata": metadata, "temperature": 1.0, "exploratory": True}
    (folder / "classifier.json").write_text(json.dumps(model_config, indent=2))
    result = {"classes": classes, "config": config, "metadata": metadata, "trials": trials,
              "C": trials[best]["C"], "dimension": x.shape[1],
              "validation": metrics(y[masks["val"]], probabilities["val"], classes),
              "test": metrics(y[masks["test"]], probabilities["test"], classes),
              "elapsed_seconds": time.perf_counter() - start}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    test_rows = [r for r in rows if r["split"] == "test"]
    p = probabilities["test"]
    np.savez_compressed(out / "test_predictions.npz", y=y[masks["test"]], p=p,
                        ids=np.array([r["id"] for r in test_rows]),
                        groups=np.array([r["group"] for r in test_rows]))
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "group", "true", "predicted"] + ["p_" + c for c in classes])
        for row, probability in zip(test_rows, p):
            writer.writerow([row["id"], row["group"], row["label"], classes[probability.argmax()]] + probability.tolist())
    print(json.dumps({"C": result["C"], "accuracy": result["test"]["accuracy"],
                      "macro_f1": result["test"]["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
