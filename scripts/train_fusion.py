"""Validation-first learned fusion follow-up; keep original models unchanged.

Run --stage validation first. It freezes the selected gate before --stage test.
Only training groups produce the out-of-fold examples used to fit the gate.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cricketshot.core import softmax
from cricketshot.images import image_descriptor
from cricketshot.fusion import gate_features, fit_gate, mix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    cfg_path = ROOT / "configs/fusion_experiment.json"
    cfg = json.loads(cfg_path.read_text())
    out = ROOT / "results/fusion"
    out.mkdir(exist_ok=True)
    features_path = ROOT / "data/images/processed/features.npz"
    heads_path = ROOT / "models/image/heads.npz"
    binding = {"config_sha256": sha(cfg_path), "features_sha256": sha(features_path),
               "heads_sha256": sha(heads_path), "script_sha256": sha(Path(__file__)),
               "implementation_sha256": sha(ROOT / "src/cricketshot/fusion.py")}
    with np.load(features_path, allow_pickle=False) as data:
        features, labels, split, groups, ids = [data[k] for k in ("features", "labels", "split", "groups", "ids")]
    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(label) for label in labels])
    train, val, test = [np.flatnonzero(split == part) for part in ("train", "val", "test")]
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(groups[a]) & set(groups[b]), "Outer group leakage"
    with np.load(heads_path, allow_pickle=False) as data:
        heads = {key: data[key] for key in data.files}
    experts = cfg["experts"]
    # No validation-fitted temperatures enter the training out-of-fold scores.
    def expert_scores(indices):
        return np.stack([
            softmax(image_descriptor(features[indices], mode) @ heads[mode + "_coef"].T + heads[mode + "_intercept"])
            for mode in experts
        ], axis=1)

    if args.stage == "validation":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import f1_score, log_loss
        from threadpoolctl import threadpool_limits

        # Written before fitting. Refuse silently replacing a prior experiment.
        if (out / "selection.json").exists():
            raise RuntimeError("Selection already frozen; preserve it rather than overwrite")
        protocol = {"binding": binding, "config": cfg, "classes": classes,
                    "counts": {"train": len(train), "validation": len(val), "test": len(test)},
                    "test_previously_seen": True}
        write(out / "protocol.json", protocol)
        folds = StratifiedGroupKFold(n_splits=cfg["training_folds"], shuffle=True, random_state=cfg["seed"])
        oof = np.full((len(train), 3, 4), np.nan)
        fold_ids = np.full(len(train), -1, dtype=int)
        fold_audit = []
        with threadpool_limits(limits=2):
            for fold, (fit, held) in enumerate(folds.split(train, y[train], groups[train])):
                assert not set(groups[train[fit]]) & set(groups[train[held]])
                audit = {"fold": fold, "fit_groups": sorted(set(map(int, groups[train[fit]]))),
                         "held_groups": sorted(set(map(int, groups[train[held]]))), "warnings": []}
                for e, mode in enumerate(experts):
                    z = image_descriptor(features[train], mode)
                    head = LogisticRegression(C=cfg["expert_C"], class_weight="balanced", solver="lbfgs", max_iter=2000)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        head.fit(z[fit], y[train[fit]])
                    audit["warnings"].extend(str(w.message) for w in caught)
                    if caught:
                        raise RuntimeError("An expert fit emitted a warning; inspect before continuing")
                    oof[held, e] = head.predict_proba(z[held])
                fold_ids[held] = fold
                fold_audit.append(audit)
                print(f"Training fold {fold + 1}/5: {len(held)} held-out images", flush=True)
            assert np.isfinite(oof).all() and (fold_ids >= 0).all()
            raw = gate_features(oof, features[train])
            mean, scale = raw.mean(0), np.maximum(raw.std(0), 0.05)
            x = (raw - mean) / scale
            vp = expert_scores(val)
            vx = (gate_features(vp, features[val]) - mean) / scale
            constant_slopes, constant_bias, constant_fit = fit_gate(np.empty((len(train), 0)), oof, y[train], 0)
            constant, constant_weights = mix(vp, np.empty((len(val), 0)), constant_slopes, constant_bias)
            def scores(p):
                return {"macro_f1": float(f1_score(y[val], p.argmax(1), average="macro")),
                        "accuracy": float((p.argmax(1) == y[val]).mean()),
                        "nll": float(log_loss(y[val], p, labels=np.arange(4)))}
            trials, gates = [], []
            for l2 in cfg["gate_l2_candidates"]:
                slopes, bias, fit_info = fit_gate(x, oof, y[train], l2)
                p, _ = mix(vp, vx, slopes, bias)
                trials.append({"l2": l2, **scores(p), **fit_info})
                gates.append((slopes, bias))
            best = max(range(len(trials)), key=lambda k: (trials[k]["macro_f1"], trials[k]["l2"]))
        old = json.loads((ROOT / "results/images/metrics.json").read_text())
        controls = {"uniform_mean": scores(vp.mean(1)), "constant_weights": scores(constant)}
        strongest = max([m["validation"]["macro_f1"] for m in old["models"].values()] + [m["macro_f1"] for m in controls.values()])
        selected = {"binding": binding, "selected_l2": trials[best]["l2"], "trials": trials,
                    "controls": controls, "constant_weights": constant_weights[0].tolist(),
                    "constant_fit": constant_fit, "selected_validation": trials[best],
                    "best_reference_validation_f1": strongest,
                    "deploy_adaptive": trials[best]["macro_f1"] > strongest,
                    "selection_uses_test": False}
        np.savez_compressed(out / "oof_predictions.npz", probabilities=oof, y=y[train], ids=ids[train], groups=groups[train], fold=fold_ids)
        write(out / "folds.json", fold_audit)
        model_dir = ROOT / "models/fusion"
        model_dir.mkdir(exist_ok=True)
        slopes, bias = gates[best]
        np.savez_compressed(model_dir / "gate.npz", mean=mean, scale=scale, slopes=slopes, intercept=bias,
                            constant_intercept=constant_bias)
        selected["gate_sha256"] = sha(model_dir / "gate.npz")
        write(out / "selection.json", selected)
        print(json.dumps(selected, indent=2), flush=True)
    else:
        from train_evaluate import metrics
        selected = json.loads((out / "selection.json").read_text())
        assert selected["binding"] == binding, "Frozen selection inputs changed"
        gate_path = ROOT / "models/fusion/gate.npz"
        assert sha(gate_path) == selected["gate_sha256"], "Selected gate changed"
        if (out / "metrics.json").exists():
            raise RuntimeError("Test already evaluated; preserve the recorded result")
        with np.load(gate_path, allow_pickle=False) as data:
            gate = {key: data[key] for key in data.files}
        p = expert_scores(test)
        x = (gate_features(p, features[test]) - gate["mean"]) / gate["scale"]
        adaptive, weights = mix(p, x, gate["slopes"], gate["intercept"])
        constant, _ = mix(p, np.empty((len(test), 0)), np.empty((0, 3)), gate["constant_intercept"])
        predictions = {"uniform_mean": p.mean(1), "constant_weights": constant, "adaptive": adaptive}
        # Paired visual-group bootstrap, fixed 2000 draws; descriptive uncertainty.
        with np.load(ROOT / "results/images/test_predictions.npz", allow_pickle=False) as old:
            assert np.array_equal(ids[test], old["ids"]) and np.array_equal(y[test], old["y"])
            reference = {mode: old["p_" + mode] for mode in ("concat", "contrast")}
        correct = {name: values.argmax(1) == y[test] for name, values in {**reference, **predictions}.items()}
        unique = np.unique(groups[test])
        cluster_n = np.array([(groups[test] == g).sum() for g in unique])
        rng = np.random.default_rng(17)
        draws = rng.integers(len(unique), size=(2000, len(unique)))
        paired = {}
        for name in ("concat", "contrast", "uniform_mean", "constant_weights"):
            delta = correct["adaptive"].astype(int) - correct[name].astype(int)
            cluster_delta = np.array([delta[groups[test] == g].sum() for g in unique])
            boot = cluster_delta[draws].sum(1) / cluster_n[draws].sum(1)
            paired[name] = {"accuracy_difference": float(delta.mean()), "accuracy_difference_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                            "gained": int((delta == 1).sum()), "lost": int((delta == -1).sum())}
        evidence = {"classes": classes, "selection_sha256": sha(out / "selection.json"), "binding": binding,
                    "test": {name: metrics(y[test], values, classes) for name, values in predictions.items()},
                    "paired_adaptive_minus": paired, "mean_test_weights": weights.mean(0).tolist(),
                    "caution": cfg["caution"] + " Bootstrap intervals condition on fitted models and this split; visual groups are not match identities."}
        np.savez_compressed(out / "test_predictions.npz", y=y[test], ids=ids[test], groups=groups[test], weights=weights,
                            **{"p_" + name: values for name, values in predictions.items()})
        write(out / "metrics.json", evidence)
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
