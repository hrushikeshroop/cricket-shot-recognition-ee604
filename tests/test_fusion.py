"""Check the learned mixture's numerical objective and probability constraints."""

import numpy as np
import csv
import hashlib
import json
from pathlib import Path
from scipy.optimize import check_grad
from cricketshot.fusion import objective, mix, gate_features, fit_gate


def test_gate_gradient_and_probability_constraints():
    rng = np.random.default_rng(604)
    p = rng.dirichlet(np.ones(4), size=(12, 3))
    x = rng.normal(size=(12, 5))
    y = np.arange(12) % 4
    theta = rng.normal(scale=0.1, size=18)
    args = (x, p, y, np.ones(12), 0.1)
    error = check_grad(lambda t: objective(t, *args)[0], lambda t: objective(t, *args)[1], theta)
    assert error < 1e-6
    q, weights = mix(p, x, theta[:15].reshape(5, 3), theta[15:])
    np.testing.assert_allclose(q.sum(1), 1)
    np.testing.assert_allclose(weights.sum(1), 1)
    assert (q >= 0).all() and (weights >= 0).all()


def test_constant_control_learns_the_reliable_expert():
    y = np.arange(40) % 4
    good = np.full((40, 4), 0.05)
    good[np.arange(40), y] = 0.85
    bad = np.roll(good, 1, axis=1)
    p = np.stack([good, bad, np.full_like(good, 0.25)], axis=1)
    x = np.empty((40, 0))
    slopes, bias, info = fit_gate(x, p, y, 0)
    q, weights = mix(p, x, slopes, bias)
    assert info["converged"] and (weights[:, 0] > 0.99).all()
    np.testing.assert_array_equal(q.argmax(1), y)


def test_gate_inputs_agreement_and_uniform_scores():
    p = np.full((2, 3, 4), 0.25)
    embedding = np.ones((2, 2, 8))
    values = gate_features(p, embedding)
    assert values.shape == (2, 13)
    np.testing.assert_allclose(values[:, :3], 1)
    np.testing.assert_allclose(values[:, 3:6], 0)
    np.testing.assert_allclose(values[:, 9:], 0, atol=1e-7)


def test_recorded_folds_exclude_each_predicted_training_group():
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "data/images/processed/manifest.csv").open(encoding="utf-8", newline="")))
    expected = {row["id"]: int(row["group"].removeprefix("photo_")) for row in rows if row["split"] == "train"}
    folds = json.loads((root / "results/fusion/folds.json").read_text())
    with np.load(root / "results/fusion/oof_predictions.npz", allow_pickle=False) as data:
        assert set(data["ids"]) == set(expected)
        assert len(set(data["ids"])) == len(data["ids"])
        # sklearn preserves float32 feature precision in these expert outputs.
        np.testing.assert_allclose(data["probabilities"].sum(2), 1, rtol=0, atol=2e-7)
        for row in folds:
            fit, held = set(row["fit_groups"]), set(row["held_groups"])
            assert fit.isdisjoint(held) and fit | held == set(expected.values())
            selected = data["fold"] == row["fold"]
            assert set(data["groups"][selected]) == held
            assert all(expected[str(i)] == int(g) for i, g in zip(data["ids"][selected], data["groups"][selected]))


def test_frozen_selection_and_recorded_metrics():
    from sklearn.metrics import f1_score
    root = Path(__file__).resolve().parents[1]
    selection_path = root / "results/fusion/selection.json"
    selection = json.loads(selection_path.read_text())
    recorded = json.loads((root / "results/fusion/metrics.json").read_text())
    assert not selection["selection_uses_test"]
    assert recorded["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert selection["gate_sha256"] == hashlib.sha256((root / "models/fusion/gate.npz").read_bytes()).hexdigest()
    with np.load(root / "results/fusion/test_predictions.npz", allow_pickle=False) as data:
        for mode, metrics in recorded["test"].items():
            predicted = data["p_" + mode].argmax(1)
            assert abs((predicted == data["y"]).mean() - metrics["accuracy"]) < 1e-12
            assert abs(f1_score(data["y"], predicted, average="macro") - metrics["macro_f1"]) < 1e-12
