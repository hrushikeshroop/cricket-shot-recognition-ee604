"""Small, inspectable mixture of frozen view classifiers (no CNN training)."""

import numpy as np
from .core import normalize, softmax


def gate_features(probabilities, embeddings):
    """Scores: (images, experts, classes); embeddings: (images, 2, 1280)."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (3, 4):
        raise ValueError("Expected three four-class experts")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(2), 1):
        raise ValueError("Expert probabilities must be finite and sum to one")
    x = normalize(np.asarray(embeddings, dtype=np.float64))
    if x.ndim != 3 or x.shape[:2] != (len(p), 2):
        raise ValueError("Expected two embeddings per image")
    entropy = -(p * np.log(np.clip(p, 1e-15, 1))).sum(2) / np.log(4)
    ordered = np.sort(p, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    maximum = ordered[:, :, -1]
    distance = np.column_stack([
        0.5 * np.abs(p[:, a] - p[:, b]).sum(1)
        for a, b in ((0, 1), (0, 2), (1, 2))
    ])
    cosine = (1 - (x[:, 0] * x[:, 1]).sum(1))[:, None]
    return np.concatenate([entropy, margin, maximum, distance, cosine], axis=1)


def mix(probabilities, features, slopes, intercept):
    """A convex mixture: weights and output probabilities both sum to one."""
    weights = softmax(np.asarray(features) @ slopes + intercept)
    return (weights[:, :, None] * probabilities).sum(1), weights


def objective(theta, features, probabilities, labels, sample_weights, l2):
    """Balanced NLL and its analytic gradient; shared by fitting and tests."""
    n, dimensions = features.shape
    experts = probabilities.shape[1]
    slopes = theta[:dimensions * experts].reshape(dimensions, experts)
    intercept = theta[dimensions * experts:]
    q, weights = mix(probabilities, features, slopes, intercept)
    true_expert = probabilities[np.arange(n), :, labels]
    true_q = np.clip(q[np.arange(n), labels], 1e-15, 1)
    loss = -(sample_weights * np.log(true_q)).mean() + 0.5 * l2 * (slopes ** 2).sum()
    dz = sample_weights[:, None] * weights * (1 - true_expert / true_q[:, None]) / n
    gradient = np.concatenate([(features.T @ dz + l2 * slopes).ravel(), dz.sum(0)])
    return float(loss), gradient


def fit_gate(features, probabilities, labels, l2):
    """Training only. Constant-weight control passes an (N, 0) feature matrix."""
    from scipy.optimize import minimize

    counts = np.bincount(labels, minlength=probabilities.shape[2])
    if (counts == 0).any():
        raise ValueError("Every class must be present in gate training")
    sample_weights = len(labels) / (len(counts) * counts[labels])
    initial = np.zeros((features.shape[1] + 1) * probabilities.shape[1])
    fitted = minimize(
        objective, initial, args=(features, probabilities, labels, sample_weights, l2),
        jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError("Gate fit failed: " + str(fitted.message))
    dimensions, experts = features.shape[1], probabilities.shape[1]
    return (
        fitted.x[:dimensions * experts].reshape(dimensions, experts),
        fitted.x[dimensions * experts:],
        {"iterations": int(fitted.nit), "objective": float(fitted.fun), "converged": True},
    )
