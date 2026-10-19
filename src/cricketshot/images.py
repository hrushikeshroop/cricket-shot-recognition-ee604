"""Four-class still-image recognition with dual-view contrast and review gating."""

from pathlib import Path
import json
import time
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .core import Encoder, normalize, softmax, ROOT

LABELS = {
    "drive": "Drive",
    "legglance-flick": "Leg glance / flick",
    "pullshot": "Pull",
    "sweep": "Sweep",
}


def load_image(path):
    try:
        with Image.open(path) as source:
            if source.width * source.height > 24_000_000:
                raise ValueError("Use an image below 24 megapixels.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "Cannot read this image. Use a valid JPEG, PNG or WebP file."
        ) from exc
    if min(image.size) < 32:
        raise ValueError("Image is too small. Use at least 32 pixels on each side.")
    return image


def image_views(image):
    """Two fixed views. Central crop is a context test, not an actor detector."""
    image = image.copy()
    image.thumbnail((1024, 1024))
    w, h = image.size
    focus = image.crop(
        (round(0.15 * w), round(0.05 * h), round(0.85 * w), round(0.95 * h))
    )
    return image, focus


def image_descriptor(features, mode):
    """Compare the two frozen 1,280-value embeddings without training the CNN.

    Normalize each view first so its vector length cannot dominate fusion.
    The absolute difference adds a non-linear relationship to a linear head.
    It is a regional-view comparison, not a keypoint or bat-motion descriptor.
    """
    x = normalize(np.asarray(features, np.float32))
    if mode == "global":
        z = x[..., 0, :]
    elif mode == "focus":
        z = x[..., 1, :]
    elif mode == "concat":
        z = np.concatenate([x[..., 0, :], x[..., 1, :]], axis=-1)
    elif mode == "contrast":
        z = np.concatenate(
            [x[..., 0, :], x[..., 1, :], np.abs(x[..., 0, :] - x[..., 1, :])], axis=-1
        )
    else:
        raise ValueError("Unknown image descriptor")
    return normalize(z).astype(np.float32)


class ImageRecognizer:
    def __init__(self, model_dir=None):
        self.dir = Path(model_dir or ROOT / "models/image")
        self.config = json.loads(
            (self.dir / "classifier.json").read_text(encoding="utf-8")
        )
        with np.load(self.dir / "heads.npz", allow_pickle=False) as a:
            self.heads = {k: a[k].copy() for k in a.files}
        self.encoder = Encoder(ROOT / "models/mobilenetv2_features.onnx", view="full")

    def predict_features(self, features, mode="contrast"):
        """Apply learned cricket weights, then the unchanged validation temperature."""
        z = image_descriptor(features, mode)
        return softmax(
            z @ self.heads[mode + "_coef"].T + self.heads[mode + "_intercept"],
            self.config["temperatures"][mode],
        )

    def predict(self, path):
        t = time.perf_counter()
        image = load_image(path)
        views = image_views(image)
        features = self.encoder.encode([np.asarray(v) for v in views])
        p = self.predict_features(features)
        pg, pf = [self.predict_features(features, m) for m in ("global", "focus")]
        agree = bool(pg.argmax() == pf.argmax())
        consistent = bool(agree and pg.argmax() == p.argmax())
        ci = int(p.argmax())
        threshold = self.config["review_threshold"]
        accepted = bool(p.max() >= threshold and consistent)
        return {
            "label": self.config["classes"][ci],
            "display_label": LABELS[self.config["classes"][ci]],
            "score_type": "temperature-scaled softmax",
            "score_note": "A relative score among four supported labels, not a guarantee of correctness. Test calibration worsened after temperature scaling.",
            "confidence": float(p.max()),
            "accepted": accepted,
            "decision": "Prediction" if accepted else "Needs review",
            "probabilities": dict(zip(self.config["classes"], map(float, p))),
            "views_agree": agree,
            "prediction_consistent": consistent,
            "global_label": LABELS[self.config["classes"][int(pg.argmax())]],
            "focus_label": LABELS[self.config["classes"][int(pf.argmax())]],
            "threshold": threshold,
            "seconds": time.perf_counter() - t,
            "input_size": list(image.size),
            "scope": "Four-class still-image classification. Unknown classes can receive confident scores.",
        }, views
