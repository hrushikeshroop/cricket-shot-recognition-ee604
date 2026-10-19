"""Shared decoding, feature extraction, temporal descriptors and safe inference.

Training and the demo import the same functions to avoid preprocessing drift.
The saved classifier is numeric NPZ/JSON, so inference never unpickles objects.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
N_FRAMES = 9
DISPLAY = {
    "cut": "Cut",
    "defence": "Defence",
    "glance": "Glance",
    "offdrive": "Off drive",
    "ondrive": "On drive",
    "pull": "Pull",
    "sweep": "Sweep",
    "unorthodox": "Unorthodox",
}


def decode_video(path, max_frames=1500):
    """Decode a trimmed clip in order, then sample nine evenly spaced frames.

    Sequential decoding avoids inaccurate frame seeking on short AVI files.
    Bounds protect the local demo from accidentally processing full matches.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open this video. Use an AVI, MP4 or MOV clip.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if len(frames) >= max_frames:
                raise ValueError(
                    "Clip is too long. Please trim it to at most 30 seconds."
                )
            if bgr.shape[0] * bgr.shape[1] > 3840 * 2160:
                raise ValueError("Video exceeds the 4K input limit.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Retain bounded previews rather than full-resolution video frames.
            image = Image.fromarray(rgb)
            image.thumbnail((640, 480))
            frames.append(np.asarray(image))
    finally:
        cap.release()
    if not frames:
        raise ValueError("The file contains no decodable video frames.")
    duration = len(frames) / fps if fps > 0 else None
    if duration and duration > 30:
        raise ValueError("Clip is too long. Please trim it to at most 30 seconds.")
    indices = np.rint(np.linspace(0, len(frames) - 1, N_FRAMES)).astype(int)
    return [frames[i] for i in indices], {
        "decoded_frames": len(frames),
        "fps": fps,
        "duration_seconds": duration,
        "sample_indices": indices.tolist(),
    }


def crop_frame(rgb, view="batter_roi"):
    """Fixed broadcast-layout crop, not a learned person detector."""
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    if view == "batter_roi":
        w, h = im.size
        im = im.crop(
            (round(0.30 * w), round(0.02 * h), round(0.70 * w), round(0.65 * h))
        )
    elif view != "full":
        raise ValueError("Unknown encoder view")
    return im


def prepare_frame(rgb, view="batter_roi"):
    """Resize with preserved aspect ratio, center crop and ImageNet normalize."""
    im = crop_frame(rgb, view)
    im = ImageOps.fit(im, (224, 224), method=Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def normalize(x):
    """Per-vector L2 normalization; this learns no statistics from any split."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, model=None, view="batter_roi"):
        self.view = view
        model = Path(model or ROOT / "models/mobilenetv2_features.onnx")
        if not model.exists():
            raise FileNotFoundError(
                "Missing visual encoder. Run python scripts/prepare_data.py first."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, frames):
        """Run frozen ImageNet parameters; no gradient or weight update occurs."""
        embeddings = []
        for frame in frames:
            tensor = prepare_frame(frame, self.view)  # (1, 3, 224, 224)
            output = self.session.run(None, {self.input_name: tensor})[0]
            embeddings.append(output.reshape(-1))  # pooled 1,280-value vector
        return np.stack(embeddings).astype(np.float32)


def descriptor(features, mode):
    """Order-sensitive compact representation. Accepts (..., 9, 1280)."""
    x = normalize(np.asarray(features, np.float32))
    if mode == "middle":
        z = x[..., 4, :]
    elif mode == "mean":
        z = x.mean(axis=-2)
    elif mode in ("phase", "phase_delta"):
        early, middle, late = [x[..., i : i + 3, :].mean(axis=-2) for i in (0, 3, 6)]
        parts = [early, middle, late]
        if mode == "phase_delta":
            parts += [late - early, x.std(axis=-2)]
        z = np.concatenate(parts, axis=-1)
    else:
        raise ValueError(f"Unknown descriptor: {mode}")
    return normalize(z).astype(np.float32)


def color_descriptor(frames):
    """A deliberately shallow baseline for background and color shortcuts."""
    hs = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
        hs.append(hist / max(hist.sum(), 1))
    return normalize(np.mean(hs, axis=0)).astype(np.float32)


def softmax(logits, temperature=1.0):
    """Convert logits to relative class scores, stably by subtracting the max."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


class Recognizer:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or ROOT / "models")
        self.config = json.loads(
            (self.model_dir / "classifier.json").read_text(encoding="utf-8")
        )
        with np.load(self.model_dir / "classifier.npz", allow_pickle=False) as a:
            self.coef, self.intercept = a["coef"].copy(), a["intercept"].copy()
        self.encoder = Encoder(
            self.model_dir / "mobilenetv2_features.onnx",
            view=self.config.get("view", "batter_roi"),
        )

    def predict_features(self, features):
        z = descriptor(features, self.config["mode"])
        logits = z @ self.coef.T + self.intercept
        p = softmax(logits, self.config["temperature"])
        return p

    def predict(self, path):
        t = time.perf_counter()
        frames, meta = decode_video(path)
        features = self.encoder.encode(frames)
        p = self.predict_features(features)
        i = int(p.argmax())
        # This is a confidence-based review rule, not proof that an input is cricket.
        accepted = bool(p[i] >= self.config["review_threshold"])
        return {
            "label": self.config["classes"][i],
            "display_label": DISPLAY.get(
                self.config["classes"][i], self.config["classes"][i]
            ),
            "confidence": float(p[i]),
            "accepted": accepted,
            "decision": "Prediction" if accepted else "Needs review",
            "probabilities": {c: float(v) for c, v in zip(self.config["classes"], p)},
            "threshold": self.config["review_threshold"],
            "mode": self.config["mode"],
            "seconds": time.perf_counter() - t,
            "video": meta,
        }, frames
