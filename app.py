"""Local-only Flask demo. No uploads or predictions leave this computer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import base64
import csv
import io
import json
import tempfile
import threading
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from cricketshot.core import Recognizer, DISPLAY, crop_frame
from cricketshot.images import ImageRecognizer
from cricketshot.hog import HOGRecognizer


def create_app(recognizer=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    inference_lock = threading.Lock()
    model_holder = [recognizer]
    image_model_holder = [None]
    hog_model_holder = [None]

    def model():
        if model_holder[0] is None:
            model_holder[0] = Recognizer()
        return model_holder[0]

    def examples():
        path = ROOT / "demo/examples.json"
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return [e for e in entries if (ROOT / e["path"]).is_file() and (ROOT / e["thumbnail"]).is_file()]

    def infer(path):
        with inference_lock:
            prediction, frames = model().predict(path)
        previews = []
        for frame in frames:
            im = ImageOps.fit(
                crop_frame(frame, model().config.get("view", "batter_roi")),
                (224, 224),
                method=Image.Resampling.BILINEAR,
            )
            im.thumbnail((240, 180))
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=85)
            previews.append(
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
        prediction["previews"] = previews
        return prediction

    @app.get("/")
    def index():
        return render_template("images.html")

    @app.context_processor
    def project_identity():
        return {"submission": json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))}

    @app.get("/video")
    def video_index():
        return render_template("index.html", classes=DISPLAY)

    def image_examples():
        p = ROOT / "demo/image_examples.json"
        entries = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        return [e for e in entries if (ROOT / e["path"]).is_file()]

    def infer_image(path):
        with inference_lock:
            if image_model_holder[0] is None:
                image_model_holder[0] = ImageRecognizer()
            result, views = image_model_holder[0].predict(path)
            if hog_model_holder[0] is None:
                hog_model_holder[0] = HOGRecognizer()
            result["hog_baseline"] = hog_model_holder[0].predict(path)
        result["previews"] = []
        for view in views:
            im = ImageOps.fit(view, (224, 224), method=Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            result["previews"].append(
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
        return result

    @app.get("/image-api/examples")
    def image_example_list():
        return jsonify(
            [{k: v for k, v in e.items() if k != "path"} for e in image_examples()]
        )

    @app.get("/image-api/example/<example_id>/thumbnail")
    def image_thumbnail(example_id):
        entry = next((e for e in image_examples() if e["id"] == example_id), None)
        if not entry:
            return jsonify(error="Example not found"), 404
        return send_file(ROOT / entry["path"])

    @app.post("/image-api/example/<example_id>")
    def predict_image_example(example_id):
        entry = next((e for e in image_examples() if e["id"] == example_id), None)
        if not entry:
            return jsonify(error="Example not found"), 404
        result = infer_image(ROOT / entry["path"])
        result["example"] = {k: v for k, v in entry.items() if k != "path"}
        return jsonify(result)

    @app.post("/image-api/predict")
    def predict_image_upload():
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            return jsonify(error="Choose an image first."), 400
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return jsonify(error="Use JPEG, PNG, WebP or BMP."), 400
        with tempfile.TemporaryDirectory(prefix="cricketshot_image_") as folder:
            path = Path(folder) / ("image" + suffix)
            uploaded.save(path)
            return jsonify(infer_image(path))

    @app.get("/image-api/evidence")
    def image_evidence():
        return jsonify(
            json.loads(
                (ROOT / "results/images/metrics.json").read_text(encoding="utf-8")
            )
        )

    @app.get("/api/status")
    def status():
        ready = all(
            (ROOT / "models" / f).exists()
            for f in ("classifier.json", "classifier.npz", "mobilenetv2_features.onnx")
        )
        image_ready = all(
            (ROOT / f).exists()
            for f in (
                "models/image/classifier.json",
                "models/image/heads.npz",
                "models/hog/head.npz",
                "models/mobilenetv2_features.onnx",
            )
        )
        return jsonify(
            ready=image_ready,
            video_pilot_ready=ready,
            version="2.0.0",
            scope="One image, four dataset classes; video is an exploratory backup",
        )

    @app.get("/api/evidence")
    def evidence():
        return jsonify(
            json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
        )

    @app.get("/api/examples")
    def example_list():
        return jsonify(
            [{k: v for k, v in e.items() if k != "path"} for e in examples()]
        )

    @app.get("/api/example/<example_id>/thumbnail")
    def thumbnail(example_id):
        entry = next((e for e in examples() if e["id"] == example_id), None)
        if not entry:
            return jsonify(error="Example not found"), 404
        return send_file(ROOT / entry["thumbnail"], mimetype="image/jpeg")

    @app.post("/api/example/<example_id>")
    def predict_example(example_id):
        entry = next((e for e in examples() if e["id"] == example_id), None)
        if not entry:
            return jsonify(error="Example not found"), 404
        result = infer(ROOT / entry["path"])
        result["example"] = {k: v for k, v in entry.items() if k != "path"}
        return jsonify(result)

    @app.post("/api/predict")
    def predict_upload():
        uploaded = request.files.get("video")
        if not uploaded or not uploaded.filename:
            return jsonify(error="Choose a trimmed video first."), 400
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in {".avi", ".mp4", ".mov", ".mkv", ".webm"}:
            return jsonify(error="Use an AVI, MP4, MOV, MKV or WebM video."), 400
        # A private temporary file avoids trusting user-supplied filenames.
        with tempfile.TemporaryDirectory(prefix="cricketshot_") as folder:
            path = Path(folder) / ("upload" + suffix)
            uploaded.save(path)
            return jsonify(infer(path))

    @app.errorhandler(ValueError)
    def invalid_video(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(FileNotFoundError)
    def missing_model(exc):
        return (
            jsonify(
                error="Model files are missing. Follow the setup steps in README.md."
            ),
            503,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):
        return (
            jsonify(error="The upload limit is 40 MB. Trim or compress the clip."),
            413,
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6040)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
