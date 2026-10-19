"""Checks for HOG vote arithmetic, cache binding, export and API parity."""
from pathlib import Path
import csv
import io
import json
import hashlib
import numpy as np
from PIL import Image
from cricketshot.hog import (PARAMETERS, HOGRecognizer, cell_histograms,
                            gradients, hog_descriptor)
from cricketshot.images import load_image
ROOT = Path(__file__).resolve().parents[1]


def test_hog_votes_are_magnitude_weighted_and_interpolated():
    magnitude = np.array([[90.0]])
    angle = np.array([[50.0]])
    cells = cell_histograms(magnitude, angle, cell_size=1)
    assert np.allclose(cells.sum(), 90)
    # Nine unsigned bins have centers at 0,20,40,...,160 degrees.
    assert cells[0, 0, 2] == 45 and cells[0, 0, 3] == 45
    wrap = cell_histograms(np.array([[20.0]]), np.array([[170.0]]), cell_size=1)
    assert wrap[0, 0, 8] == 10 and wrap[0, 0, 0] == 10


def test_hog_geometry_flat_image_and_unsigned_edges():
    flat = Image.new("RGB", (320, 240), (40, 40, 40))
    x = hog_descriptor(flat)
    assert x.shape == (6084,) and np.isfinite(x).all() and np.count_nonzero(x) == 0
    ramp = np.tile(np.linspace(0, 1, 224), (224, 1)).astype(np.float32)
    m, a = gradients(ramp)
    assert np.allclose(a[:, 1:-1], 0) and np.all(m[:, 1:-1] > 0)
    mr, ar = gradients(1-ramp)
    assert np.allclose(mr, m, atol=1e-6) and np.allclose(ar[:, 1:-1], 0)


def test_hog_saved_features_and_export_match_fresh_inputs():
    meta = json.loads((ROOT / "data/images/processed/hog_features.json").read_text())
    assert meta["parameters"] == PARAMETERS
    for key, path in [("manifest_sha256", "data/images/processed/manifest.csv"),
                      ("implementation_sha256", "src/cricketshot/hog.py"),
                      ("protocol_sha256", "configs/hog_experiment.json")]:
        assert meta[key] == hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    rows = list(csv.DictReader((ROOT / "data/images/processed/manifest.csv").open(encoding="utf-8", newline="")))
    model = HOGRecognizer()
    with np.load(ROOT / "data/images/processed/hog_features.npz", allow_pickle=False) as cache:
        features = cache["features"]
        assert cache["ids"].tolist() == [r["id"] for r in rows]
        for index in np.linspace(0, len(rows)-1, 8, dtype=int):
            fresh = hog_descriptor(load_image(ROOT / rows[index]["path"]))
            assert np.allclose(fresh, features[index], atol=1e-7)
        test = np.array([r["split"] == "test" for r in rows])
        with np.load(ROOT / "results/hog/test_predictions.npz", allow_pickle=False) as predictions:
            p = model.predict_features(features[test])
            assert np.allclose(p, predictions["p"], atol=1e-7)
            assert np.allclose(p.sum(1), 1)


def test_uploaded_image_compares_both_real_models():
    from app import create_app
    client = create_app().test_client()
    example = json.loads((ROOT / "demo/image_examples.json").read_text())[2]
    payload = (ROOT / example["path"]).read_bytes()
    response = client.post("/image-api/predict", data={"image": (io.BytesIO(payload), "failure.png")})
    assert response.status_code == 200
    result = response.json
    assert "hog_baseline" in result and result["score_type"] == "temperature-scaled softmax"
    expected = HOGRecognizer().predict(ROOT / example["path"])
    assert result["hog_baseline"]["label"] == expected["label"]
    assert np.allclose(list(result["hog_baseline"]["probabilities"].values()), list(expected["probabilities"].values()))


def test_missing_optional_examples_are_hidden(monkeypatch):
    import app as demo_app
    absent = "demo/images/not_distributed.png"
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.as_posix().endswith(absent) else original(p))
    original_read = Path.read_text
    def read(p, *args, **kwargs):
        if p.name == "image_examples.json":
            return json.dumps([{"id": "absent", "path": absent}])
        return original_read(p, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    client = demo_app.create_app().test_client()
    assert client.get("/image-api/examples").json == []
    assert client.post("/image-api/example/absent").status_code == 404
