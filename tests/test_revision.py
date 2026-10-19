"""Checks for HOG vote arithmetic, cache binding, export and API parity."""
from pathlib import Path
import csv
import io
import json
import hashlib
import pytest
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
    example = next(e for e in json.loads((ROOT / "demo/image_examples.json").read_text()) if e["demo_role"] == "confident_error")
    payload = (ROOT / example["path"]).read_bytes()
    response = client.post("/image-api/predict", data={"image": (io.BytesIO(payload), "failure.png")})
    assert response.status_code == 200
    result = response.json
    assert "hog_baseline" in result and result["score_type"] == "temperature-scaled softmax"
    expected = HOGRecognizer().predict(ROOT / example["path"])
    assert result["hog_baseline"]["label"] == expected["label"]
    assert np.allclose(list(result["hog_baseline"]["probabilities"].values()), list(expected["probabilities"].values()))


def test_bundled_examples_cover_all_labels_and_preserve_demo_behaviour():
    """A public clone must show all classes and the documented real error."""
    from app import create_app
    client = create_app().test_client()
    manifest = json.loads((ROOT / "demo/image_examples.json").read_text())
    bundled = [e for e in manifest if e.get("bundled")]
    assert len(bundled) == 5
    assert {e["label"] for e in bundled if e["demo_role"] == "success"} == {"drive", "legglance-flick", "pullshot", "sweep"}
    assert sum(e["demo_role"] == "confident_error" for e in bundled) == 1
    for e in bundled:
        assert hashlib.sha256((ROOT/e["path"]).read_bytes()).hexdigest() == e["example_sha256"]
        result = client.post("/image-api/example/"+e["id"])
        assert result.status_code == 200
        assert result.json["label"] == e["expected_prediction"]
        if e["demo_role"] == "success":
            assert result.json["label"] == e["label"]
        else:
            assert result.json["label"] != e["label"] and result.json["confidence"] > .95


def test_posthoc_view_diagnostics_reconcile_with_saved_predictions():
    """Check both decision directions and that agreement partitions the test."""
    d = json.loads((ROOT/"results/revision/view_diagnostics.json").read_text())
    assert d["prediction_archive_sha256"] == hashlib.sha256((ROOT/"results/images/test_predictions.npz").read_bytes()).hexdigest()
    with np.load(ROOT/"results/images/test_predictions.npz", allow_pickle=False) as a:
        concat_ok = a["p_concat"].argmax(1) == a["y"]
        contrast_ok = a["p_contrast"].argmax(1) == a["y"]
        assert d["strata"]["all"]["gained"] == int((contrast_ok & ~concat_ok).sum())
        assert d["strata"]["all"]["lost"] == int((concat_ok & ~contrast_ok).sum())
    for key in ("n", "concat_correct", "contrast_correct", "gained", "lost"):
        assert d["strata"]["all"][key] == d["strata"]["views_agree"][key]+d["strata"]["views_disagree"][key]
    all_test = d["strata"]["all"]
    assert all_test["gained"]-all_test["lost"] == all_test["contrast_correct"]-all_test["concat_correct"] == 3


@pytest.mark.parametrize("external", [False, True])
def test_runtime_verifier_cli_exercises_available_examples(tmp_path, external):
    """The initial input hash must survive checks of later demo examples."""
    import subprocess
    import sys
    proof=tmp_path/"runtime.json"
    command=[sys.executable,str(ROOT/"scripts/verify_runtime.py"),"--output",str(proof)]
    if external:
        entry=next(e for e in json.loads((ROOT/"demo/image_examples.json").read_text()) if e.get("bundled"))
        source=ROOT/entry["path"]
        initial_bytes=source.read_bytes()
        command.extend(["--image",str(source)])
    else:
        from PIL import ImageDraw
        pattern=Image.new("RGB", (320,240), "#448344")
        ImageDraw.Draw(pattern).line((40,210,240,30),fill="white",width=12)
        buffer=io.BytesIO();pattern.save(buffer,format="PNG")
        initial_bytes=buffer.getvalue()
    run=subprocess.run(command,
                       cwd=ROOT,capture_output=True,text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    result=json.loads(proof.read_text())
    assert result["all_pass"] and len(result["example_checks"])>=5
    assert sum(e["bundled"] for e in result["example_checks"]) == 5
    assert result["input_sha256"] == hashlib.sha256(initial_bytes).hexdigest()
    assert result["input"] == ("external photograph" if external else "generated wiring pattern")


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
