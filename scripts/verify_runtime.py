"""Check repository-only inference without training data or development packages.

Uses Flask's request client and an actual multipart upload. With --image, the
input may be a local photograph outside the repository. Otherwise a generated
pattern checks wiring only; no recognition-quality claim follows from it.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import importlib.metadata
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    client = create_app().test_client()
    home = client.get("/")
    assert home.status_code == 200
    sub = json.loads((ROOT/"configs/submission.json").read_text())
    html = home.get_data(as_text=True)
    assert all(m["name"] in html and m["roll_number"] in html for m in sub["members"])
    assert sub["github_url"] in html
    assert client.get("/video").status_code == 200
    assert client.get("/api/status").json["ready"]
    if args.image:
        payload = args.image.read_bytes()
        filename = args.image.name
    else:
        im = Image.new("RGB", (320, 240), "#448344")
        ImageDraw.Draw(im).line((40, 210, 240, 30), fill="white", width=12)
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        payload, filename = stream.getvalue(), "generated_wiring_check.png"
    input_sha256 = hashlib.sha256(payload).hexdigest()
    response = client.post("/image-api/predict", data={"image": (io.BytesIO(payload), filename)})
    assert response.status_code == 200, response.get_data(as_text=True)
    result = response.json
    for p in (result["probabilities"], result["hog_baseline"]["probabilities"]):
        assert len(p) == 4 and abs(sum(p.values())-1) < 1e-8
    assert len(result["previews"]) == 2
    assert client.post("/image-api/predict", data={"image": (io.BytesIO(b"bad"), "bad.png")}).status_code == 400
    examples = client.get("/image-api/examples").json
    example_metadata = {entry["id"]: entry for entry in json.loads((ROOT/"demo/image_examples.json").read_text())}
    bundled = [entry for entry in example_metadata.values() if entry.get("bundled")]
    assert len(bundled) == 5 and {e["label"] for e in bundled if e["demo_role"] == "success"} == {"drive","legglance-flick","pullshot","sweep"}
    assert {e["id"] for e in bundled}.issubset({e["id"] for e in examples})
    demo_checks=[]
    for available in examples:
        # The public API deliberately omits filesystem paths. Resolve them
        # from the local manifest when checking actual upload bytes.
        entry=example_metadata[available["id"]]
        assert client.get("/image-api/example/"+entry["id"]+"/thumbnail").status_code == 200
        example_payload=(ROOT/entry["path"]).read_bytes()
        assert hashlib.sha256(example_payload).hexdigest()==entry["example_sha256"]
        example=client.post("/image-api/example/"+entry["id"])
        upload=client.post("/image-api/predict",data={"image":(io.BytesIO(example_payload),Path(entry["path"]).name)})
        assert example.status_code==upload.status_code==200
        assert example.json["probabilities"]==upload.json["probabilities"]
        assert example.json["hog_baseline"]==upload.json["hog_baseline"]
        assert upload.json["label"]==entry["expected_prediction"]
        if entry["demo_role"]=="success":assert upload.json["label"]==entry["label"]
        if entry["demo_role"]=="confident_error":assert upload.json["label"]!=entry["label"]
        if entry["demo_role"]=="view_disagreement":assert not upload.json["prediction_consistent"]
        demo_checks.append({"id":entry["id"],"role":entry["demo_role"],"bundled":entry["bundled"],"label":upload.json["label"],"example_upload_parity":True})
    paths = [ROOT/"app.py", ROOT/"configs/submission.json", Path(__file__).resolve()]
    for folder in ("src", "templates", "static", "models"):
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in {".py", ".html", ".js", ".css", ".json", ".npz", ".onnx"})
    proof = {"all_pass": True, "scope": "Repository runtime; Flask request client, actual multipart upload, CNN/HOG probabilities, previews, malformed-upload rejection, team and URL rendering. No new accuracy measurement.",
             "input": "external photograph" if args.image else "generated wiring pattern",
             "input_sha256": input_sha256, "label": result["label"],
             "available_image_examples": len(examples), "example_checks":demo_checks, "python": sys.version,
             "packages": {p: importlib.metadata.version(p) for p in ["numpy", "Pillow", "onnxruntime", "opencv-python-headless", "Flask"]},
             "source_model_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in proof.items() if k != "source_model_hashes"}, indent=2))


if __name__ == "__main__":
    main()
