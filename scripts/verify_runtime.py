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
    response = client.post("/image-api/predict", data={"image": (io.BytesIO(payload), filename)})
    assert response.status_code == 200, response.get_data(as_text=True)
    result = response.json
    for p in (result["probabilities"], result["hog_baseline"]["probabilities"]):
        assert len(p) == 4 and abs(sum(p.values())-1) < 1e-8
    assert len(result["previews"]) == 2
    assert client.post("/image-api/predict", data={"image": (io.BytesIO(b"bad"), "bad.png")}).status_code == 400
    examples = client.get("/image-api/examples").json
    for entry in examples:
        assert client.get("/image-api/example/"+entry["id"]+"/thumbnail").status_code == 200
    paths = [ROOT/"app.py", ROOT/"configs/submission.json", Path(__file__).resolve()]
    for folder in ("src", "templates", "static", "models"):
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in {".py", ".html", ".js", ".css", ".json", ".npz", ".onnx"})
    proof = {"all_pass": True, "scope": "Repository runtime; Flask request client, actual multipart upload, CNN/HOG probabilities, previews, malformed-upload rejection, team and URL rendering. No new accuracy measurement.",
             "input": "external photograph" if args.image else "generated wiring pattern",
             "input_sha256": hashlib.sha256(payload).hexdigest(), "label": result["label"],
             "available_image_examples": len(examples), "python": sys.version,
             "packages": {p: importlib.metadata.version(p) for p in ["numpy", "Pillow", "onnxruntime", "opencv-python-headless", "Flask"]},
             "source_model_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in proof.items() if k != "source_model_hashes"}, indent=2))


if __name__ == "__main__":
    main()
