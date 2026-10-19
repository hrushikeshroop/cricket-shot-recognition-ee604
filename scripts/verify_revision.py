"""Current-revision checks; keep earlier verification records explicitly separate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import csv
import hashlib
import json
import numpy as np
from cricketshot.core import Encoder
from cricketshot.images import load_image,image_views
from cricketshot.hog import PARAMETERS,HOGRecognizer
from analyze_revision import confusion_statistics


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/"results/revision";out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((ROOT/"data/images/processed/manifest.csv").open(encoding="utf-8",newline="")))
    comparison=json.loads((out/"comparison.json").read_text())
    original=json.loads((ROOT/"results/images/metrics.json").read_text())
    checks=[]
    def check(name,condition,detail=""):
        checks.append({"name":name,"pass":bool(condition),"detail":detail})
        if not condition:raise AssertionError(name+": "+detail)
    for name in ("hog","global","focus","concat","contrast"):
        path=ROOT/("results/hog/test_predictions.npz" if name=="hog" else "results/images/test_predictions.npz")
        with np.load(path,allow_pickle=False) as arrays:
            p=arrays["p"] if name=="hog" else arrays["p_"+name]
            y=arrays["y"];cm=np.zeros((4,4),dtype=int)
            np.add.at(cm,(y,p.argmax(1)),1)
        computed=confusion_statistics(cm)
        recorded=comparison["models"][name]["test"]
        check(name+" independent confusion arithmetic",np.allclose(computed,[recorded["accuracy"],recorded["macro_f1"]],atol=1e-12))
    with np.load(ROOT/"results/images/test_predictions.npz",allow_pickle=False) as arrays:
        accepted=arrays["accepted"];correct=arrays["p_contrast"].argmax(1)==arrays["y"]
    check("Review exact counts preserved",int(accepted.sum())==557 and int((accepted&correct).sum())==433)
    pair=comparison["paired_contrast_concat"]
    check("Paired contrast gain is uncertain",pair["correct_difference"]==3 and pair["accuracy_difference_95"][0]<0<pair["accuracy_difference_95"][1])
    cal=original["models"]["contrast"]
    check("Calibration failure recorded",cal["test"]["ece_10_bins"]>cal["test_uncalibrated"]["ece_10_bins"] and cal["test"]["nll"]>cal["test_uncalibrated"]["nll"])
    check("Original threshold retained",abs(original["review"]["threshold"]-.70)<1e-12)
    model=HOGRecognizer()
    check("HOG preprocessing matches model",model.config["parameters"]==PARAMETERS)
    meta=json.loads((ROOT/"data/images/processed/hog_features.json").read_text())
    check("HOG cache binds manifest and implementation",meta["manifest_sha256"]==digest(ROOT/"data/images/processed/manifest.csv") and meta["implementation_sha256"]==digest(ROOT/"src/cricketshot/hog.py"))
    with np.load(ROOT/"data/images/processed/features.npz",allow_pickle=False) as arrays:
        features=arrays["features"].copy()
        check("CNN cache IDs match manifest",arrays["ids"].tolist()==[r["id"] for r in rows])
    encoder=Encoder(view="full");maximum_error=0.0
    for index in np.linspace(0,len(rows)-1,64,dtype=int):
        row=rows[index];image=load_image(ROOT/row["path"])
        pixel_hash=hashlib.sha256(image.width.to_bytes(4,"big")+image.height.to_bytes(4,"big")+image.tobytes()).hexdigest()
        check("Raw pixel hash "+row["id"],pixel_hash==row["rgb_sha256"])
        fresh=encoder.encode([np.asarray(v) for v in image_views(image)])
        error=float(np.max(np.abs(fresh-features[index])))
        maximum_error=max(maximum_error,error)
        check("Fresh two-view features "+row["id"],np.allclose(fresh,features[index],atol=1e-6))
    cache_proof={"manifest_sha256":digest(ROOT/"data/images/processed/manifest.csv"),
                 "feature_cache_sha256":digest(ROOT/"data/images/processed/features.npz"),
                 "encoder_sha256":digest(ROOT/"models/mobilenetv2_features.onnx"),
                 "images_module_sha256":digest(ROOT/"src/cricketshot/images.py"),
                 "core_module_sha256":digest(ROOT/"src/cricketshot/core.py"),
                 "fresh_images_checked":64,"two_views_per_image":True,"maximum_absolute_error":maximum_error,
                 "scope":"64 deterministic samples across all splits; not every cached image independently recomputed"}
    (out/"cache_validation.json").write_text(json.dumps(cache_proof,indent=2))
    # If the original sibling workspace exists, verify it was not modified.
    sibling=ROOT.parents[1]
    baseline_path=ROOT/".build/local_notes/docs/original_files_baseline.json"
    if baseline_path.exists() and (sibling/"app.py").exists() and (sibling/"revision").exists():
        baseline=json.loads(baseline_path.read_text())
        changed=[item["path"] for item in baseline if not (sibling/item["path"]).exists() or digest(sibling/item["path"])!=item["sha256"]]
        check("Original project files preserved",not changed,str(changed))
        preserved=["data/images/processed/manifest.csv","models/image/heads.npz","models/image/classifier.json",
                   "results/images/metrics.json","results/images/test_predictions.npz","results/images_initial_hash/metrics.json",
                   "results/images_initial_hash/test_predictions.npz","results/metrics.json","results/test_predictions.npz"]
        for name in preserved:check("Preserved evidence "+name,digest(ROOT/name)==digest(sibling/name))
    result={"all_checks_pass":True,"checks":checks,"cache_validation":cache_proof,
            "scope":"Current revision: arithmetic, fixed evidence, preprocessing parity and original preservation. Pytest, HTTP, render and document checks have separate records."}
    (out/"verification.json").write_text(json.dumps(result,indent=2))
    print(f"{len(checks)} current revision checks passed; maximum CNN feature discrepancy {maximum_error:g}")


if __name__=="__main__":main()
