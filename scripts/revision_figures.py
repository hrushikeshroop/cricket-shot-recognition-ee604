"""Create explanatory figures, a measured image trace and current demo timing."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cricketshot.core import prepare_frame
from cricketshot.images import ImageRecognizer, image_descriptor, image_views, load_image
from cricketshot.hog import HOGRecognizer, hog_descriptor


def main():
    out = ROOT / "results/revision"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.set(xlim=(0, 12), ylim=(0, 3.7)); ax.axis("off")
    for y, labels, color in [(2.2, ["RGB image\n+ fixed regional crop", "Frozen MobileNetV2\ntwo 1,280-value vectors", "Normalize and combine\n[u; v; |u-v|]", "Trained linear head\nfour shot scores"], "#dfeadb"),
                              (.55, ["Same global input\n224 x 224 grayscale", "Gradients, cells\nand overlapping blocks", "Normalize HOG\n6,084 values", "Trained linear head\nfour shot scores"], "#f0e5d8")]:
        for i, label in enumerate(labels):
            x = .05 + i*3
            ax.add_patch(Rectangle((x,y),2.7,.9,facecolor=color,edgecolor="#485b48"))
            ax.text(x+1.35,y+.45,label,ha="center",va="center",fontsize=14)
            if i < 3: ax.annotate("",(x+2.98,y+.45),(x+2.72,y+.45),arrowprops={"arrowstyle":"->"})
    ax.text(.05,3.35,"CNN branch: one shared encoder, four controlled descriptor ablations",fontsize=13)
    ax.text(.05,.1,"HOG branch: deterministic features; identical labels, visual groups and train/validation/test split",fontsize=13)
    fig.tight_layout(pad=.4)
    for ext in ("pdf","png"):fig.savefig(figures/("pipeline."+ext),bbox_inches="tight")
    plt.close(fig)

    example = json.loads((ROOT/"demo/image_examples.json").read_text())[0]
    image = load_image(ROOT/example["path"])
    feature, gray, magnitude, cells = hog_descriptor(image, return_maps=True)
    fig, axes = plt.subplots(1,3,figsize=(6,2.2))
    axes[0].imshow(gray,cmap="gray",vmin=0,vmax=1);axes[0].set_title("Luminance",fontsize=14)
    axes[1].imshow(magnitude,cmap="gray");axes[1].set_title("Magnitude",fontsize=14)
    axes[2].imshow(gray,cmap="gray",alpha=.35,vmin=0,vmax=1)
    for yy in range(14):
        for xx in range(14):
            for b in range(9):
                angle=np.deg2rad(b*20);length=6*cells[yy,xx,b]/max(cells[yy,xx].max(),1e-12)
                cx,cy=xx*16+8,yy*16+8
                axes[2].plot([cx-length*np.cos(angle),cx+length*np.cos(angle)],
                             [cy-length*np.sin(angle),cy+length*np.sin(angle)],color="#284b30",lw=.7)
    axes[2].set_title("Cell bins",fontsize=14)
    for ax in axes:ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures/"hog_steps.png",bbox_inches="tight");plt.close(fig)

    recognizer=ImageRecognizer();hog=HOGRecognizer()
    views=image_views(image)
    features=recognizer.encoder.encode([np.asarray(v) for v in views])
    z=image_descriptor(features,"contrast")
    logits=z@recognizer.heads["contrast_coef"].T+recognizer.heads["contrast_intercept"]
    result,_=recognizer.predict(ROOT/example["path"])
    trace={"id":example["id"],"dataset_label":example["label"],"selected_after_evaluation":True,
           "functions":["load_image","image_views","prepare_frame","Encoder.encode","image_descriptor","ImageRecognizer.predict_features"],
           "loaded_size_wh":list(image.size),"view_sizes_wh":[list(v.size) for v in views],
           "encoder_input_shape":list(prepare_frame(np.asarray(views[0]),"full").shape),
           "encoder_output_shapes":[o.shape for o in recognizer.encoder.session.get_outputs()],
           "two_view_features_shape":list(features.shape),"contrast_shape":list(z.shape),
           "head_weight_shape":list(recognizer.heads["contrast_coef"].shape),
           "contrast_l2_norm":float(np.linalg.norm(z)),"first_eight_contrast_values":z[:8].tolist(),
           "logits":logits.tolist(),"temperature":recognizer.config["temperatures"]["contrast"],
           "result":result,"hog_shape":list(feature.shape),"hog_l2_norm":float(np.linalg.norm(feature)),
           "hog_result":hog.predict(ROOT/example["path"])}
    (out/"worked_trace.json").write_text(json.dumps(trace,indent=2))
    rows=list(csv.DictReader((ROOT/"data/images/processed/manifest.csv").open(encoding="utf-8",newline="")))
    tests=[r for r in rows if r["split"]=="test"]
    timings=[]
    for index in np.linspace(0,len(tests)-1,50,dtype=int):
        path=ROOT/tests[index]["path"];start=time.perf_counter()
        recognizer.predict(path);hog.predict(path)
        timings.append(time.perf_counter()-start)
    timing={"n":50,"median_seconds":float(np.median(timings)),"p95_seconds":float(np.percentile(timings,95)),
            "times_seconds":timings,"scope":"CNN prediction plus HOG comparison, including separate image decodes; excludes model initialization, preview encoding and HTTP transport; warmed models"}
    (out/"latency.json").write_text(json.dumps(timing,indent=2))
    print("Created pipeline, HOG explanation, measured trace and current inference timing")


if __name__ == "__main__":main()
