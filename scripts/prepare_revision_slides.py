"""Keep the 16-slide native deck, with a single course-grounded image narrative."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def pct(x):return f"{100*x:.1f}%"
def block(text,x,y,w,h,size=20,bold=False):
    return dict(text=text,x=x,y=y,w=w,h=h,size=size,bold=bold)
def table(headers,rows,x=50,y=150,w=860,h=270,fontSize=17,widths=None,**extra):
    return dict(headers=headers,rows=rows,x=x,y=y,w=w,h=h,fontSize=fontSize,widths=widths,**extra)


def main():
    r=json.loads((ROOT/"results/images/metrics.json").read_text())
    c=json.loads((ROOT/"results/revision/comparison.json").read_text())
    h=json.loads((ROOT/"results/hog/metrics.json").read_text())
    a=json.loads((ROOT/"data/images/processed/audit.json").read_text())
    v=json.loads((ROOT/"results/metrics.json").read_text())
    old=json.loads((ROOT/"results/pilot_fullframe/metrics.json").read_text())
    sub=json.loads((ROOT/"configs/submission.json").read_text())
    team=[m["name"]+" ("+m["roll_number"]+")" for m in sub["members"]]
    team_text=team[0]+"  /  "+team[1]+"\n"+team[2]+"\n"+team[3]+"  /  "+team[4]
    names={"hog":"HOG","global":"CNN global","focus":"CNN crop","concat":"Concatenation","contrast":"Contrast"}
    labels={"drive":"Drive","legglance-flick":"Glance / flick","pullshot":"Pull","sweep":"Sweep"}
    slides=[]
    def add(title,**kwargs):slides.append(dict(title=title,**kwargs))
    add("Cricket Shot\nRecognition",kind="cover",kicker="EE604 COURSE PROJECT / 2026",
        subtitle="Gradient features and a frozen CNN\nA controlled four-class image study",
        image="slides/assets/class_0.jpg",team_text=team_text,bottom=sub["github_url"],
        notes="Introduce one-image classification. HOG and frozen features connect directly to lectures 10 and 15. The two-view difference is a small extension to test, not a new backbone. Cover image: selected held-out source image, https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2. Assistance and actual contributions must be attributed accurately.")
    add("Problem, labels and constraints",subtitle="Input: one selected cricket image. Output: one of four dataset labels.",
        blocks=[block("Drive\nLeg glance / flick\nPull\nSweep",50,150,370,170,26,True),
                block("CPU inference, offline after setup.\nNo batter detection, contact localization or technique grading.\nUnknown shots can still receive confident scores.",50,350,850,105,20)],
        images=[dict(path="slides/assets/class_1.jpg",x=490,y=145,w=410,h=190)],
        notes="The user supplies the image. Drive and leg glance/flick are combined upstream classes. A full match is outside scope. Classification is multiclass with one target, not multilabel sigmoid prediction. Source: dataset v2; lectures 15 pages 16-19.")
    add("Course foundation and prior work",subtitle="Use established feature extraction, then test one modest extension",
        tables=[table(["Foundation","Source","Our use"],[
            ["Gradient histograms","Lecture 10; Dalal / Triggs (2005)","Explicit HOG baseline"],
            ["Frozen pretrained features","Lecture 15, pp. 11-14","MobileNetV2 + linear classifier"],
            ["Cricket recognition","KUCricShot (2023); Kang (2025)","Prior task studies; different data"],
            ["Two-view difference","Course implementation","Compare with simple concatenation"]],h=255,fontSize=16,widths=[230,330,300])],
        callout="No claim of a new backbone or an unprecedented combination. Novelty remains limited.",
        notes="Lecture 8 part 2 page 9 distinguishes local/global features; our crop is a regional descriptor, not a keypoint. Lecture 14 part 2 pp 2-3 and 27 cover extractor/classifier and softmax. MobileNetV2 is our architecture choice, not claimed taught by name. Sources: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf ; https://arxiv.org/abs/1801.04381 ; https://github.com/AtanuCSE/CricketShotDetection ; https://arxiv.org/abs/2510.09187 . Do not compare published percentages across datasets.")
    add("Images and fixed visual-group split",subtitle="4,724 original files; 4,723 exact representatives in 363 visual groups",
        tables=[table(["Class","Train","Validation","Test"],[[labels[k]]+[a["split_counts"][s][k] for s in ("train","val","test")] for k in r["classes"]],h=245,widths=[320,180,180,180])],
        blocks=[block("221 training groups / 72 validation groups / 70 test groups\nNo shared group or decoded hash. Match and player identities remain unknown.",50,420,850,58,17)],
        notes="Data: https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset v2. Unknown license in downloaded metadata; no original annotations claimed. Fixed seed 604 folds; this is one holdout, not five-fold cross-validation. Test images are clustered copies and must not be described as 945 independent photographs.")
    add("Evaluation choices and their limits",subtitle="All five methods share images, labels, visual groups and split identities",
        tables=[table(["Choice","Data used","What remains fixed"],[
            ["Fit classifier coefficients","Training only","Frozen CNN; deterministic HOG"],
            ["Choose C: 0.1, 1, 10, 100","Validation macro F1","Smallest C wins ties"],
            ["Temperature and review","Original validation reuse","Keep original settings"],
            ["Final comparison","Same 945 test identities","No new split or seed search"]],h=265,fontSize=16,widths=[300,260,300])],
        callout="Exploratory: grouping changed after preliminary evaluation; HOG was added after old results were known.",
        notes="One validation set was reused for regularization, calibration and threshold selection. This can overfit validation. We disclose it rather than create a new split and pretend the original test was untouched. Initial hash-only contrast result was 96.3%; stronger transformed-copy grouping yields 65.5%. Both results and original individual predictions remain available.")
    add("Implemented pipeline",subtitle="Only the small cricket classifiers learn from the training images",
        images=[dict(path="results/revision/figures/pipeline.png",x=40,y=135,w=880,h=280)],
        callout="The fixed crop is a regional view. It does not detect or segment the batter.",
        notes="Explain the two branches. HOG and CNN global inputs have identical aspect-preserving 224 square geometry. CNN has two view passes; its global, crop and fused heads share features. HOG uses luminance, unsigned gradients, cells, blocks and per-image normalization. No StandardScaler or PCA learns from test. Source: actual implemented pipeline; lecture mapping in docs/course_mapping.md.")
    add("HOG: gradients, cells and blocks",subtitle="Lecture 10: accumulate gradient magnitude in orientation bins",
        tables=[table(["Step","Fixed setting"],[
            ["Input","224 x 224 luminance"],["Orientation votes","9 unsigned bins; linear bin weights"],
            ["Cells","16 x 16 pixels; 14 x 14 cells"],["Blocks","2 x 2 cells; stride 1 cell"],
            ["Normalize / concatenate","L2 blocks, then 6,084-value L2 vector"]],x=50,y=150,w=455,h=260,fontSize=15,widths=[165,290])],
        images=[dict(path="results/revision/figures/hog_steps.png",x=525,y=160,w=385,h=230)],
        callout="Votes use magnitude, not frequency. No learned feature scaling; no test-based HOG tuning.",
        notes="Trace gx and gy central differences, sqrt(gx²+gy²), atan2 modulo 180. A 50-degree gradient of magnitude 90 votes 45 to the 40-degree bin and 45 to 60. No spatial vote interpolation. Each 36-value block divides by sqrt(sum(h²)+1e-10). 13*13*36=6084. This is a documented HOG variant, not the exact signed-angle lecture example or original pedestrian detector. Lecture 10 pp 7-23; original method https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf .")
    add("CNN views and the contrast feature",subtitle="Frozen ImageNet MobileNetV2 gives two 1,280-value pooled embeddings",
        tables=[table(["Representation","Normalized classifier input","Dimension"],[
            ["Global","u","1,280"],["Fixed regional crop","v","1,280"],["Concatenation","[u; v]","2,560"],["Contrast","[u; v; abs(u-v)]","3,840"]],h=235,fontSize=18,widths=[300,360,200])],
        blocks=[block("Absolute difference is nonlinear; signed difference would be redundant\nfor a linear head on concatenation. This motivates a test, not a success claim.",50,407,850,65,18)],
        notes="Both vectors are normalized before joining; the joined descriptor is normalized again. Contrast W has (4,3840) shape plus four biases. It learns weighted multiclass cross-entropy with L2 regularization. No bat-motion interpretation is assigned to feature coordinates. Actual inputs are displayed in the demo. MobileNetV2: https://arxiv.org/abs/1801.04381 ; ONNX weights: https://github.com/onnx/models .")
    result_rows=[]
    for mode in names:
        info=h if mode=="hog" else r["models"][mode]
        result_rows.append([names[mode],str(info["C"]),pct(info["validation"]["macro_f1"]),pct(info["test"]["accuracy"]),pct(info["test"]["macro_f1"])])
    add("Recognition results",subtitle="Same corrected split; every test image contributes, including review cases",
        tables=[table(["Feature","C","Val. F1","Test acc.","Test F1"],result_rows,h=265,fontSize=17,widths=[300,80,160,160,160])],
        callout="HOG is a controlled baseline. Pretrained features help here; contrast adds only three correct images.",
        notes="HOG fixed settings, validation-selected C and predictions are saved under results/hog. Do not try a new descriptor after seeing the 35.3% test score to improve a headline. CNN heads/predictions are retained unchanged. This comparison does not isolate color versus pretraining versus architecture; no general statement that all HOG methods are inferior. Evidence: results/revision/comparison.csv and saved individual probabilities.")
    interval_rows=[[names[k],pct(c["models"][k]["test"]["accuracy"]),f'{100*c["models"][k]["accuracy_95"][0]:.1f} to {100*c["models"][k]["accuracy_95"][1]:.1f}%'] for k in names]
    pair=c["paired_contrast_concat"]["accuracy_difference_95"]
    add("Uncertainty comes from 70 visual groups",subtitle="Whole-group bootstrap, 2,000 paired draws, fixed model and split",
        tables=[table(["Feature","Accuracy","Conditional 95% interval"],interval_rows,h=245,fontSize=17,widths=[300,180,380])],
        callout=f'Contrast minus concatenation: +0.32 points; interval {100*pair[0]:+.2f} to {100*pair[1]:+.2f}. No clear advantage.',
        notes="Draw groups with replacement and keep all images within each drawn group. Reuse draws between methods for paired differences. Statistics are image-weighted in each resample. Intervals omit training/split variability and unknown match correlation. They are not a hypothesis-test certificate of equality. Contrast 619/945, concatenation 616/945. Data and calculations: scripts/analyze_revision.py.")
    cm=r["models"]["contrast"]["test"]["confusion"]
    add("Class errors",subtitle="Contrast confusion counts; rows are dataset labels",
        tables=[table(["True / predicted","Drive","Glance/flick","Pull","Sweep"],[[labels[k]]+cm[i] for i,k in enumerate(r["classes"])],h=270,fontSize=19,widths=[260,150,150,150,150],heatmap=True,totals=[sum(row) for row in cm])],
        callout="Macro F1 averages the four class F1 scores equally. Review does not remove mistakes from this table.",
        notes="Read an off-diagonal entry and explain precision versus recall. Glance/flick is difficult under this protocol. HOG predicts pull for 110 of 224 sweep images. Fixed geometry, similar stances and background are plausible issues, not proven causes for each error. Evidence: saved prediction arithmetic in comparison.json.")
    model=r["models"]["contrast"]
    add("Scores, calibration and secondary review",subtitle="Temperature scaling worsened the recorded test calibration",
        tables=[table(["Contrast scores","NLL","ECE"],[[label,f'{model[key]["nll"]:.3f}',f'{model[key]["ece_10_bins"]:.3f}'] for label,key in [("Unscaled","test_uncalibrated"),("Scaled","test")]],x=50,y=160,w=420,h=165,fontSize=18,widths=[200,110,110])],
        blocks=[block("557 / 945 accepted\n433 accepted predictions correct\n58.9% coverage / 77.7% accuracy",510,165,400,150,25,True),block("Keep T = 0.802 and threshold = 0.70 unchanged.\nThe 90% validation target did not transfer to test.",50,370,850,65,20)],
        callout="Softmax is a relative class score. Agreement is not a guarantee or an unknown-shot detector.",
        notes="NLL and ECE worsen; do not say calibration improved. ECE uses ten equal-width confidence bins. The same validation set selected C, temperature and threshold. The gate needs global/crop/contrast agreement; the heads are correlated. HOG scores are unscaled and do not affect review. Sources: https://proceedings.mlr.press/v70/guo17a.html ; https://arxiv.org/abs/1705.08500 .")
    add("A confident error stays in the demo",subtitle="Selected after evaluation, alongside successful and disagreement examples",
        images=[dict(path="results/images/figures/failure.jpg",x=50,y=145,w=860,h=280)],
        callout="Inspect the upstream label, predicted label and actual input views. Confidence can be wrong.",
        notes="Use image_04199, the preserved confident failure. Do not invent its exact cause or silently relabel it. The selected picture is an illustration, not an estimate of failure frequency. Real data source: aneesh10/cricket-shot-dataset v2. Follow with the view-disagreement image_00014.")
    add("Local image demonstration",subtitle="Upload an image, compare CNN and HOG, inspect the review reason, export JSON",
        images=[dict(path="results/ui_image_prediction.png",x=50,y=135,w=860,h=325)],
        notes="Run setup.bat on a fresh machine, then run_demo.bat. Default URL http://127.0.0.1:6040. Revision verification may use 6060 to avoid another local service. Show success image_00210, confident error image_04199 and disagreement image_00014. Selected examples are honestly labeled. JSON preserves temperature-scaled score semantics. Screenshot must come from the revised running app, not a mockup.")
    add("What we can defend",subtitle="An implemented comparison with useful negative findings",
        blocks=[block("Course foundation",50,145,850,35,26,True),block("HOG and frozen features, each followed by a trained multiclass head.",50,193,850,50,21),
                block("Limited extension",50,275,850,35,26,True),block("Contrast gains three test images; uncertainty spans no gain.\nThe audit corrects optimism, and calibration fails to transfer.",50,323,850,85,21)],
        callout="Next useful step: independently labeled data with known match and player identities.",
        notes="Novelty remains a weakness, not a promise of marks. Do not claim a new architecture, detector or state-of-the-art result. Mention unknown upstream license, possible missed copies and unverified labels. Course mapping and viva guide contain actual page references and a tensor-shape trace. Team identities and the real repository URL are on the cover; assistance is attributed.")
    vm=v["models"]["phase_delta"];pm=old["models"]["phase_delta"]
    add("Backup: video pilot and duplicate audit",subtitle="Separate eight-class data and reused pilot holdout; not the main recognition claim",
        tables=[table(["Video model","Accuracy","Macro F1"],[["Full frame",pct(pm["test"]["accuracy"]),pct(pm["test"]["macro_f1"])],["Fixed crop",pct(vm["test"]["accuracy"]),pct(vm["test"]["macro_f1"])],["Majority baseline",pct(v["majority_baseline"]["accuracy"]),pct(v["majority_baseline"]["macro_f1"])]],y=150,h=205,fontSize=18,widths=[450,205,205])],
        blocks=[block("562 clips / 25 source groups / 116 test clips from five groups\nImage audit: hashes, transforms, ORB/RANSAC links; all connected copies stay together.",50,387,850,65,17)],
        notes="Use only for questions. Equal 30.2% majority accuracy means weak video recognition, despite higher macro F1. Fixed crop followed pilot test feedback. Source: https://www.kaggle.com/datasets/amittalmale/cricket-shots . Image audit: pHash <=6 bits with thumbnail MAD, rotations/mirrors, then 15 feature neighbors within cosine .20. ORB binary matching, .72 ratio, >=16 matches, >=12 RANSAC inliers, >=65% inlier fraction, >=8% hull coverage. These are evaluation tools, not a novel shot classifier. Other sources: https://github.com/onnx/models ; https://github.com/scikit-learn/scikit-learn ; https://github.com/microsoft/onnxruntime ; https://github.com/opencv/opencv ; https://arxiv.org/abs/1608.00859 . Full URLs, saved evidence and complete code appendix accompany the report.")
    assert len(slides)==16
    (ROOT/"slides/content.json").write_text(json.dumps({"slides":slides},indent=2),encoding="utf-8")
    text="# Revised presentation notes\n\nSlides 1-15 form the main talk; slide 16 is backup. Adjust timing to the assigned slot.\n\n"
    for i,slide in enumerate(slides,1):text+=f'## {i}. {slide["title"].replace(chr(10)," ")}\n\n{slide["notes"]}\n\n'
    (ROOT/"slides/speaker_notes.md").write_text(text,encoding="utf-8")
    print("Prepared 16 revised slides and speaker notes")


if __name__=="__main__":main()
