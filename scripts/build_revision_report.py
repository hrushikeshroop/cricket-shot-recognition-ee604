"""Build the course revision from saved evidence, including its full source."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import zipfile
import pymupdf as fitz
from build_report import ROOT, pct, table, code_appendix, esc


def main():
    report = ROOT / "report"
    generated = report / "generated"
    generated.mkdir(exist_ok=True)
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    build = ROOT / ".build/latex/revision"
    build.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/"results/images/metrics.json").read_text())
    comparison = json.loads((ROOT/"results/revision/comparison.json").read_text())
    hog = json.loads((ROOT/"results/hog/metrics.json").read_text())
    audit = json.loads((ROOT/"data/images/processed/audit.json").read_text())
    latency = json.loads((ROOT/"results/revision/latency.json").read_text())
    sub = json.loads((ROOT/"configs/submission.json").read_text())
    names = {"drive":"Drive", "legglance-flick":"Glance/flick", "pullshot":"Pull", "sweep":"Sweep"}
    data_rows = [[names[c]]+[audit["split_counts"][s][c] for s in ("train","val","test")]
                 for c in original["classes"]]
    data_rows.append(["Total",2834,944,945])
    (generated/"image_data_table.tex").write_text(table(["Label","Train","Val.","Test"],data_rows,
        "Retained image counts; all visual groups remain in one split.","tab:data"))
    labels={"hog":"HOG", "global":"CNN global", "focus":"CNN crop", "concat":"Concat.", "contrast":"Contrast"}
    rows=[];intervals=[]
    for mode in labels:
        info=hog if mode=="hog" else original["models"][mode]
        result=comparison["models"][mode]
        rows.append([labels[mode],info["dimension"],info["C"],pct(result["test"]["accuracy"]),pct(result["test"]["macro_f1"])])
        intervals.append([labels[mode],f'{pct(result["accuracy_95"][0])}--{pct(result["accuracy_95"][1])}',
                           f'{pct(result["macro_f1_95"][0])}--{pct(result["macro_f1_95"][1])}'])
    (generated/"image_results_table.tex").write_text(table(["Feature","Dim.","$C$","Acc.","F1"],rows,
        "All-test accuracy and macro F1 in percent. Same images, labels, groups, split and regularization grid; HOG has no ImageNet pretraining.","tab:results"))
    (generated/"image_uncertainty_table.tex").write_text(table(["Feature","Acc. 95\\%","F1 95\\%"],intervals,
        "Conditional visual-group bootstrap intervals, in percent, from saved individual predictions.","tab:intervals"))
    contrast=original["models"]["contrast"]
    values={"HOG_ACC":pct(hog["test"]["accuracy"]),"HOG_F1":pct(hog["test"]["macro_f1"]),
            "HOG_VAL_F1":pct(hog["validation"]["macro_f1"]),"HOG_C":str(hog["C"]),
            "GLOBAL_ACC":pct(original["models"]["global"]["test"]["accuracy"]),
            "GLOBAL_F1":pct(original["models"]["global"]["test"]["macro_f1"]),
            "ACC":pct(contrast["test"]["accuracy"]),"F1":pct(contrast["test"]["macro_f1"]),
            "PAIR_LOW":f'{100*comparison["paired_contrast_concat"]["accuracy_difference_95"][0]:+.2f}',
            "PAIR_HIGH":f'{100*comparison["paired_contrast_concat"]["accuracy_difference_95"][1]:+.2f}',
            "ECE_BEFORE":f'{contrast["test_uncalibrated"]["ece_10_bins"]:.3f}',
            "ECE_AFTER":f'{contrast["test"]["ece_10_bins"]:.3f}',
            "NLL_BEFORE":f'{contrast["test_uncalibrated"]["nll"]:.3f}',
            "NLL_AFTER":f'{contrast["test"]["nll"]:.3f}',
            "MEDIAN_MS":f'{latency["median_seconds"]*1000:.0f}',
            "P95_MS":f'{latency["p95_seconds"]*1000:.0f}'}
    recalls=contrast["test"]["per_class"]
    values["CLASS_DISCUSSION"]=(f'Glance/flick recall is {pct(recalls["legglance-flick"]["recall"])}\\% on 224 examples, '
        f'while pull recall is {pct(recalls["pullshot"]["recall"])}\\% on 252. ')
    body=(report/"image_content_template.tex").read_text(encoding="utf-8")
    for key,value in values.items():body=body.replace("{{"+key+"}}",value)
    if re.search(r"\{\{[A-Z_]+\}\}",body):raise ValueError("Unresolved report variable")
    (generated/"image_body.tex").write_text(body,encoding="utf-8")
    members = [esc(m["name"])+" ("+esc(m["roll_number"])+")" for m in sub["members"]]
    author_lines = [members[0]+r"\quad "+members[1], members[2], members[3]+r"\quad "+members[4]]
    (generated/"authors.tex").write_text("\\author{"+r"\\".join(author_lines)+r"\\Indian Institute of Technology Kanpur\\EE604, October 2026}"+"\n")
    calibration=[]
    for label,key in [("Unscaled","test_uncalibrated"),("Scaled","test")]:
        m=contrast[key];calibration.append([label,f'{m["nll"]:.3f}',f'{m["ece_10_bins"]:.3f}',f'{m["brier_multiclass"]:.3f}'])
    cal_table=table(["Scores","NLL","ECE","Brier"],calibration,"Unchanged contrast scores on test; lower is better. Scaling worsens NLL and ECE.","tab:calibration").replace("[t]","[h]")
    stress=[]
    for mode in ("global","contrast"):
        stress.append([labels[mode],pct(original["models"][mode]["test"]["accuracy"])]+[
            pct(original["stress"][condition][mode]["accuracy"]) for condition in ("dark","blur","flip")])
    stress_table=table(["Head","Clean","Dark","Blur","Mirror"],stress,"Preserved descriptive stress-test accuracy in percent; no retraining or parameter choice.","tab:stress").replace("[t]","[h]")
    url=sub.get("github_url")
    repository=(r"Project repository: \url{"+url+"}." if url else
                "The real GitHub URL is pending the group's decision and must be inserted before submission. No repository has been published for this handoff.")
    supplement=r"""\section{Supplementary evidence}
\subsection{Visual-copy audit}\label{app:audit}
The initial audit uses decoded RGB hashes, 64-bit perceptual hash Hamming distance at most six, and 16 by 16 RGB thumbnail mean absolute difference at most 18. Exact copies collapse; conflicting-label components are quarantined. The stronger audit adds quarter-turn/mirror hashes (same bit cutoff, aligned thumbnail difference at most 24), then proposes the nearest 15 frozen-feature neighbors within cosine distance 0.20.

ORB binary descriptors and a RANSAC homography verify proposed geometric links. Requirements are a 0.72 match-ratio cutoff, at least 16 filtered matches, 12 inliers, 65\% inlier fraction and 8\% convex-hull coverage in both images. This adds 948 transform and 1,797 geometric edges. There are no final conflicting components. All linked copies and initial hash components stay in one split. These checks reduce leakage but can still miss related photographs or join visually similar ones. Full criteria and preserved edges are in the commented audit script and data directory.

\subsection{Secondary diagnostics}\label{app:secondary}
"""+cal_table+r"""
The original temperature is approximately 0.802. Validation is reused for regularization, temperature and review selection. The review search chooses the lowest threshold from 0.50 to 0.95 in steps of 0.05 attaining 90\% validation accuracy with at least 20 accepted images and 20\% coverage, with 0.80 fallback. Its chosen 0.70 threshold stays unchanged. At that threshold confidence alone covers 62.2\% with 77.6\% accepted accuracy; adding agreement covers 58.9\% with 77.7\%. This modest difference is descriptive, not a guaranteed improvement.

"""+stress_table+r"""
Dark multiplies source brightness by 0.6; blur uses Gaussian radius two at source resolution; mirror reflects horizontally while retaining the four broad labels. Both CNN views are recomputed. These conditions do not represent a new match or camera, and blur severity varies with source resolution. Original predictions, stress evidence and CNN-only timings remain under \texttt{results/images/}; preliminary hash-only evidence remains under \texttt{results/images\_initial\_hash/}.

\subsection{Exploratory video pilot}\label{app:video}
The separate Kaggle CRICKET\_SHOTS archive~\cite{dataset} contains 562 clips in eight labels and 25 normalized source-title groups. Its test set has 116 clips from five groups. Nine uniformly sampled frames feed the frozen encoder. Three ordered phase means, late-minus-early difference and temporal standard deviation form 6,400 features. This sparse temporal summary is motivated by segment-based representations~\cite{wang}, without reproducing TSN training.

The full-frame pilot obtains 18.1\% accuracy. A fixed upper-center crop, introduced after pilot test feedback, reaches 30.2\% accuracy and 21.3\% macro F1. The training-majority baseline also reaches 30.2\% accuracy (5.8\% macro F1). Equal majority accuracy is not a successful general video recognizer. Defence has only three test clips; validation has one unorthodox clip. The reused holdout makes this exploratory. Original and revised predictions remain available; different data and labels prevent a direct image-versus-video comparison.

\subsection{Code, data and repositories}
The complete report appends every current custom executable source and configuration, with a SHA-256 inventory. Public libraries, pretrained weights and the unmodified CVPR style are not original code. The project includes dependency files, manifests, numeric heads, saved predictions, tests and reproduction commands. The separate course mapping and viva guide explain lecture references, advanced audit tools and one measured image trace.

"""+repository+r"""

Data: \href{https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset}{Cricket Shots Dataset v2} and \href{https://www.kaggle.com/datasets/amittalmale/cricket-shots}{CRICKET\_SHOTS v1}. Pretrained weights: \href{https://github.com/onnx/models}{ONNX Model Zoo}. Libraries: \href{https://github.com/scikit-learn/scikit-learn}{scikit-learn}, \href{https://github.com/microsoft/onnxruntime}{ONNX Runtime}, \href{https://github.com/opencv/opencv}{OpenCV}, \href{https://github.com/pallets/flask}{Flask}, and \href{https://github.com/numpy/numpy}{NumPy}. Complete dependency links and attribution accompany \texttt{docs/sources.md}. Our NumPy HOG follows the cited method and supplied lectures; no external cricket classifier code was imported.
"""
    (generated/"image_supplement.tex").write_text(supplement,encoding="utf-8")
    for name in ("cvpr.sty","ieee_fullname.bst"):
        shutil.copy2(report/"template/latex"/name,report/name)
    subprocess.run([str(ROOT/"tools/tectonic/tectonic.exe"),"--keep-logs","--outdir",str(build),str(report/"image_main.tex")],cwd=report,check=True)
    shutil.copy2(build/"image_main.pdf",out/"CricketShot_Report_Main.pdf")
    count=code_appendix(out/"CricketShot_Code_Appendix.pdf")
    complete=fitz.open(out/"CricketShot_Report_Main.pdf")
    appendix=fitz.open(out/"CricketShot_Code_Appendix.pdf");complete.insert_pdf(appendix)
    complete.set_metadata({"title":"Cricket Shot Recognition - EE604 course revision", "author":"; ".join(sub["member_names"])})
    complete.save(out/"CricketShot_Report_Complete.pdf",garbage=4,deflate=True)
    with zipfile.ZipFile(ROOT/"output/CricketShot_Overleaf.zip","w",zipfile.ZIP_DEFLATED) as z:
        main=(report/"image_main.tex").read_text()
        main=re.sub(r"\\graphicspath\{.*?\}\n",lambda _:r"\graphicspath{{figures/}}"+"\n",main)
        z.writestr("main.tex",main)
        for name in ("cvpr.sty","ieee_fullname.bst","references.bib"):z.write(report/name,name)
        for path in generated.glob("*.tex"):
            if path.name.startswith("image_") or path.name=="authors.tex":z.write(path,"generated/"+path.name)
        for folder in (ROOT/"results/images/figures",ROOT/"results/revision/figures"):
            for path in folder.iterdir():
                if path.suffix in {".pdf",".png",".jpg"}:z.write(path,"figures/"+path.name)
        z.write(out/"CricketShot_Code_Appendix.pdf","code_appendix.pdf")
        z.write(out/"CricketShot_Report_Complete.pdf","complete_report.pdf")
        z.write(ROOT/"scripts/merge_report.py","merge_report.py")
        z.writestr("README.txt","Compile main.tex for the revised CVPR manuscript and supplementary evidence. The unmodified 2022 style is included.\nThe current complete report and entire source appendix are included. After editing, export main.pdf and run:\npython -m pip install pymupdf\npython merge_report.py main.pdf code_appendix.pdf complete_updated.pdf\nThe PDF merge avoids the 2022 style's embedded eso-pic conflict with pdfpages.\n")
    print(f"Revised manuscript with supplement: {len(complete)-len(appendix)} pages; entire source appendix: {count}; complete: {len(complete)}")


if __name__=="__main__":main()
