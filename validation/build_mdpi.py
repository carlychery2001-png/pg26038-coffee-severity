"""Build the MDPI (Agriculture) LaTeX insert for the expert-panel validation, plus a compilable preview.
All numbers come from facts.py (same source as the thesis section)."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import os, re, shutil
from facts import *
from facts import _sev
CL = json.load(open("closeness.json"))

grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3
OUTDIR = _os.environ.get("PG26038_OUTDIR", "build_mdpi_out")  # (release) output folder
os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
for f in os.listdir("figures"):
    if f.startswith("Fig_simple_") and f.endswith(".pdf"): shutil.copy(os.path.join("figures", f), os.path.join(OUTDIR, "figures", f))

# ---- LaTeX number formatting ----
def m(x, sign=False):
    x = round(x, 2) + 0.0
    s = f"{abs(x):.2f}"
    if x < 0: return f"$-${s}"
    return ("+" if sign else "") + s
ci = lambda c: f"[{m(c[0])}, {m(c[1])}]"
sO, sR, sA = T["ojo_segformer"], T["roya_segformer"], T["all_segformer"]
dO, dR = T["ojo_deeplab"], T["roya_deeplab"]
uO, uR = T["ojo_unetpp"], T["roya_unetpp"]
o, r_, a_ = irr["ojo"], irr["roya"], irr["all"]
OUT_SEV = sorted(_sev(r["leaf_id"]) for r in E if r["leaf_id"] not in TRAIN and not (
    min(int(r[e]) for e in EXPN) <= grade(_sev(r["leaf_id"])) <= max(int(r[e]) for e in EXPN)))
V = dict(
    NLS=n_ls_dis, NR=n_rust_dis, NDIS=n_ls_dis + n_rust_dis, NUH=n_unseen_healthy, NTR=n_train, NP=P["all"]["n"], NPO=P["ojo"]["n"], NPR=P["roya"]["n"],
    TIES=ties["ojo"] + ties["roya"], TIESO=ties["ojo"], TIESR=ties["roya"],
    AO=m(o["alpha"]), AR=m(r_["alpha"]), AA=m(a_["alpha"]), IO=m(o["icc"]), IR=m(r_["icc"]), IA=m(a_["icc"]),
    FO=m(o["fleiss"]), FR=m(r_["fleiss"]), FA=m(a_["fleiss"]),
    UO=o["unanimous"], UR=r_["unanimous"], UA=a_["unanimous"], WO=o["within1"], WR=r_["within1"], WA=a_["within1"], WIDE=wide_txt,
    PWO="--".join(m(x) for x in o["pairwise_qwk"]), PWR="--".join(m(x) for x in r_["pairwise_qwk"]), PWA="--".join(m(x) for x in a_["pairwise_qwk"]),
    RMO="--".join(m(x) for x in (min(o["rater_mean"]), max(o["rater_mean"]))), RMR="--".join(m(x) for x in (min(r_["rater_mean"]), max(r_["rater_mean"]))),
    RMA="--".join(m(x) for x in (min(a_["rater_mean"]), max(a_["rater_mean"]))),
    RMO_T=f"{m(min(o['rater_mean']))} to {m(max(o['rater_mean']))}", RMR_T=f"{m(min(r_['rater_mean']))} to {m(max(r_['rater_mean']))}",
    SAF=m(sA["fair_model"]), SAE=m(sA["fair_expert"]), SAD=m(sA["diff"], True), SADC=ci(sA["diff_ci"]),
    SOF=m(sO["fair_model"]), SOE=m(sO["fair_expert"]), SOD=m(sO["diff"], True), SODC=ci(sO["diff_ci"]),
    SRF=m(sR["fair_model"]), SRE=m(sR["fair_expert"]), SRD=m(sR["diff"], True), SRDC=ci(sR["diff_ci"]),
    DOD=m(dO["diff"], True), DODC=ci(dO["diff_ci"]), DRD=m(dR["diff"], True), DRDC=ci(dR["diff_ci"]),
    UOD=m(uO["diff"], True), UODC=ci(uO["diff_ci"]), URD=m(uR["diff"], True), URDC=ci(uR["diff_ci"]),
    SPM=m(sA["pairwise_model"]), SPE=m(sA["pairwise_expert"]), SCL=m(min(sA["cons_down"], sA["cons_up"])), SCH=m(max(sA["cons_down"], sA["cons_up"])),
    SACL=round(100 * min(sA["acc_down"], sA["acc_up"])), SACH=round(100 * max(sA["acc_down"], sA["acc_up"])),
    INA=inr("all", "segformer"), INR=inr("roya", "segformer"),
    UG0=sum(unet_g0), UG1=unet_g0[1], UBIAS=m(P["ojo"]["unetpp"]["bias"], True),
    DLU=dl_under, DLO_T=(f"and a higher grade on {dl_over}" if dl_over else "and never a higher one"), DBIAS=m(P["roya"]["deeplab"]["bias"], True),
    SFU=sf_under, SFO=sf_over, SBIAS=m(P["roya"]["segformer"]["bias"], True),
    OS1=f"{OUT_SEV[0]:.1f}", OS2=f"{OUT_SEV[1]:.1f}", RHO_O=m(rho["ojo"]), RHO_R=m(rho["roya"]),
    A60S=m(A60["segformer"]["fair"]), A60E=m(A60["expert_fair"]), BFS=m(BF["segformer"]["fair"]),
    SPREAD=m(max(o["rater_mean"]) - min(o["rater_mean"])),
    ERL=m(F["primary_ojo"]["expert_fair_range"][0]), ERH=m(F["primary_ojo"]["expert_fair_range"][1]),
    CLS=f"{CL['all']['segformer'][0]*100:.0f}", CLE=f"{CL['all']['expert'][0]*100:.0f}",
)

rows7 = []
for d, dname in (("ojo", "American leaf spot"), ("roya", "Coffee leaf rust"), ("all", "Pooled")):
    e = T[f"{d}_segformer"]
    rows7.append(f"\\multirow{{4}}{{*}}{{\\makecell[l]{{{dname}\\\\(\\textit{{n}} = {P[d]['n']})}}}} & Evaluators (mean) & {m(e['fair_expert'])} & -- & {m(e['pairwise_expert'])} & -- & -- \\\\")
    for a, aname in ARCH:
        t_ = T[f"{d}_{a}"]
        lo_, hi_ = t_["diff_ci"]
        rows7.append(f" & {aname} & {m(t_['fair_model'])} & \\makecell{{{m(t_['diff'], True)}\\\\{{[}}{m(lo_)}, {m(hi_)}{{]}}}} & {m(t_['pairwise_model'])} & "
                     f"{m(min(t_['cons_down'], t_['cons_up']))}--{m(max(t_['cons_down'], t_['cons_up']))} & {inr(d, a)}/{P[d]['n']} \\\\")
    rows7.append("\\midrule" if d != "all" else "")
V["ROWS7"] = "\n".join(x for x in rows7 if x)

TEX = r"""% =====================================================================================
%  EXPERT-PANEL VALIDATION -- insert for the MDPI Agriculture manuscript (PG-26038)
%  Generated from the analysis outputs in <ROOT>\validation (build_mdpi.py).
%
%  WHERE EACH BLOCK GOES
%   Block A  -> new Section 2.8, after 2.7 "Evaluation Design and Statistical Analysis"
%               (current 2.8 Deployment and 2.9 Reproducibility become 2.9 and 2.10)
%   Block B  -> new Section 3.5, after 3.4 "Ordinal grade agreement"
%               (current 3.5-3.7 become 3.6-3.8). Adds two tables and four figures;
%               with \ref labels the numbering of later tables/figures updates itself.
%   Block C  -> new Section 4.3, after 4.2 "Pixel overlap is not severity"
%   Block D  -> replacement text (Limitations, Conclusions, Abstract) and back matter
%   Block E  -> four \bibitem entries; place them right after the Cohen (1968) entry,
%               because they are first cited in Section 2.8 (before the current [49]).
%
%  PACKAGES: the MDPI class already loads booktabs, tabularx, float, graphicx; add
%  \usepackage{multirow,makecell} to the preamble if they are not there yet.
%  CITATION KEYS: \cite{efron1979} and \cite{cohen1968} must match the keys you already
%  use for Efron (1979) and Cohen (1968) -- rename them below if yours differ.
%  FIGURES: copy the four PDFs from the figures/ folder next to your .tex file.
% =====================================================================================


% ------------------------------- BLOCK A: METHODS ------------------------------------
\subsection{Comparison with an Independent Expert Panel}\label{sec:expert_methods}

To place the models' grading against human visual assessment, and to quantify the inter-observer variability that a single-annotator reference standard leaves unmeasured, an independent panel of four evaluators graded a common set of 60 leaves, 30 for American leaf spot and 30 for coffee leaf rust. The panel comprised three plant pathologists with 25, 35 and 57 years of experience and one agronomist with 12 years of experience at a coffee cooperative. The evaluators worked independently through a bilingual English--Spanish online form between 14 and 27 September 2026, without access to model outputs, reference annotations or one another's responses. Each item showed the abaxial and adaxial scans of one leaf, labelled with the disease under assessment, above a reference strip of four real leaves illustrating grades G0 to G3 with lesion boundaries outlined and the cut-points of the reference standard stated; evaluators assigned one grade per leaf.

Because the list of scans placed in the form was not retained, the provenance of every item was recovered by matching each form image to its source scan (blade outline and texture of both surfaces after petiole removal, every match confirmed visually) and cross-referenced with the leaf-level partitions. All @@NDIS@@ diseased leaves came from the sealed test partition: the complete set of @@NR@@ rust test leaves and @@NLS@@ of the 55 American leaf spot test leaves. Of the 12 healthy leaves, @@NUH@@ had not been used to train the corresponding model and @@NTR@@ belonged to the rust training partition; these @@NTR@@ were excluded from every model--panel comparison, leaving a primary set of @@NP@@ unseen leaves (@@NPO@@ American leaf spot, @@NPR@@ rust). All 60 leaves were retained for the inter-rater analysis, which does not involve the models.

The six models were applied to the same images the evaluators saw. Each surface was isolated, placed on the black background of the scanning protocol, rescaled to the typical leaf size of the corpus and passed through the unchanged inference pipeline (1024-pixel letterbox, ImageNet normalization, raw argmax, no post-processing or severity rescaling). Severity and grade followed the internal evaluation: abaxial face only for rust and the mean of both faces for American leaf spot, with the same cut-points. The form images had lower resolution than the original scans (about 700 pixels wide per surface) and a slightly altered aspect ratio; the aspect change does not bias severity, a ratio of two areas measured on the same image, although the lower resolution could affect the detection of the smallest lesions.

Inter-rater agreement was summarized with Krippendorff's $\alpha$ for ordinal data~\cite{krippendorff2019}, the intraclass correlation coefficient ICC(2,1) for absolute agreement of single raters~\cite{shrout1979}, interpreted following Koo and Li~\cite{koo2016}, and pairwise quadratic-weighted kappa~\cite{cohen1968}; Fleiss' $\kappa$~\cite{fleiss1971}, which ignores grade order, is reported for completeness. With four evaluators the median grade is undefined whenever the panel splits two against two, and @@TIES@@ of the 60 leaves were such splits. The primary model--panel comparison therefore used a leave-one-evaluator-out design that cannot tie: for each evaluator in turn, the median grade of the other three served as reference, and that evaluator's grades and each model's grades were scored against it with quadratic-weighted kappa, averaging the four values. A model value above the evaluators' mean thus indicates closer agreement with the panel than a typical panel member achieves. The model-minus-evaluator difference was given a 95\% confidence interval from 10,000 bootstrap resamples of leaves~\cite{efron1979}. Secondary measures were the mean kappa against individual evaluators, agreement with the four-evaluator median under both tie-resolution rules (reported as a range), and the proportion of leaves whose model grade fell within the range of panel grades. Sensitivity analyses repeated the comparison on all 60 leaves and with rust graded from both faces.


% ------------------------------- BLOCK B: RESULTS ------------------------------------
\subsection{Agreement with an independent expert panel}\label{sec:expert_results}

The four evaluators (Section~\ref{sec:expert_methods}) agreed substantially but not perfectly (Table~\ref{tab:expert_irr}). Ordinal $\alpha$ was @@AO@@ for American leaf spot and @@AR@@ for rust, and ICC(2,1) was @@IO@@ and @@IR@@, moderate to good reliability. Exact unanimity was uncommon (@@UO@@ of 30 leaf-spot and @@UR@@ of 30 rust leaves), but disagreement was almost always limited to adjacent grades: all four grades fell within one step on @@WA@@ of 60 leaves, the exception being one leaf-spot leaf graded from @@WIDE@@. Accordingly, Fleiss' $\kappa$, which counts every disagreement as complete, was low (@@FO@@ and @@FR@@). The evaluators also differed in overall severity level, with mean grades from @@RMO_T@@ for leaf spot and from @@RMR_T@@ for rust.

\begin{table}[H]
\begin{adjustwidth}{-\extralength}{0cm}
\caption{Agreement among the four expert evaluators on all 60 leaves.\label{tab:expert_irr}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCC}
\toprule
\textbf{Measure} & \textbf{American Leaf Spot (\textit{n} = 30)} & \textbf{Rust (\textit{n} = 30)} & \textbf{Pooled (\textit{n} = 60)} \\
\midrule
Krippendorff's $\alpha$ (ordinal) & @@AO@@ & @@AR@@ & @@AA@@ \\
ICC(2,1) & @@IO@@ & @@IR@@ & @@IA@@ \\
Fleiss' $\kappa$ (nominal) & @@FO@@ & @@FR@@ & @@FA@@ \\
Pairwise QWK, range & @@PWO@@ & @@PWR@@ & @@PWA@@ \\
Leaves graded identically by all four & @@UO@@/30 & @@UR@@/30 & @@UA@@/60 \\
Leaves with all grades within one step & @@WO@@/30 & @@WR@@/30 & @@WA@@/60 \\
Leaves split two against two & @@TIESO@@/30 & @@TIESR@@/30 & @@TIES@@/60 \\
Mean grade per evaluator, range & @@RMO@@ & @@RMR@@ & @@RMA@@ \\
\bottomrule
\end{tabularx}
\noindent{\footnotesize{Grades on the G0--G3 scale (mean grade with G0 = 0 to G3 = 3). ICC = intraclass correlation coefficient, two-way random effects, absolute agreement, single rater. QWK = quadratic-weighted kappa.}}
\end{adjustwidth}
\end{table}

On the @@NP@@ leaves unseen during training, SegFormer-B5 agreed with the leave-one-out panel consensus at a quadratic-weighted kappa of @@SAF@@, against @@SAE@@ for the evaluators themselves, a difference of @@SAD@@ @@SADC@@ (Table~\ref{tab:expert_models}, Figure~\ref{fig:expert_forest}). The margin was clear for American leaf spot, @@SOF@@ against @@SOE@@ (@@SOD@@ @@SODC@@). For rust the point estimate was similar, @@SRF@@ against @@SRE@@ (@@SRD@@), but its interval @@SRDC@@ includes zero, so over @@NPR@@ leaves SegFormer-B5 is at least comparable to an individual evaluator on rust without being shown to be better. DeepLabV3+ also exceeded the evaluators on leaf spot (@@DOD@@ @@DODC@@) but not on rust (@@DRD@@ @@DRDC@@), whereas UNet++ was indistinguishable from them on both diseases (@@UOD@@ @@UODC@@ and @@URD@@ @@URDC@@).

\begin{table}[H]
\begin{adjustwidth}{-\extralength}{0cm}
\caption{Agreement of each model with the expert panel on the @@NP@@ leaves unseen during training.\label{tab:expert_models}}
\footnotesize
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{p{2.6cm}p{2.6cm}CcCCC}
\toprule
\textbf{Disease} & \textbf{Rater} & \textbf{QWK vs LOO Consensus} & \textbf{Difference [95\% CI]} & \textbf{Mean QWK vs Each Evaluator} & \textbf{QWK vs Panel Median} & \textbf{Grade Within Panel Range} \\
\midrule
@@ROWS7@@
\bottomrule
\end{tabularx}
\noindent{\footnotesize{QWK = quadratic-weighted kappa; LOO = leave-one-evaluator-out consensus (median of the other three evaluators). The evaluators' row scores each evaluator against the consensus of the other three; each model is scored against the same four references. Difference = model minus evaluators, 95\% CI from 10,000 bootstrap resamples of leaves. Panel median = median grade of all four evaluators, shown as the range across the two tie-resolution rules. For the evaluators, the mean-QWK column is the mean pairwise QWK among evaluators. Rust grades use the abaxial face and American leaf spot grades use the mean of both faces. Differences were computed before rounding. The @@NTR@@ healthy rust leaves from the training partition are excluded.}}
\end{adjustwidth}
\end{table}

\begin{figure}[H]
\includegraphics[width=13.5cm]{figures/Fig_simple_1_agreement.pdf}
\caption{Agreement of each model and of a typical expert evaluator with the expert panel. Agreement is the quadratic-weighted kappa with the panel consensus (for each evaluator, the median grade of the other three); a typical expert is the mean over the four evaluators, and each model is scored against the same four consensus grades. For rust, the difference between SegFormer-B5 and a typical expert is not statistically certain (Table~\ref{tab:expert_models}). Leaves unseen during training only (\textit{n} = @@NP@@). Created by the authors.\label{fig:expert_forest}}
\end{figure}

The secondary measures told the same story. Scored against each evaluator individually, SegFormer-B5 reached a mean kappa of @@SPM@@, against a mean pairwise kappa of @@SPE@@ among the evaluators, and on American leaf spot both SegFormer-B5 and DeepLabV3+ scored above every individual evaluator on the leave-one-out measure (evaluators @@ERL@@--@@ERH@@). Overall, SegFormer-B5 gave the same grade as the panel consensus on @@CLS@@\% of leaves, against @@CLE@@\% for a typical evaluator, and every other grade it gave was one step away (Figure~\ref{fig:expert_raters}). Against the four-evaluator median SegFormer-B5 reached a kappa of @@SCL@@ to @@SCH@@ and an exact agreement of @@SACL@@\% to @@SACH@@\%, depending on tie resolution; under either rule every SegFormer-B5 grade was within one step of the consensus. Its grade fell inside the range of panel grades for @@INA@@ of @@NP@@ leaves, all @@NPO@@ for leaf spot and @@INR@@ of @@NPR@@ for rust (Figure~\ref{fig:expert_perleaf}).

\begin{figure}[H]
\includegraphics[width=13.5cm]{figures/Fig_simple_2_closeness.pdf}
\caption{How often each rater gives the same grade as the expert panel: share of leaves on which the grade equals the panel consensus (median grade of the other evaluators), is one grade away, or is two or more grades away, averaged over the four leave-one-out consensus grades. Leaves unseen during training only. Created by the authors.\label{fig:expert_raters}}
\end{figure}

\begin{figure}[H]
\includegraphics[width=13.5cm]{figures/Fig_simple_3_perleaf.pdf}
\caption{Grades of the four expert evaluators and of SegFormer-B5 for each leaf. Each column is one leaf, sorted by the experts' average grade. Grey bars span the four expert grades; blue dots are SegFormer-B5 grades inside that range and red rings grades outside it. Created by the authors.\label{fig:expert_perleaf}}
\end{figure}

The errors differed in character between architectures (counts relative to the four-evaluator median, ties rounded up). UNet++ over-graded near-healthy leaf-spot leaves: of the @@UG0@@ leaves the panel rated G0, it graded @@UG1@@ as G1 (mean bias @@UBIAS@@ grades). DeepLabV3+ under-graded rust, assigning a lower grade than the panel on @@DLU@@ of @@NPR@@ leaves @@DLO_T@@ (bias @@DBIAS@@), whereas SegFormer-B5 errors on rust were balanced (@@SFU@@ lower, @@SFO@@ higher; bias @@SBIAS@@). The two rust leaves that SegFormer-B5 graded outside the panel's range were rated G1 by all four evaluators and received predicted severities of @@OS1@@\% and @@OS2@@\%, above the 5\% G1/G2 cut-point. The continuous severity behind the grades followed the panel closely, rising monotonically with the mean expert grade (Spearman $\rho$ = @@RHO_O@@ for American leaf spot and @@RHO_R@@ for rust; Figure~\ref{fig:expert_severity}). Neither sensitivity analysis changed these conclusions: including the @@NTR@@ training leaves (all 60 leaves pooled) raised SegFormer-B5 agreement to @@A60S@@ (evaluators @@A60E@@), and grading rust from both faces lowered it from @@SRF@@ to @@BFS@@, consistent with the abaxial-face rule of the internal evaluation.

\begin{figure}[H]
\includegraphics[width=13.5cm]{figures/Fig_simple_4_severity.pdf}
\caption{Diseased leaf area measured by SegFormer-B5 against the experts' average grade. Each dot is one leaf; logarithmic vertical axis (values $\leq$0.01\% plotted at the bottom); shaded bands mark the area ranges of grades G0 to G3. Rust area from the abaxial face, American leaf spot area from the mean of both faces; points jittered slightly along the horizontal axis. Leaves unseen during training only. Created by the authors.\label{fig:expert_severity}}
\end{figure}


% ------------------------------- BLOCK C: DISCUSSION ---------------------------------
\subsection{The models against expert visual grading}\label{sec:expert_discussion}

The expert panel speaks directly to the claim of consistency. Four experienced evaluators, grading the same images against the same scale and reference examples, gave an identical grade to only @@UA@@ of 60 leaves, and their mean grades for American leaf spot differed by @@SPREAD@@ of a grade between the most lenient and the most severe evaluator: the evaluator-to-evaluator variability that motivated this work, now measured rather than assumed. Against that background, SegFormer-B5 agreed with the panel more closely than a typical panel member for American leaf spot and at least as closely for rust, never departing from the panel consensus by more than one grade, and, unlike a human evaluator, it always returns the same grade for the same image.

The panel also bears on the reference standard. For American leaf spot the internal reference grade was a single annotator's visual reading, so strong internal agreement could in principle reflect alignment with one person's habits; agreement with an independent panel on sealed test leaves indicates that the leaf-spot grades of SegFormer-B5 and DeepLabV3+ are not tied to that annotator. The contrast with UNet++ is instructive: internally it had the highest leaf-spot grade accuracy, yet against the panel it was the weakest model for that disease because it moved near-healthy leaves from G0 to G1, which suggests that part of its internal advantage reflected the annotator's tendency to rate faint leaf-spot leaves G1 (Section~3.4). Two cautions apply. A panel median is itself a visual estimate, and the pixel annotation remains the reference for lesion area and continuous severity, which the panel did not assess; and with @@NPR@@ rust leaves and few G3 leaves, superiority over individual evaluators is established for American leaf spot but not yet for rust.


% --------------------- BLOCK D: REPLACEMENT TEXT AND BACK MATTER ---------------------
% D1. Limitations (Section 4.5), "Reference standard": replace the sentences that call the
%     inter-rater study planned with:
%
% Inter-observer variability was subsequently measured with an independent panel of four evaluators on a 60-leaf subset (Section~\ref{sec:expert_results}), whose agreement was substantial but imperfect (ordinal $\alpha$ = @@AR@@--@@AO@@); SegFormer-B5 agreed with this panel at least as closely as its own members did, while DeepLabV3+ and UNet++ were not shown to differ from them. The panel graded whole leaves and did not delineate lesions, so it validates the ordinal grades but not the lesion masks or continuous severity. An intra-rater re-annotation after a washout period remains planned.
%
% D2. Limitations, new sentence(s) for the sample-size paragraph:
%
% The expert form's item list was reconstructed afterwards by image matching, which identified @@NTR@@ healthy rust training leaves that were then excluded; evaluators graded reduced-resolution images with the disease named on each item, and with @@NPR@@ rust leaves the comparison could not show that any model outperforms individual evaluators for rust.
%
% D3. Conclusions, insert before "Together these results demonstrate...":
%
% An independent panel of four evaluators, whose own agreement was substantial (ordinal $\alpha$ = @@AR@@--@@AO@@), confirmed these grades on @@NP@@ leaves unseen during training: SegFormer-B5 agreed with the panel more closely than a typical evaluator did (quadratic-weighted kappa @@SAF@@ vs. @@SAE@@), significantly so for American leaf spot but not yet for rust, and none of its grades departed from the panel consensus by more than one step.
%
% D4. Abstract (optional; check the 200-word limit), one sentence before the final sentence:
%
% Against an independent panel of four evaluators, SegFormer-B5 agreed with the panel consensus more closely than the evaluators agreed with one another (quadratic-weighted kappa @@SAF@@ vs. @@SAE@@ on @@NP@@ held-out leaves).
%
% D5. Back matter -- CONFIRM each statement is true before using it:
% \authorcontributions{... add "validation, C.C.; formal analysis, C.C." ...}
% \informedconsent{Informed consent was obtained from all expert evaluators involved in the study.}
% \institutionalreview{[State whether the grading study was reviewed or exempted by the EARTH University committee.]}
% \acknowledgments{The authors thank the four expert evaluators [names, if they agree to be named] for grading the validation leaves.}


% ------------------------------- BLOCK E: REFERENCES ---------------------------------
% Place right after the Cohen (1968) \bibitem, in this order of first citation:
\bibitem[Krippendorff(2019)]{krippendorff2019}
Krippendorff, K. \textit{Content Analysis: An Introduction to Its Methodology}, 4th ed.; SAGE Publications: Thousand Oaks, CA, USA, 2019. https://doi.org/10.4135/9781071878781.

\bibitem[Shrout and Fleiss(1979)]{shrout1979}
Shrout, P.E.; Fleiss, J.L. Intraclass correlations: Uses in assessing rater reliability. \textit{Psychol. Bull.} \textbf{1979}, \textit{86}, 420--428. https://doi.org/10.1037/0033-2909.86.2.420.

\bibitem[Koo and Li(2016)]{koo2016}
Koo, T.K.; Li, M.Y. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. \textit{J. Chiropr. Med.} \textbf{2016}, \textit{15}, 155--163. https://doi.org/10.1016/j.jcm.2016.02.012.

\bibitem[Fleiss(1971)]{fleiss1971}
Fleiss, J.L. Measuring nominal scale agreement among many raters. \textit{Psychol. Bull.} \textbf{1971}, \textit{76}, 378--382. https://doi.org/10.1037/h0031619.
"""
for k, v in V.items(): TEX = TEX.replace(f"@@{k}@@", str(v))
left = re.findall(r"@@\w+@@", TEX); assert not left, left
open(os.path.join(OUTDIR, "expert_panel_mdpi.tex"), "w", encoding="utf-8").write(TEX)

# ---- preview wrapper (not for submission): compiles the blocks with a generic class ----
body = TEX.split("% ------------------------------- BLOCK A: METHODS")[1]
body = "% BLOCK A" + body.split("% --------------------- BLOCK D")[0]
bib = TEX.split("% ------------------------------- BLOCK E: REFERENCES")[1]
PREVIEW = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,tabularx,float,multirow,makecell,array,amsmath}
\usepackage[T1]{fontenc}\usepackage{newpxtext,newpxmath}
\setcounter{section}{3}\setcounter{subsection}{4}
\begin{document}
\noindent\textbf{PREVIEW of the expert-panel insert (generic layout, not the MDPI class).}
""" + body + r"""
\begin{thebibliography}{99}
\bibitem{efron1979} Efron, B. Bootstrap methods: Another look at the jackknife. \textit{Ann. Stat.} \textbf{1979}, \textit{7}, 1--26.
\bibitem{cohen1968} Cohen, J. Weighted kappa. \textit{Psychol. Bull.} \textbf{1968}, \textit{70}, 213--220.
""" + bib + r"""
\end{thebibliography}
\end{document}
"""
open(os.path.join(OUTDIR, "preview_expert_panel.tex"), "w", encoding="utf-8").write(PREVIEW)
print("wrote", OUTDIR)
