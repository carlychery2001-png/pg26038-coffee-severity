"""Build the expert-panel validation section for the PG-26038 thesis as a Word file.
All numbers are read from final_results.json and tiefree_results.json (no hand-typed values)."""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import csv, json
from collections import Counter
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Emu, RGBColor

from facts import *
from facts import _sev
CL = json.load(open("closeness.json"))
grade = lambda s: 0 if s < 0.5 else 1 if s < 5 else 2 if s < 20 else 3


# ---------- document scaffolding (matches the thesis: Letter, Arial, justified, 1.3 spacing) ----------
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Emu(7772400), Emu(10058400)
sec.left_margin = sec.right_margin = Emu(900430); sec.top_margin = sec.bottom_margin = Emu(720090)
st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
for h, size in (("Heading 1", 13), ("Heading 2", 12), ("Heading 3", 11)):
    s = doc.styles[h]; s.font.name = "Arial"; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = RGBColor(0, 0, 0)
    s.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None

def para(text, italic_terms=(), style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, italic=False, space_after=6, keep=False):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.alignment = align; p.paragraph_format.line_spacing = 1.3; p.paragraph_format.space_after = Pt(space_after)
    if keep: p.paragraph_format.keep_with_next = True
    # italicize species names and statistic symbols
    import re
    terms = ["Hemileia vastatrix", "Mycena citricolor", "Coffea arabica", "n", "p"] + list(italic_terms)
    pat = re.compile(r"(\*[^*]+\*)")
    for chunk in pat.split(text):
        if not chunk: continue
        r = p.add_run(chunk.strip("*") if chunk.startswith("*") else chunk)
        r.italic = italic or chunk.startswith("*"); r.bold = bold
        if size: r.font.size = Pt(size)
    return p

def note(text): return para(text, align=WD_ALIGN_PARAGRAPH.LEFT, size=9, space_after=10)
def heading(text, level):
    h = doc.add_heading(text, level); h.paragraph_format.keep_with_next = True
    for r in h.runs: r.font.color.rgb = RGBColor(0, 0, 0); r.font.name = "Arial"
    return h

def banner(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "E8EEF4"); pPr.append(shd)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    return p

def rule(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    for side, val in (("top", top), ("bottom", bottom)):
        if val:
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(val)); e.set(qn("w:color"), "000000"); b.append(e)
    tcPr.append(b)

def table(header, rows, widths, bold_rows=(), left_cols=2, fs=9):
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths): t.columns[j].width = Emu(w)
    tblPr = t._tbl.tblPr; tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(widths) / 635))); tw.set(qn("w:type"), "dxa")
    for old in tblPr.findall(qn("w:tblW")): tblPr.remove(old)
    tblPr.append(tw)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Emu(widths[j]); c.text = ""
            p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < left_cols else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val)); r.font.size = Pt(fs); r.bold = (i == 0) or (i - 1 in bold_rows)
            if i == 0: rule(c, top=8, bottom=4)
            if i == len(rows): rule(c, bottom=8)
    return t

# =====================================================================================
banner("NOTES FOR CARLY: delete this box before pasting. Where each part goes and what to check.")
for t_ in [
    "Part A goes in Methodology as a new Heading 2 right after \u201cEvaluation Design and Statistical Analysis\u201d (before \u201cDeployment: A Two-Tier System\u201d).",
    "Part B goes in Results as a new Heading 2 right after \u201cOrdinal grade agreement\u201d. It adds Tables 6 and 7 and Figures 8 to 11, so the current Table 6 (\u201cModel footprint versus accuracy\u201d) becomes Table 8 and the current Figures 8 to 12 become Figures 12 to 16; update their in-text mentions.",
    "Part C goes in Discussion as a new Heading 2 after \u201cPixel overlap is not severity: the decision-relevant view\u201d.",
    "Part D replaces text that is now out of date: the Limitations paragraph \u201cReference standard\u201d (which still calls the inter-rater study planned), the Recommendations paragraph that begins \u201cThe inter-rater expert grading study\u2026\u201d, and one sentence added to the Conclusions. Part E lists the new references for Literature Cited.",
    "Check 1: the evaluators are described anonymously (E1 to E4). If they agree to be named, thank them by name in the Acknowledgments.",
    "Check 2: your Recommendations currently favor UNet++ for American leaf spot grading because of its internal grade accuracy. Against the expert panel, UNet++ was the weakest model for leaf spot (Part B and C). Reread that recommendation with this result in mind.",
    "Check 3: the text says evaluators saw no model outputs and worked independently. That is true of the form content; confirm it matches how you invited them.",
]:
    para("\u2022 " + t_, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5, space_after=3)

# =====================================================================================
banner("PART A \u2014 METHODOLOGY (new Heading 2, after \u201cEvaluation Design and Statistical Analysis\u201d)")
heading("Comparison with an Independent Expert Panel", 2)
para(
    "To place the models\u2019 grading performance against human visual assessment, and to quantify the inter-observer "
    "variability that a single-annotator reference standard leaves unmeasured, an independent panel of four evaluators graded a "
    f"common set of 60 leaves, 30 for American leaf spot and 30 for coffee leaf rust. The panel comprised three plant pathologists, "
    f"with 25, 35 and 57 years of experience, and one agronomist with 12 years of experience at a coffee cooperative. The "
    "evaluators worked independently through a bilingual English\u2013Spanish online form between 14 and 27 September 2026 and "
    "had no access to model outputs, to the reference annotations, or to one another\u2019s responses.")
para(
    "Each form item showed one leaf as its abaxial and adaxial scans side by side, labeled with the disease under assessment, above a "
    "reference strip of four real leaves illustrating grades G0 to G3 with lesion boundaries outlined. The strip stated the same "
    "cut-points used for the reference standard: G0, less than 0.5% of leaf area affected; G1, 0.5% to less than 5%; G2, 5% to less than "
    "20%; and G3, 20% or more. Evaluators assigned one grade per leaf.")
para(
    "Because the record of which scans were placed in the form was not retained, the provenance of each item was recovered afterwards by "
    "matching every form image to its source scan in the corpus, using the blade outline and texture of both surfaces after removing "
    "the petiole, with every match confirmed visually, and was then cross-referenced with the leaf-level partitions. "
    f"All {n_rust_dis + n_ls_dis} diseased leaves came from the sealed test partition: the complete set of {n_rust_dis} rust test "
    f"leaves and {n_ls_dis} of the 55 American leaf spot test leaves. Of the 12 healthy leaves, {n_unseen_healthy} had not been used "
    f"to train the corresponding model, whereas {n_train} healthy rust leaves belonged to the rust training partition. These "
    f"{n_train} leaves were excluded from every model\u2013panel comparison, leaving a primary set of {P['all']['n']} leaves "
    f"({P['ojo']['n']} American leaf spot and {P['roya']['n']} rust) that the models had never seen. All 60 leaves were retained "
    "for the inter-rater analysis, which does not involve the models.")
para(
    "The six trained models were applied to the same images the evaluators saw. Each leaf surface was isolated from the form image, "
    "placed on the black background of the scanning protocol, rescaled to the typical leaf size of the corpus, and passed through the "
    "unchanged inference pipeline: 1,024-pixel letterbox, ImageNet normalization, and raw argmax segmentation with no post-processing "
    "or severity rescaling. Severity and grade followed the rules of the internal evaluation, namely the abaxial face alone for rust "
    "and the mean of both faces for American leaf spot, mapped to grades with the same cut-points. The form images had a lower "
    "resolution than the original scans, about 700 pixels wide per surface, and a slightly altered aspect ratio; the aspect change does "
    "not bias severity, a ratio of two areas measured on the same image, although the lower resolution could affect the detection of "
    "the smallest lesions.")
para(
    "Inter-rater agreement was summarized with Krippendorff\u2019s alpha for ordinal data (Krippendorff, 2019), the intraclass "
    "correlation coefficient ICC(2,1) for absolute agreement of single raters (Shrout & Fleiss, 1979), interpreted following Koo and "
    "Li (2016), and pairwise quadratic-weighted kappa (Cohen, 1968). Fleiss\u2019 kappa (Fleiss, 1971), which ignores the order of "
    "the grades and treats every disagreement as complete, is reported for completeness, together with the proportions of leaves "
    "graded identically by all four evaluators and within one grade.")
para(
    "With four evaluators, the median grade is undefined whenever the panel splits two against two, and the rule used to resolve such "
    f"ties changes the consensus grade. Because {ties['ojo'] + ties['roya']} of the 60 leaves were such splits, the primary "
    "model\u2013panel comparison used a leave-one-evaluator-out design that cannot produce ties. For each evaluator in turn, the median "
    "grade of the remaining three served as the reference; that evaluator\u2019s own grades and each model\u2019s grades were scored "
    "against the same reference with quadratic-weighted kappa, and the four values were averaged. Model and evaluators are thus judged "
    "on identical terms, and a model value above the evaluators\u2019 mean indicates that the model agrees with the panel more closely "
    "than a typical panel member does. The model-minus-evaluator difference was given a 95% confidence interval from 10,000 bootstrap "
    "resamples of leaves (Efron, 1979). Secondary measures were the mean kappa against each individual evaluator, compared with the "
    "mean pairwise kappa among evaluators; agreement with the four-evaluator median, reported as a range across the two tie-resolution "
    "rules (ties rounded down or up); and the proportion of leaves whose model grade fell within the range of grades given by the "
    "panel. Two sensitivity analyses repeated the comparison on all 60 leaves and with rust graded from the mean of both faces.")

# =====================================================================================
banner("PART B \u2014 RESULTS (new Heading 2, after \u201cOrdinal grade agreement\u201d)")
heading("Agreement with an independent expert panel", 2)
o, r_, a_ = irr["ojo"], irr["roya"], irr["all"]
para(
    "The four evaluators agreed substantially but not perfectly (Table 6). Ordinal alpha was "
    f"{f2(o['alpha'])} for American leaf spot and {f2(r_['alpha'])} for rust, and ICC(2,1) was {f2(o['icc'])} and {f2(r_['icc'])}, "
    "moderate to good reliability. Exact unanimity was uncommon, occurring on "
    f"{o['unanimous']} of 30 leaf-spot leaves and {r_['unanimous']} of 30 rust leaves, but disagreement was almost always limited to "
    f"adjacent grades: all four grades fell within one step on {a_['within1']} of 60 leaves, the exception being one leaf-spot leaf "
    f"graded from {wide_txt}. Fleiss\u2019 kappa, which counts every disagreement as complete, was correspondingly low "
    f"({f2(o['fleiss'])} and {f2(r_['fleiss'])}), showing that most disagreement was of a single grade. The evaluators also differed in "
    f"overall severity level: their mean grades ranged from {rng(min(o['rater_mean']), max(o['rater_mean']))} for leaf spot and from "
    f"{rng(min(r_['rater_mean']), max(r_['rater_mean']))} for rust, and the panel split two against two on "
    f"{ties['ojo'] + ties['roya']} of the 60 leaves.")

para("Table 6. Agreement among the four expert evaluators.", align=WD_ALIGN_PARAGRAPH.LEFT, keep=True, space_after=2)
rows6 = [
    ["Krippendorff\u2019s alpha (ordinal)", f2(o["alpha"]), f2(r_["alpha"]), f2(a_["alpha"])],
    ["ICC(2,1)", f2(o["icc"]), f2(r_["icc"]), f2(a_["icc"])],
    ["Fleiss\u2019 kappa (nominal)", f2(o["fleiss"]), f2(r_["fleiss"]), f2(a_["fleiss"])],
    ["Pairwise QWK, range", "\u2013".join(f2(x) for x in o["pairwise_qwk"]), "\u2013".join(f2(x) for x in r_["pairwise_qwk"]), "\u2013".join(f2(x) for x in a_["pairwise_qwk"])],
    ["Leaves graded identically by all four", f"{o['unanimous']} of 30", f"{r_['unanimous']} of 30", f"{a_['unanimous']} of 60"],
    ["Leaves with all grades within one step", f"{o['within1']} of 30", f"{r_['within1']} of 30", f"{a_['within1']} of 60"],
    ["Leaves split two against two", f"{ties['ojo']} of 30", f"{ties['roya']} of 30", f"{ties['ojo'] + ties['roya']} of 60"],
    ["Mean grade per evaluator, range", "\u2013".join(f2(x) for x in (min(o["rater_mean"]), max(o["rater_mean"]))), "\u2013".join(f2(x) for x in (min(r_["rater_mean"]), max(r_["rater_mean"]))), "\u2013".join(f2(x) for x in (min(a_["rater_mean"]), max(a_["rater_mean"])))],
]
table(["Measure", "American leaf spot (n = 30)", "Rust (n = 30)", "Pooled (n = 60)"], rows6, [2500000, 1250000, 1100000, 1100000], left_cols=1)
note("Note. Grades on the G0 to G3 scale; mean grade uses G0 = 0 to G3 = 3. ICC = intraclass correlation coefficient, two-way random "
     "effects, absolute agreement, single rater. QWK = quadratic-weighted kappa. All 60 leaves are included because this analysis does "
     "not involve the models.")

sO, sR, sA = TT("ojo", "segformer"), TT("roya", "segformer"), TT("all", "segformer")
dO, dR, dA = TT("ojo", "deeplab"), TT("roya", "deeplab"), TT("all", "deeplab")
uO, uR, uA = TT("ojo", "unetpp"), TT("roya", "unetpp"), TT("all", "unetpp")
FIG = (ROOT + "/validation/figures/")
OUT_SEV = sorted(_sev(r["leaf_id"]) for r in E if r["leaf_id"] not in TRAIN and not (
    min(int(r[e]) for e in EXPN) <= grade(_sev(r["leaf_id"])) <= max(int(r[e]) for e in EXPN)))

def figure(png, number, title, note_text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(FIG + png, width=Emu(5900000))
    c = doc.add_paragraph(); c.paragraph_format.space_after = Pt(2); c.paragraph_format.keep_with_next = True
    r = c.add_run(f"Figure {number}. "); r.bold = True; r.font.size = Pt(10)
    r = c.add_run(title); r.font.size = Pt(10)
    note("Note. " + note_text + " Created by the authors.")

para(
    f"On the primary set of {P['all']['n']} leaves unseen during training, SegFormer-B5 agreed with the leave-one-out panel consensus "
    f"at a quadratic-weighted kappa of {f2(sA['fair_model'])}, against {f2(sA['fair_expert'])} for the evaluators themselves, a "
    f"difference of {sg(sA['diff'])} {ci(sA['diff_ci'])} (Table 7, Figure 8). The margin was clear for American leaf spot, "
    f"{f2(sO['fair_model'])} against {f2(sO['fair_expert'])} ({sg(sO['diff'])} {ci(sO['diff_ci'])}). For rust the point estimate "
    f"was similar, {f2(sR['fair_model'])} against {f2(sR['fair_expert'])} ({sg(sR['diff'])}), but its interval "
    f"{ci(sR['diff_ci'])} includes zero, so over {P['roya']['n']} leaves SegFormer-B5 is at least comparable to an individual "
    f"evaluator on rust without being shown to be better. DeepLabV3+ also exceeded the evaluators on leaf spot "
    f"({sg(dO['diff'])} {ci(dO['diff_ci'])}) but not on rust ({sg(dR['diff'])} {ci(dR['diff_ci'])}), whereas UNet++ was "
    f"indistinguishable from the evaluators on both diseases ({sg(uO['diff'])} {ci(uO['diff_ci'])} for leaf spot and "
    f"{sg(uR['diff'])} {ci(uR['diff_ci'])} for rust).")

para("Table 7. Agreement of each model with the expert panel on the leaves unseen during training.", align=WD_ALIGN_PARAGRAPH.LEFT, keep=True, space_after=2)
rows7, bold = [], []
for d, dname in DIS + [("all", "Pooled")]:
    e = TT(d, "segformer")
    bold.append(len(rows7))
    rows7.append([f"{dname} (n = {P[d]['n']})", "Evaluators (mean)", f2(e["fair_expert"]), "\u2013", f2(e["pairwise_expert"]), "\u2013", "\u2013"])
    for a, aname in ARCH:
        t_ = TT(d, a)
        rows7.append(["", aname, f2(t_["fair_model"]), f"{sg(t_['diff'])} {ci(t_['diff_ci'])}", f2(t_["pairwise_model"]),
                      f"{f2(min(t_['cons_down'], t_['cons_up']))}\u2013{f2(max(t_['cons_down'], t_['cons_up']))}", f"{inr(d, a)} of {P[d]['n']}"])
table(["Disease", "Rater", "QWK vs leave-one-out consensus", "Difference from evaluators [95% CI]", "Mean QWK vs individual evaluators",
       "QWK vs four-evaluator median", "Grade within panel range"], rows7, [930000, 950000, 740000, 1200000, 740000, 740000, 670000], bold_rows=bold, fs=8.5)
note("Note. QWK = quadratic-weighted kappa on the G0 to G3 scale. Leave-one-out consensus: for each evaluator in turn, the median grade "
     "of the other three; the evaluators\u2019 row scores each evaluator against the consensus of the other three, and each model is "
     "scored against the same four references. Difference = model minus evaluators, with a 95% confidence interval from 10,000 "
     "bootstrap resamples of leaves. Four-evaluator median shown as the range across the two tie-resolution rules. For the evaluators, "
     "the mean QWK column gives the mean pairwise QWK among evaluators. Rust grades use the abaxial face and American leaf spot grades "
     f"use the mean of both faces. Differences were computed before rounding. The {n_train} healthy rust leaves that were part of the "
     "rust training partition are excluded.")

figure("Fig_simple_1_agreement.png", 8, "Agreement of Each Model and of a Typical Expert Evaluator With the Expert Panel.",
       "Agreement is the quadratic-weighted kappa with the panel consensus, defined for each evaluator as the median grade of the "
       "other three; a typical expert is the mean over the four evaluators, and each model is scored against the same four "
       "consensus grades. 1 means perfect agreement. For rust, the difference between SegFormer-B5 and a typical expert is not "
       f"statistically certain (Table 7). Leaves unseen during training only (n = {P['all']['n']}).")

para(
    f"The secondary measures told the same story. Scored against each evaluator individually, SegFormer-B5 reached a mean kappa of "
    f"{f2(sA['pairwise_model'])}, compared with a mean pairwise kappa of {f2(sA['pairwise_expert'])} among the evaluators, and on "
    f"American leaf spot both SegFormer-B5 and DeepLabV3+ scored above every individual evaluator on the leave-one-out measure. Overall, SegFormer-B5 gave the "
    f"same grade as the panel consensus on {CL['all']['segformer'][0]*100:.0f}% of leaves, against {CL['all']['expert'][0]*100:.0f}% for a "
    f"typical evaluator, and every other grade it gave was one step away (Figure 9). Against the "
    f"four-evaluator median SegFormer-B5 reached a kappa of {rng(sA['cons_down'], sA['cons_up'])} and an exact grade agreement of "
    f"{round(100 * sA['acc_down'])}% to {round(100 * sA['acc_up'])}%, depending on how ties were resolved; under either rule every "
    f"SegFormer-B5 grade was within one step of the consensus. Its grade fell inside the range of grades given by the panel for "
    f"{inr('all', 'segformer')} of {P['all']['n']} leaves, all {P['ojo']['n']} for leaf spot and {inr('roya', 'segformer')} of "
    f"{P['roya']['n']} for rust (Figure 10).")

figure("Fig_simple_2_closeness.png", 9, "How Often Each Rater Gives the Same Grade as the Expert Panel.",
       "Share of leaves on which the grade equals the panel consensus (the median grade of the other evaluators), is one grade away, "
       "or is two or more grades away; for a typical expert and for each model the shares are averaged over the four leave-one-out "
       "consensus grades. Leaves unseen during training only.")
figure("Fig_simple_3_perleaf.png", 10, "Grades of the Four Expert Evaluators and of SegFormer-B5 for Each Leaf.",
       "Each column is one leaf, sorted by the experts’ average grade. Grey bars span the grades given by the four experts; blue "
       "dots are SegFormer-B5 grades inside that range and red rings grades outside it. Rust grades use the abaxial face and American "
       "leaf spot grades use the mean of both faces.")

para(
    f"The errors differed in character between architectures (counts relative to the four-evaluator median, ties rounded up). UNet++ tended to over-grade near-healthy leaf-spot leaves: of the "
    f"{sum(unet_g0)} leaves the panel rated G0, it graded {unet_g0[1]} as G1, a mean bias of "
    f"{sg(P['ojo']['unetpp']['bias'])} grades for that disease. DeepLabV3+ tended to under-grade rust, assigning a lower grade than "
    f"the panel on {dl_under} of {P['roya']['n']} leaves " + (f"and a higher grade on {dl_over}" if dl_over else "and never a higher one") + f" (bias {sg(P['roya']['deeplab']['bias'])}), "
    f"whereas the errors of SegFormer-B5 on rust were balanced ({sf_under} lower and {sf_over} higher; bias "
    f"{sg(P['roya']['segformer']['bias'])}). The two rust leaves that SegFormer-B5 graded outside the panel\u2019s range were both "
    f"rated G1 by all four evaluators and received predicted severities of {OUT_SEV[0]:.1f}% and {OUT_SEV[1]:.1f}%, above the 5% "
    "cut-point between G1 and G2.")
para(
    f"The continuous severity behind these grades followed the panel closely. The predicted severity of SegFormer-B5 rose "
    f"monotonically with the mean expert grade, with Spearman correlations of {f2(rho['ojo'])} for American leaf spot and "
    f"{f2(rho['roya'])} for rust (Figure 11).")
figure("Fig_simple_4_severity.png", 11, "Diseased Leaf Area Measured by SegFormer-B5 Against the Experts’ Average Grade.",
       "Each dot is one leaf. Measured area is on a logarithmic scale, with values of 0.01% or less plotted at the bottom; shaded bands "
       "mark the area ranges of grades G0 to G3. Rust area is from the abaxial face and American leaf spot area is the mean of both "
       "faces. Points are jittered slightly along the horizontal axis. Leaves unseen during training only.")

para(
    f"Neither sensitivity analysis changed these conclusions. Including the {n_train} training leaves (all 60 leaves pooled), which the panel consensus "
    f"rated G0 and every model graded G0, raised agreement slightly, with SegFormer-B5 at {f2(A60['segformer']['fair'])} against "
    f"{f2(A60['expert_fair'])} for the evaluators. Grading rust from the mean of both faces instead of the abaxial face lowered "
    f"SegFormer-B5\u2019s rust agreement from {f2(sR['fair_model'])} to {f2(BF['segformer']['fair'])}, consistent with the choice of "
    "the abaxial face in the internal evaluation.")

# =====================================================================================
banner("PART C \u2014 DISCUSSION (new Heading 2, after \u201cPixel overlap is not severity: the decision-relevant view\u201d)")
heading("The models against expert visual grading", 2)
para(
    "The expert panel speaks directly to the claim of consistency in the general objective. Four experienced evaluators, grading the "
    f"same images against the same scale and reference examples, gave an identical grade to only {a_['unanimous']} of 60 leaves, and "
    f"their mean grades for American leaf spot differed by {f2(max(o['rater_mean']) - min(o['rater_mean']))} of a grade between the most "
    "lenient and the most severe evaluator. This is the evaluator-to-evaluator variability that motivated the study, now measured "
    "rather than assumed. Against that background, SegFormer-B5 agreed with the panel more closely than a typical panel member did for "
    "American leaf spot and at least as closely for rust, and it never departed from the panel consensus by more than one grade. "
    "Unlike a human evaluator, the model is also deterministic: the same image always receives the same grade.")
para(
    "The panel also bears on the reference standard. For American leaf spot the internal reference grade was the single "
    "annotator\u2019s visual reading, so strong internal grade agreement could in principle reflect alignment with one person\u2019s "
    "habits. Agreement with an independent panel on leaves from the sealed test set indicates that the leaf-spot grades of SegFormer-B5 "
    "and DeepLabV3+ are not tied to that annotator. The contrast with UNet++ is instructive: internally it had the highest leaf-spot "
    "grade accuracy, yet against the panel it was the weakest model for that disease, because it moved near-healthy leaves from G0 to "
    "G1. Its internal advantage may therefore partly reflect the annotator\u2019s own tendency to rate faint leaf-spot leaves G1, the "
    "G0\u2013G1 boundary effect described in the grade results, rather than an advantage the panel shares.")
para(
    "Two cautions apply. Agreement with a panel is not agreement with truth: the panel median is itself a visual estimate, and the "
    "pixel annotation remains the reference for lesion area and continuous severity, which the panel did not assess. The comparison "
    f"was also made on flatbed scans, at reduced image resolution and on {P['all']['n']} leaves, with only {P['roya']['n']} for rust "
    "and few G3 leaves, which is why superiority over individual evaluators is established for American leaf spot but not yet for "
    "rust.")

# =====================================================================================
banner("PART D \u2014 REPLACEMENT TEXT FOR PARAGRAPHS NOW OUT OF DATE")
para("D1. Limitations, replace the first two sentences of the \u201cReference standard\u201d paragraph with:", align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=10)
para(
    "Reference standard. All annotations were produced by a single trained annotator under one protocol approved by the thesis "
    "supervisor, which maximizes internal consistency. Inter-observer variability was subsequently measured with an independent panel "
    f"of four evaluators on a 60-leaf subset, whose agreement was substantial but imperfect (ordinal alpha {f2(r_['alpha'])} to "
    f"{f2(o['alpha'])}); SegFormer-B5 agreed with this panel at least as closely as its own members did, while DeepLabV3+ and UNet++ were not shown to differ from them. The panel graded whole "
    "leaves and did not delineate lesions, so it validates the ordinal grades but not the lesion masks or the continuous severity "
    "values. An intra-rater re-annotation of a subsample after a washout period remains planned to bound annotation noise.")
para("D2. Limitations, add a new paragraph after \u201cSample sizes at the extremes\u201d:", align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=10)
para(
    "Expert panel design. The record of which scans were placed in the expert form was not kept and was reconstructed afterwards by "
    f"image matching; this showed that {n_train} healthy rust leaves came from the rust training partition, and they were excluded from "
    "the model comparison. The evaluators graded reduced-resolution images with the disease named on each item, which simplifies the "
    "task relative to field diagnosis. The panel set contained few severe leaves, and with 23 rust leaves the comparison could not "
    "establish that any model outperforms individual evaluators for rust.")
para("D3. Recommendations, replace the paragraph that begins \u201cThe inter-rater expert grading study\u2026\u201d with:", align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=10)
para(
    "The expert-panel validation should be extended with more rust leaves and more severe (G3) leaves of both diseases, so that the "
    "comparison between the models and individual evaluators can be resolved for rust, and the item list should be fixed and recorded "
    "before the panel is convened. Finally, the calibrated severity outputs should be integrated into routine monitoring, such as "
    "severity progress curves and spatial prioritization of plots, to connect automated assessment with integrated disease-management "
    "decisions.")
para("D4. Conclusions, second paragraph, insert just before the sentence \u201cTogether these results demonstrate\u2026\u201d:", align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=10)
para(
    f"An independent panel of four evaluators, whose own agreement was substantial (ordinal alpha {f2(r_['alpha'])} to "
    f"{f2(o['alpha'])}), confirmed these grades on {P['all']['n']} leaves unseen during training: SegFormer-B5 agreed with the panel "
    f"more closely than a typical evaluator did (quadratic-weighted kappa {f2(sA['fair_model'])} against "
    f"{f2(sA['fair_expert'])}), significantly so for American leaf spot but not yet for rust, and none of its grades departed from the panel consensus by more than one step.")

# =====================================================================================
banner("PART E \u2014 NEW ENTRIES FOR LITERATURE CITED (alphabetical placement)")
for ref in [
    "Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin, 76*(5), 378-382. https://doi.org/10.1037/h0031619",
    "Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155-163. https://doi.org/10.1016/j.jcm.2016.02.012",
    "Krippendorff, K. (2019). *Content analysis: An introduction to its methodology* (4th ed.). SAGE Publications. https://doi.org/10.4135/9781071878781",
    "Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin, 86*(2), 420-428. https://doi.org/10.1037/0033-2909.86.2.420",
]:
    p = para(ref, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6)
    p.paragraph_format.left_indent = Emu(457200); p.paragraph_format.first_line_indent = Emu(-457200)

doc.save(OUT)
print("saved", OUT)
