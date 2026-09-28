"""Round 4, analysis A4 part 2: the deployment-only rust severity rescaling.

What the manuscript says (Supplement note_methods.tex, 'Deployment-only post-processing'): for rust only, the deployed
application rescales the REFINED-lesion severity with s_cal = 1.1915 s + 0.4329 (clipped to 0-100); the coefficients
were fitted on the 21 rust test leaves. The refined pipeline (BiRefNet leaf mask eroded 12 px, lesion probability
>= 0.50, removal of components < 20 px, closing, redness-guided region grow with a per-leaf gate at the 60th
percentile of leaf redness) and the fitting script are NOT present in any searched location (see results.md), and
no per-leaf refined severities are stored anywhere, so the published coefficients cannot be reproduced or refitted
on their own input.

What this script does instead (the only validation-only refit the local data permit):
  * Input: published models' RAW-ARGMAX rust severities (reanalysis/out/per_image.csv), abaxial face (leaf rule).
  * PRIMARY: SegFormer-B5. Fit reference severity = a * predicted severity + b by ordinary least squares on the 21
    diseased rust VALIDATION leaves only (the same functional form as s_cal). t-based and bootstrap (validation
    leaves, 10,000 resamples) 95% CIs for a and b.
  * Apply s_cal = clip(a s + b, 0, 100) to the 21 diseased rust TEST leaves and the 21 local healthy test leaves.
    Test MAE, bias, RMSE, grade accuracy (Wilson), QWK (bootstrap), McNemar raw vs rescaled; paired bootstrap of the
    change (10,000 resamples of test leaves).
  * Sensitivity: validation fit including the 21 healthy validation leaves (n = 42); a through-origin fit (s_cal = a s);
    the same refit for the published UNet++ and DeepLabV3+.
  * Illustration only (NOT recommended): the published deployment coefficients applied to raw-argmax outputs (they were
    fitted for a different input), and an in-sample fit on the test leaves (to show the optimism of test fitting).
Bootstrap: numpy default_rng(20260928); one validation index matrix and one test index matrix (drawn in that order),
shared by all models and variants.
"""
from __future__ import annotations
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))).replace(chr(92), "/")

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
RE = Path((ROOT + "/reanalysis"))
CSV_PUB = RE / "out/per_image.csv"
PAIRED_PY = RE / "results/paired/paired_analysis.py"
SEED = 20260928
B = 10_000
PUB_A, PUB_B = 1.1915, 0.4329  # Supplement note_methods.tex; fitted on the 21 rust test leaves (refined pipeline)

spec = importlib.util.spec_from_file_location("paired_analysis", PAIRED_PY)
PA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PA)
grade, qwk, wilson, pct_ci, mcnemar_exact = PA.grade, PA.qwk, PA.wilson, PA.pct_ci, PA.mcnemar_exact

ARCHS = ["segformer", "unetpp", "deeplab"]
ARCH_NAME = {"segformer": "SegFormer-B5", "unetpp": "UNet++", "deeplab": "DeepLabV3+"}
PUB_TEST = {"segformer": (0.99, 0.99), "unetpp": (1.77, 1.75), "deeplab": (1.47, 1.41)}  # MAE, bias (Table 4)


def load():
    rows = list(csv.DictReader(open(CSV_PUB, newline="")))
    D = {}
    for a in ARCHS:
        for split in ["val", "test"]:
            sel = [r for r in rows if r["model_disease"] == "roya" and r["arch"] == a and r["image_pack"] == "roya"
                   and r["split"] == split]
            by = defaultdict(dict)
            for r in sel:
                by[r["leaf_id"]][r["face"]] = r
            dis, hea = {}, {}
            for lid, f in sorted(by.items()):
                assert set(f) == {"abaxial", "adaxial"}, lid
                ab = f["abaxial"]
                rec = dict(ref=float(ab["ref_sev"]), pred=float(ab["pred_sev"]),
                           pred_either_max=max(float(f["abaxial"]["pred_sev"]), float(f["adaxial"]["pred_sev"])),
                           pred_adaxial=float(f["adaxial"]["pred_sev"]))
                (dis if ab["grade_label"] == "roya" else hea)[lid] = rec
            D[(a, split)] = dict(dis=dis, hea=hea)
    return D


def ols(x, y):
    r = stats.linregress(x, y)
    n = len(x)
    tq = stats.t.ppf(0.975, n - 2)
    resid = y - (r.slope * x + r.intercept)
    return dict(a=float(r.slope), b=float(r.intercept),
                a_ci95_t=[float(r.slope - tq * r.stderr), float(r.slope + tq * r.stderr)],
                b_ci95_t=[float(r.intercept - tq * r.intercept_stderr), float(r.intercept + tq * r.intercept_stderr)],
                r2=float(r.rvalue ** 2), resid_sd=float(np.std(resid, ddof=2)), n=n)


def ols_boot(x, y, idx):
    xb, yb = x[idx], y[idx]
    mx, my = xb.mean(1, keepdims=True), yb.mean(1, keepdims=True)
    sxx = ((xb - mx) ** 2).sum(1)
    a = ((xb - mx) * (yb - my)).sum(1) / sxx
    b = my[:, 0] - a * mx[:, 0]
    return a, b


def origin_fit(x, y):
    return float((x * y).sum() / (x * x).sum())


def cal(s, a, b):
    return np.clip(a * s + b, 0.0, 100.0)


def evaluate(ref, raw, calv, idx, hea_raw=None, hea_cal=None, hea_either_raw=None, hea_either_cal=None):
    """Test-set metrics before and after rescaling; paired bootstrap over test leaves."""
    out = {}
    for tag, p in (("raw", raw), ("rescaled", calv)):
        d = p - ref
        g, gr = grade(p), grade(ref)
        corr = g == gr
        k = int(corr.sum())
        out[tag] = dict(mae_pp=float(np.abs(d).mean()), mae_ci95=pct_ci(np.abs(d)[idx].mean(1)),
                        bias_pp=float(d.mean()), bias_ci95=pct_ci(d[idx].mean(1)),
                        rmse_pp=float(np.sqrt((d ** 2).mean())),
                        sd_diff_pp=float(np.std(d, ddof=1)),
                        loa_pp=[float(d.mean() - 1.96 * np.std(d, ddof=1)), float(d.mean() + 1.96 * np.std(d, ddof=1))],
                        grade_correct=k, n=len(ref), grade_accuracy=k / len(ref), grade_accuracy_wilson95=wilson(k, len(ref)),
                        qwk=float(qwk(g, gr)), qwk_ci95=pct_ci(qwk(g[idx], gr[idx])),
                        over_graded=int((g > gr).sum()), under_graded=int((g < gr).sum()),
                        confusion_rows_reference_cols_predicted=[[int(((gr == i) & (g == j)).sum()) for j in range(4)]
                                                                 for i in range(4)])
    d_raw, d_cal = np.abs(raw - ref), np.abs(calv - ref)
    out["delta_rescaled_minus_raw"] = dict(
        d_mae_pp=float(d_cal.mean() - d_raw.mean()), d_mae_ci95=pct_ci((d_cal - d_raw)[idx].mean(1)),
        d_abs_bias_pp=float(abs((calv - ref).mean()) - abs((raw - ref).mean())),
        d_abs_bias_ci95=pct_ci(np.abs((calv - ref)[idx].mean(1)) - np.abs((raw - ref)[idx].mean(1))),
        leaves_closer_after=int((d_cal < d_raw).sum()), leaves_farther_after=int((d_cal > d_raw).sum()))
    gr = grade(ref)
    out["mcnemar_rescaled_vs_raw"] = mcnemar_exact(grade(calv) == gr, grade(raw) == gr)
    if hea_raw is not None:
        out["healthy_test"] = dict(
            n=len(hea_raw),
            ge_0p5_abaxial_raw=int((hea_raw >= 0.5).sum()), ge_0p5_abaxial_rescaled=int((hea_cal >= 0.5).sum()),
            ge_0p5_either_face_raw=int((hea_either_raw >= 0.5).sum()),
            ge_0p5_either_face_rescaled=int((hea_either_cal >= 0.5).sum()),
            max_abaxial_raw=float(hea_raw.max()), max_abaxial_rescaled=float(hea_cal.max()),
            min_abaxial_rescaled=float(hea_cal.min()),
            wilson95_fp_rate_abaxial_rescaled=wilson(int((hea_cal >= 0.5).sum()), len(hea_cal)))
    return out


def main():
    D = load()
    ids_val = list(D[("segformer", "val")]["dis"])
    ids_test = list(D[("segformer", "test")]["dis"])
    hv = list(D[("segformer", "val")]["hea"])
    ht = list(D[("segformer", "test")]["hea"])
    assert len(ids_val) == 21 and len(ids_test) == 21 and len(hv) == 21 and len(ht) == 21
    rng = np.random.default_rng(SEED)
    idx_val = rng.integers(0, 21, size=(B, 21))
    idx_val42 = rng.integers(0, 42, size=(B, 42))
    idx_test = rng.integers(0, 21, size=(B, 21))

    res = dict(description=__doc__.strip().splitlines()[0], source=str(CSV_PUB),
               published_deployment_rescaling=dict(a=PUB_A, b=PUB_B, input="refined-lesion severity of the deployed "
                                                   "pipeline (not stored locally)", fitted_on="21 rust test leaves "
                                                   "(manuscript text)", reproducible_from_local_files=False),
               bootstrap=dict(resamples=B, seed=SEED, order="validation 21-leaf matrix, validation 42-leaf matrix, "
                              "test 21-leaf matrix"),
               leaf_rule="rust abaxial face; healthy leaves also reported under the either-face (max) rule",
               n=dict(val_diseased=21, val_healthy=21, test_diseased=21, test_healthy_local=21, test_healthy_paper=54),
               models={})
    for a in ARCHS:
        V, T = D[(a, "val")], D[(a, "test")]
        xv = np.array([V["dis"][i]["pred"] for i in ids_val])
        yv = np.array([V["dis"][i]["ref"] for i in ids_val])
        xt = np.array([T["dis"][i]["pred"] for i in ids_test])
        yt = np.array([T["dis"][i]["ref"] for i in ids_test])
        xh = np.array([T["hea"][i]["pred"] for i in ht])
        xh_e = np.array([T["hea"][i]["pred_either_max"] for i in ht])
        xh_ad = np.array([T["hea"][i]["pred_adaxial"] for i in ht])
        xvh = np.array([V["hea"][i]["pred"] for i in hv])
        yvh = np.array([V["hea"][i]["ref"] for i in hv])
        assert np.all(yvh == 0)
        # reproduction of the published raw test MAE and bias
        m0 = dict(mae=round(float(np.abs(xt - yt).mean()), 2), bias=round(float((xt - yt).mean()), 2))
        assert (m0["mae"], m0["bias"]) == PUB_TEST[a], (a, m0)
        e = dict(name=ARCH_NAME[a], raw_test_reproduces_published=True,
                 validation_raw=dict(mae_pp=float(np.abs(xv - yv).mean()), bias_pp=float((xv - yv).mean()),
                                     calibration_slope_pred_on_ref=float(stats.linregress(yv, xv).slope),
                                     calibration_intercept_pred_on_ref=float(stats.linregress(yv, xv).intercept),
                                     healthy_val_max_abaxial=float(xvh.max())),
                 test_raw_calibration_slope_pred_on_ref=float(stats.linregress(yt, xt).slope),
                 fits={})

        def helper(tag, aa, bb, extra):
            ev = evaluate(yt, xt, cal(xt, aa, bb), idx_test, xh, cal(xh, aa, bb), xh_e,
                          np.maximum(cal(xh, aa, bb), cal(xh_ad, aa, bb)))
            e["fits"][tag] = dict(a=aa, b=bb, **extra, test=ev)

        # PRIMARY: OLS on the 21 diseased validation leaves
        f = ols(xv, yv)
        ab, bb_ = ols_boot(xv, yv, idx_val)
        f["a_ci95_boot"], f["b_ci95_boot"] = pct_ci(ab), pct_ci(bb_)
        helper("val_ols_diseased", f["a"], f["b"], dict(fit=f, fitted_on="21 diseased rust validation leaves"))
        # sensitivity: include the 21 healthy validation leaves
        x42, y42 = np.concatenate([xv, xvh]), np.concatenate([yv, yvh])
        f2 = ols(x42, y42)
        ab2, bb2 = ols_boot(x42, y42, idx_val42)
        f2["a_ci95_boot"], f2["b_ci95_boot"] = pct_ci(ab2), pct_ci(bb2)
        helper("val_ols_diseased_plus_healthy", f2["a"], f2["b"],
               dict(fit=f2, fitted_on="21 diseased + 21 healthy rust validation leaves"))
        # sensitivity: through the origin
        a0 = origin_fit(xv, yv)
        a0b = (xv[idx_val] * yv[idx_val]).sum(1) / (xv[idx_val] ** 2).sum(1)
        helper("val_origin_diseased", a0, 0.0, dict(fit=dict(a=a0, a_ci95_boot=pct_ci(a0b), n=21),
                                                    fitted_on="21 diseased rust validation leaves, no intercept"))
        # illustration only
        helper("published_coefficients_on_raw_ILLUSTRATION", PUB_A, PUB_B,
               dict(fitted_on="published deployment coefficients (fitted for the refined pipeline on test leaves); "
                              "applied to raw-argmax outputs for illustration only"))
        ft = ols(xt, yt)
        helper("test_ols_IN_SAMPLE_ILLUSTRATION", ft["a"], ft["b"],
               dict(fit=ft, fitted_on="the 21 diseased rust TEST leaves (in-sample; shows optimism only)"))
        res["models"][a] = e

    (OUT / "rescale_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for a in ARCHS:
        e = res["models"][a]
        print(f"== {a}: val raw MAE {e['validation_raw']['mae_pp']:.3f} bias {e['validation_raw']['bias_pp']:+.3f} "
              f"cal slope {e['validation_raw']['calibration_slope_pred_on_ref']:.3f}")
        for tag, f in e["fits"].items():
            t = f["test"]
            h = t["healthy_test"]
            print(f"  {tag:45s} a={f['a']:.4f} b={f['b']:+.4f} | MAE {t['raw']['mae_pp']:.3f}->{t['rescaled']['mae_pp']:.3f} "
                  f"(d {t['delta_rescaled_minus_raw']['d_mae_pp']:+.3f} {np.round(t['delta_rescaled_minus_raw']['d_mae_ci95'], 3)}) "
                  f"bias {t['raw']['bias_pp']:+.3f}->{t['rescaled']['bias_pp']:+.3f} | acc {t['raw']['grade_correct']}->"
                  f"{t['rescaled']['grade_correct']} QWK {t['raw']['qwk']:.3f}->{t['rescaled']['qwk']:.3f} | healthy>=0.5 "
                  f"ab {h['ge_0p5_abaxial_raw']}->{h['ge_0p5_abaxial_rescaled']} either {h['ge_0p5_either_face_raw']}->"
                  f"{h['ge_0p5_either_face_rescaled']}")
    return res


if __name__ == "__main__":
    main()
