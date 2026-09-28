"""Supplementary check of the qualitative description (manuscript TODO: "re-check this qualitative description"):
where are SegFormer-B5 lesion errors located, over ALL test faces that enter the leaf-level metrics?

Faces: rust = abaxial face of the 21 diseased test leaves; leaf spot = both faces of the 55 test leaves (110 faces).
For every false-positive pixel (predicted lesion, not reference lesion) the Euclidean distance to the nearest
reference-lesion pixel is computed in letterbox space; for every false-negative pixel the distance to the nearest
predicted-lesion pixel. Errors within 2 px (and 5 px) are "at a lesion boundary"; the rest are separate blobs or
missed lesions. Also re-verifies, for every face, that the rebuilt reference and the saved prediction reproduce the
per_image.csv pixel counts. Output: error_location.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import distance_transform_edt

import common as C

ARCH = "segformer"


def main():
    rows = C.read_rows()
    out = {"model": C.MODEL_NAME[ARCH], "distance_unit": "letterbox px (1024 x 1024 canvas)",
           "rust_letterbox_scale": "rust pack images are 791 x 1024 px, so letterbox px = pack px",
           "leaf_spot_letterbox_scale": "leaf-spot pack images are 1583 x 2048 px, so 1 letterbox px = 2 pack px",
           "diseases": {}}
    for d in C.DISEASES:
        tab = C.leaf_table(rows, d, ARCH)
        faces = [r["faces"]["abaxial"]["file"] for r in tab] if d == "roya" else \
                [r["faces"][f]["file"] for r in tab for f in ("abaxial", "adaxial")]
        tot = dict(tp=0, fp=0, fn=0, fp_le2=0, fp_le5=0, fn_le2=0, fn_le5=0)
        per_face = []
        for f in faces:
            rgb, lab = C.load_letterbox(d, f)
            pred = C.load_pred(d, ARCH, d, "test", f)
            r = [x for x in rows if x["model_disease"] == d and x["arch"] == ARCH and x["image_pack"] == d
                 and x["split"] == "test" and x["file"] == f][0]
            ref, pr = lab == 2, pred == 2
            assert int((lab >= 1).sum()) == int(r["ref_leaf_px"]) and int(ref.sum()) == int(r["ref_lesion_px"])
            assert int((pred >= 1).sum()) == int(r["pred_leaf_px"]) and int(pr.sum()) == int(r["pred_lesion_px"])
            assert int((ref & pr).sum()) == int(r["inter_lesion"]) and int((ref | pr).sum()) == int(r["union_lesion"])
            fp, fn = pr & ~ref, ref & ~pr
            d_ref = distance_transform_edt(~ref) if ref.any() else np.full(ref.shape, np.inf)
            d_pr = distance_transform_edt(~pr) if pr.any() else np.full(pr.shape, np.inf)
            c = dict(tp=int((ref & pr).sum()), fp=int(fp.sum()), fn=int(fn.sum()),
                     fp_le2=int((fp & (d_ref <= 2)).sum()), fp_le5=int((fp & (d_ref <= 5)).sum()),
                     fn_le2=int((fn & (d_pr <= 2)).sum()), fn_le5=int((fn & (d_pr <= 5)).sum()))
            for k in tot:
                tot[k] += c[k]
            per_face.append(dict(file=f, **c))
        s = dict(n_faces=len(faces), **tot,
                 fp_over_fn_ratio=tot["fp"] / tot["fn"],
                 share_fp_within_2px_of_reference_lesion=tot["fp_le2"] / tot["fp"],
                 share_fp_within_5px_of_reference_lesion=tot["fp_le5"] / tot["fp"],
                 share_fn_within_2px_of_predicted_lesion=tot["fn_le2"] / tot["fn"],
                 share_fn_within_5px_of_predicted_lesion=tot["fn_le5"] / tot["fn"],
                 faces_with_fp_gt_fn=int(sum(p["fp"] > p["fn"] for p in per_face)))
        out["diseases"][d] = dict(summary=s, per_face=per_face)
        print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    (C.OUT / "error_location.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
