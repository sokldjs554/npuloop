"""E3: does the static lint predict INT8 damage?

Layer level: per-layer weight-range disparity (static, data-free) vs measured per-layer sensitivity to
per-tensor weight quantization (from E2). Model level: quant-robustness score vs measured PTQ drop.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def _rank(v):
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    v = np.asarray(v, float); order = v.argsort(kind="mergesort"); ranks = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    return ranks


def spearman(a, b):
    if len(a) <= 2:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def main():
    e1 = Results("e1_baselines").data["records"]; e2 = Results("e2_ptq_grid").data["records"]
    res = Results("e3_lint_vs_drop", meta=dict(source=["e1_baselines", "e2_ptq_grid"]))
    res.data["records"] = []
    layer_points = []
    for r1 in e1:
        for scheme in ["per-tensor", "npu-default"]:
            r2 = next((r for r in e2 if r["model"] == r1["model"] and r["scheme"] == scheme and "sensitivity" in r), None)
            if not r2:
                continue
            ratios = r1["weight_range_ratio"]
            for layer, s in r2["sensitivity"]["weights"].items():
                key = layer.replace(".", "_")
                if key in ratios:
                    layer_points.append(dict(model=r1["model"], scheme=scheme, layer=layer, range_ratio=ratios[key], dloss=s["dloss"], dacc=s["dacc"], sqnr_db=s["sqnr_db"]))
    model_points = []
    for r1 in e1:
        for r2 in e2:
            if r2["model"] == r1["model"]:
                model_points.append(dict(model=r1["model"], scheme=r2["scheme"], quant_robustness=r1["lint"]["edge-10tops"]["scores"]["quant_robustness"],
                                         efficiency=r1["lint"]["edge-10tops"]["scores"]["efficiency"], drop_fake=r2["drop_fake"], drop_int=r2["drop_int"]))
    summary = {}
    for scheme in ["per-tensor", "npu-default"]:
        pts = [p for p in layer_points if p["scheme"] == scheme]
        if pts:
            summary[f"layer_spearman_range_vs_dloss_{scheme}"] = spearman([p["range_ratio"] for p in pts], [p["dloss"] for p in pts])
            summary[f"layer_spearman_range_vs_sqnr_{scheme}"] = spearman([p["range_ratio"] for p in pts], [-p["sqnr_db"] for p in pts])
            summary[f"n_layers_{scheme}"] = len(pts)
        mp = [p for p in model_points if p["scheme"] == scheme]
        if len(mp) > 2:
            summary[f"model_spearman_robustness_vs_drop_{scheme}"] = spearman([-p["quant_robustness"] for p in mp], [p["drop_fake"] for p in mp])
    res.data["meta"]["summary"] = summary
    res.data["records"] = []
    for p in layer_points:
        res.add(dict(kind="layer", **p), provenance="measured+simulated")
    for p in model_points:
        res.add(dict(kind="model", **p), provenance="measured+simulated")
    res.save()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
