"""Recompute the analytical `prescribe` blocks of the E9 records with the current pruner/cost model.

Prescriptions are pure analysis (no training, no measurement), so refreshing them after a tooling change does
not touch any measured number in the record. Used after the pruner learned to find groups on the graph.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from common import *
from npuloop.intake import prescriptions, alternatives

res = Results("e9_customer_intake")
for r in res.data["records"]:
    m = load_model(r["model"])
    for key, spec in (("intake", r["intake"]["spec"]), ("intake_strict", r["intake_strict"]["spec"])):
        r[key]["prescribe"] = prescriptions(m, spec)
        r[key]["alternatives"] = alternatives(m, spec)
    r["alternatives"] = r["intake"]["alternatives"]
    log(f"{r['model']}: " + " | ".join(f"{p['kind']}: {p['action'][:40]} saving {p['saving'] if p['saving'] is None else round(p['saving'], 3)}" for p in r["intake"]["prescribe"]))
res.save()
