"""Merge results/<name>_<suffix>.json records into results/<name>.json (used for parallel experiment lanes)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def merge(name, suffix):
    dst = os.path.join(ROOT, "results", f"{name}.json"); src = os.path.join(ROOT, "results", f"{name}_{suffix}.json")
    if not os.path.exists(src):
        print("nothing to merge"); return
    d = json.load(open(dst)) if os.path.exists(dst) else {"meta": {}, "records": []}
    s = json.load(open(src))
    keys = {(r.get("model"), r.get("strategy"), r.get("ratio"), r.get("align")) for r in d["records"]}
    added = 0
    for r in s["records"]:
        k = (r.get("model"), r.get("strategy"), r.get("ratio"), r.get("align"))
        if k not in keys:
            d["records"].append(r); keys.add(k); added += 1
    d["meta"].update({k: v for k, v in s["meta"].items() if k not in d["meta"]})
    json.dump(d, open(dst, "w"), indent=1)
    os.remove(src)
    print(f"merged {added} records into {dst}")

if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
