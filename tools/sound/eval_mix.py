"""Recall under difficult conditions (ESC-50 events mixed into conversation / TV / quiet room; see
make_mixtures.py) and false detections on the backgrounds alone, with the shipped thresholds and
the application's decision rule.

usage: eval_mix.py SHORT.npz CONTEXT.npz ITEMS.json CATEGORIES.json"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audiorefine.sound.model import labels  # noqa: E402

sh, cx, items_f, cat_f = sys.argv[1:5]
d = np.load(sh)
names = list(d["names"])
nwin = d["nwin"]
S = d["scores"].astype(np.float32)
offs = np.concatenate([[0], np.cumsum(nwin)])
c = np.load(cx)
assert list(c["names"]) == names
C = c["clip"].astype(np.float32)
items = {it["name"]: it for it in json.load(open(items_f))}
cats = json.load(open(cat_f))["categories"]
L = {n: i for i, n in enumerate(labels())}
en = [x for x in cats if x["enabled"]]
cols = {x["id"]: [L[n] for n in x["labels"]] for x in en}


def detected(i, cat):
    x = next(z for z in en if z["id"] == cat)
    w = S[offs[i]:offs[i + 1]][:, cols[cat]].max(1)
    ctx = C[i, cols[cat]].max()
    comb = np.full_like(w, ctx) if x["kind"] == "media" else np.sqrt(w * ctx)
    return int((comb >= x["moderate"]).sum()) >= x["min_windows"], bool((comb >= x["high"]).any())


rec = defaultdict(lambda: [0, 0])
per_cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
fp = defaultdict(lambda: [0, 0, defaultdict(int)])
for i, n in enumerate(names):
    it = items[n]
    if it["cat"] is None:
        key = it["bg"]
        fp[key][1] += 1
        for x in en:
            if x["kind"] == "media" and key == "tv":
                continue                     # music/TV in the TV background is correct
            if detected(i, x["id"])[0]:
                fp[key][0] += 1
                fp[key][2][x["id"]] += 1
        continue
    if it["cat"] not in cols:
        continue
    cond = f"{it['bg']} {it['cond']}"
    ok = detected(i, it["cat"])[0]
    rec[cond][0] += ok
    rec[cond][1] += 1
    per_cat[it["cat"]][cond][0] += ok
    per_cat[it["cat"]][cond][1] += 1
order = ["none clean", "quiet -20dB", "quiet distant", "speech 0dB", "speech -10dB", "speech -20dB", "speech distant",
         "tv 0dB", "tv -10dB"]
print("| Condition | Detected (enabled categories) |\n|---|---|")
for k in order:
    if k in rec:
        a, b = rec[k]
        print(f"| {k} | {a}/{b} = {a / b:.0%} |")
print("\n| Category | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
for cat in sorted(per_cat):
    row = []
    for k in order:
        a, b = per_cat[cat].get(k, [0, 0])
        row.append(f"{a}/{b}" if b else "–")
    print(f"| {cat} | " + " | ".join(row) + " |")
print("\nBackground only (10 s clips):")
for k, (a, b, which) in fp.items():
    print(f"  {k}: {a} false detections in {b} clips ({b * 10 / 3600:.2f} h) – {dict(which)}")
