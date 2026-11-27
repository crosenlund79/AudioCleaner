"""Markdown tables of the tuned vocabulary for ACOUSTIC_ANALYSIS.md.
usage: report_tables.py CATEGORIES.json TUNE_REPORT.json"""
import json
import sys

cats = json.load(open(sys.argv[1]))["categories"]
rep = json.load(open(sys.argv[2]))
groups = {}
for c in cats:
    groups.setdefault(c["group"], []).append(c)
print("### Enabled categories\n")
print("| Group | Category | AudioSet classes | Moderate / High threshold | Precision / recall at moderate (test half) | n | False detections per hour of speech + quiet room |")
print("|---|---|---|---|---|---|---|")
for g, cs in groups.items():
    for c in cs:
        if not c["enabled"]:
            continue
        r = rep["categories"][c["id"]]
        print(f"| {g} | {c['display']} | {', '.join(c['labels'])} | {c['moderate']:.2f} / {c['high']:.2f} | "
              f"{r['precision_mod_B']:.2f} / {r['recall_mod_B']:.2f} | {r['positives_B']} | {r.get('bg_fp_per_hour_mod', '–')} |")
print("\n### Investigated but disabled\n")
print("| Group | Category | Reason (test half) |")
print("|---|---|---|")
for g, cs in groups.items():
    for c in cs:
        if c["enabled"]:
            continue
        r = rep["categories"][c["id"]]
        print(f"| {g} | {c['display']} | {r['reason']} (P {r['precision_mod_B']:.2f}, R {r['recall_mod_B']:.2f}, n {r['positives_B']}) |")
