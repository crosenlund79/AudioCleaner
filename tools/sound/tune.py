"""Choose per-category thresholds and decide which categories are enabled.

Inputs (produced with score_clips.py / score_long.py from the AudioSet evaluation split, which the
model never saw in training, and from event-free background recordings):

* clip scores  – AudioSet eval clips (10 s, weak labels): 2 s windows every 1 s and the 10 s
                 context score, combined exactly as in the app (``analyze.combine``);
* background   – long speech / meeting / quiet-room recordings without the sounds in question.

Method, per category (scores = max over its AudioSet classes):

1. Split the AudioSet clips 50/50 by video id (A: tuning, B: testing).
2. Clip decision = at least ``min_windows`` windows at or above the threshold (the app's rule).
3. On A: *moderate* = lowest threshold with precision >= P_MOD; *high* = lowest with precision >= P_HIGH
   (vocal sounds: P_VOCAL, stricter, because they occur inside conversation).
4. Raise both until the background material yields at most FP_MOD / FP_HIGH events per hour.
5. Floors: moderate >= 0.12, high >= max(moderate + 0.1, 0.3) (combined scores of sustained sounds
   are lower than those of short sounds, so a higher floor would silence them).
6. On B (untouched): precision and recall at the chosen thresholds. A category is enabled only if
   it has >= 12 test positives, precision >= 0.62 and recall >= 0.25 at the moderate threshold,
   and passes the manual review list below.

usage: tune.py SHORT.npz CONTEXT.npz LABELS.json CANDIDATES.json BG_DIR OUT_CATEGORIES.json OUT_REPORT.json
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

P_MOD, P_HIGH = 0.65, 0.85
# vocal sounds occur inside conversation, which the read-speech background cannot fully represent
# (the meeting recordings showed many more triggers in spontaneous speech): stricter targets
P_VOCAL = (0.70, 0.88)
FP_MOD, FP_HIGH = 0.5, 0.1        # false events per hour of event-free background material
BUDGET_FILES = {"speech_fleurs", "quiet_room"}   # truly event-free; meetings (real laughter) and
                                                 # degraded speech (contains music) are only reported
MIN_POS_B, MIN_PREC_B, MIN_REC_B = 12, 0.62, 0.25

# categories never enabled, whatever the numbers say (reason in ACOUSTIC_ANALYSIS.md)
EXCLUDE = {
    "speech": "speech is covered by the transcription",
    "water": "too generic (any water sound) to be useful",
    "hum": "constant background hum of fridges, mains and fans – would mark almost every hour",
    "baby_cry": "included in 'Crying / sobbing' (one sound must not be reported twice)",
    "pour": "included in 'Running or pouring water'",
    "cutlery": "included in 'Dishes / cutlery'",
}


def runs_ok(win: np.ndarray, thr: float, k: int) -> bool:
    return int((win >= thr).sum()) >= k


def kth(win: np.ndarray, k: int) -> float:
    s = np.sort(win)[::-1]
    return float(s[min(k, len(s)) - 1]) if len(s) else 0.0


def pr_at(scores: np.ndarray, y: np.ndarray, t: float) -> tuple[float, float]:
    pred = scores >= t
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return (tp / (tp + fp) if tp + fp else 1.0), (tp / (tp + fn) if tp + fn else 0.0)


def lowest_threshold(scores: np.ndarray, y: np.ndarray, target: float) -> float | None:
    cands = np.unique(np.round(scores[scores >= 0.05], 3))
    for t in cands:
        p, r = pr_at(scores, y, t)
        if p >= target and (scores >= t).sum() >= 5:
            return float(t)
    return None


def bg_events_per_hour(bg: list[dict], j: int, thr: float, k: int, merge_gap: float) -> float:
    n, hours = 0, 0.0
    for b in bg:
        s = b["scores"][:, j].astype(np.float32)
        st = b["starts"]
        hours += b["duration"] / 3600
        on = s >= 0.5 * thr
        i = 0
        last_end = -1e9
        while i < len(s):
            if on[i]:
                a = i
                while i < len(s) and on[i]:
                    i += 1
                seg = s[a:i]
                if (seg >= thr).sum() >= k:
                    if st[a] - last_end > merge_gap:
                        n += 1
                    last_end = st[i - 1] + 2.0
            else:
                i += 1
    return n / max(hours, 1e-9)


def main():
    clips, ctx_f, labels_f, cand_f, bg_dir, out_cat, out_rep = sys.argv[1:8]
    d = np.load(clips)
    names = list(d["names"])
    nwin = d["nwin"]
    S = d["scores"]
    cx = np.load(ctx_f)
    assert list(cx["names"]) == names
    CTX = cx["clip"].astype(np.float32)
    lab = json.load(open(labels_f))
    cand = json.load(open(cand_f))
    from audiorefine.sound.model import labels as model_labels  # noqa: E402
    idx = {n: i for i, n in enumerate(model_labels())}
    offs = np.concatenate([[0], np.cumsum(nwin)])
    bg = []
    for f in sorted(Path(bg_dir).glob("*.npz")):
        b = np.load(f)
        bg.append({"name": f.stem, "scores": b["scores"], "starts": b["starts"], "duration": float(b["duration"]),
                   "cats": list(b["cats"])})
    bg_hours = sum(b["duration"] for b in bg if b["name"] in BUDGET_FILES) / 3600
    split = np.array([zlib.crc32(n.encode()) % 2 for n in names])
    report = {"params": {"P_MOD": P_MOD, "P_HIGH": P_HIGH, "FP_MOD": FP_MOD, "FP_HIGH": FP_HIGH,
                         "background_hours": round(bg_hours, 2), "clips": len(names)}, "categories": {}}
    out = dict(cand)
    out["categories"] = []
    for c in cand["categories"]:
        cols = [idx[n] for n in c["labels"]]
        k = int(c["min_windows"])
        y = np.array([any(i in cols for i in lab[n]) for n in names])
        win_cat = S[:, cols].astype(np.float32).max(axis=1)
        ctx_cat = CTX[:, cols].max(axis=1)
        if c["kind"] == "media":
            sc = ctx_cat.copy()
        else:
            sc = np.array([kth(np.sqrt(win_cat[offs[i]:offs[i + 1]] * ctx_cat[i]), k) for i in range(len(names))])
        A, B = split == 0, split == 1
        pm_t, ph_t = P_VOCAL if c["kind"] == "vocal" else (P_MOD, P_HIGH)
        t_mod = lowest_threshold(sc[A], y[A], pm_t)
        t_high = lowest_threshold(sc[A], y[A], ph_t)
        rep = {"positives_A": int(y[A].sum()), "positives_B": int(y[B].sum()), "t_mod_A": t_mod, "t_high_A": t_high}
        t_mod = max(0.12, t_mod if t_mod is not None else 0.95)
        t_high = max(t_mod + 0.1, 0.3, t_high if t_high is not None else 0.98)
        # background false-positive budget
        if bg and c["id"] in bg[0]["cats"]:
            j = bg[0]["cats"].index(c["id"])
            budget = [b for b in bg if b["name"] in BUDGET_FILES]
            while t_mod < 0.95 and bg_events_per_hour(budget, j, t_mod, k, c["merge_gap"]) > FP_MOD:
                t_mod = round(t_mod + 0.02, 3)
            while t_high < 0.98 and bg_events_per_hour(budget, j, t_high, k, c["merge_gap"]) > FP_HIGH:
                t_high = round(t_high + 0.02, 3)
            t_high = max(t_high, t_mod + 0.1)
            rep["bg_fp_per_hour_mod"] = round(bg_events_per_hour(budget, j, t_mod, k, c["merge_gap"]), 3)
            rep["bg_fp_per_hour_high"] = round(bg_events_per_hour(budget, j, min(t_high, 0.99), k, c["merge_gap"]), 3)
            rep["bg_fp_by_file_mod"] = {b["name"]: round(bg_events_per_hour([b], j, t_mod, k, c["merge_gap"]), 2) for b in bg}
        t_high = min(t_high, 0.99)
        pm, rm = pr_at(sc[B], y[B], t_mod)
        ph, rh = pr_at(sc[B], y[B], t_high)
        rep.update({"moderate": round(t_mod, 3), "high": round(t_high, 3), "precision_mod_B": round(pm, 3),
                    "recall_mod_B": round(rm, 3), "precision_high_B": round(ph, 3), "recall_high_B": round(rh, 3)})
        reasons = []
        if c["id"] in EXCLUDE:
            reasons.append(EXCLUDE[c["id"]])
        if y[B].sum() < MIN_POS_B:
            reasons.append(f"too few test examples ({int(y[B].sum())})")
        if pm < MIN_PREC_B:
            reasons.append(f"precision {pm:.2f} < {MIN_PREC_B}")
        if rm < MIN_REC_B:
            reasons.append(f"recall {rm:.2f} < {MIN_REC_B}")
        if t_mod >= 0.95:
            reasons.append("no usable threshold")
        enabled = not reasons and c["id"] in {x["id"] for x in cand["categories"]}
        rep["enabled"] = enabled
        rep["reason"] = "; ".join(reasons)
        report["categories"][c["id"]] = rep
        cc = dict(c, moderate=round(t_mod, 3), high=round(t_high, 3), enabled=enabled)
        if not enabled:
            cc["note"] = rep["reason"]
        out["categories"].append(cc)
    json.dump(out, open(out_cat, "w"), indent=1, ensure_ascii=False)
    json.dump(report, open(out_rep, "w"), indent=1)
    en = [c for c in out["categories"] if c["enabled"]]
    print(f"enabled {len(en)} of {len(out['categories'])}")
    for cid, r in report["categories"].items():
        print(f"{cid:14s} {'ON ' if r['enabled'] else 'off'} mod {r['moderate']:.2f} high {r['high']:.2f} "
              f"P/R mod {r['precision_mod_B']:.2f}/{r['recall_mod_B']:.2f} high {r['precision_high_B']:.2f}/"
              f"{r['recall_high_B']:.2f} n={r['positives_B']} bgFP/h {r.get('bg_fp_per_hour_mod', '-')} {r['reason']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()
