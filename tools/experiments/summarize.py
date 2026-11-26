"""Turn run_eval.py JSONL results into Markdown tables (used for TEST_REPORT.md).

Usage: python summarize.py results_dir > tables.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

COND_ORDER = ["clean", "quiet", "babble_5db", "traffic_0db", "reverb_far", "distance_mix", "music_10db", "hum_rumble", "bangs"]
COND_LABEL = {
    "clean": "clean", "quiet": "quiet (−58 dBFS speech)", "babble_5db": "babble 5 dB SNR",
    "traffic_0db": "traffic 0 dB SNR", "reverb_far": "far + reverb (RT60 0.8 s)", "distance_mix": "speakers 20 dB apart",
    "music_10db": "music 10 dB SNR", "hum_rumble": "hum + rumble", "bangs": "sudden bangs",
}
VAR_LABEL = {"raw": "original", "raw_norm": "original + level norm. (used)", "conserv": "conservative DSP",
             "df_plan": "DeepFilterNet bounded (listening path)", "df_full": "DeepFilterNet unbounded"}


def load(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pooled(rows):
    errs = sum(r["sub"] + r["del"] + r["ins"] for r in rows)
    n = sum(r["n"] for r in rows)
    return 100.0 * errs / n if n else float("nan")


def wer_table(rows, variants, title):
    by = defaultdict(dict)
    for r in rows:
        if "wer" in r:
            by[(r["lang"], r["cond"])][r["variant"]] = r
    out = [f"**{title}** (word error rate %, lower is better; pooled over the condition's utterances)\n"]
    out.append("| Language | Condition | " + " | ".join(VAR_LABEL.get(v, v) for v in variants) + " |")
    out.append("|---|---|" + "---|" * len(variants))
    for lang in ("da_dk", "en_us"):
        for c in COND_ORDER:
            d = by.get((lang, c))
            if not d:
                continue
            cells = []
            best = min((d[v]["wer"] for v in variants if v in d), default=None)
            for v in variants:
                if v in d:
                    w = d[v]["wer"] * 100
                    cells.append(f"**{w:.1f}**" if best is not None and abs(d[v]["wer"] - best) < 1e-9 else f"{w:.1f}")
                else:
                    cells.append("–")
            out.append(f"| {lang[:2]} | {COND_LABEL.get(c, c)} | " + " | ".join(cells) + " |")
    # pooled over everything where all variants exist
    keys = [k for k, d in by.items() if all(v in d for v in variants)]
    if keys:
        cells = [f"{pooled([by[k][v] for k in keys]):.1f}" for v in variants]
        out.append(f"| all | pooled | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def estoi_table(rows, variants):
    by = defaultdict(dict)
    for r in rows:
        if "estoi" in r:
            by[(r["lang"], r["cond"])][r["variant"]] = r["estoi"]
    out = ["**Intelligibility estimate ESTOI** (0–1, higher is better; vs. the clean reference)\n"]
    out.append("| Language | Condition | " + " | ".join(VAR_LABEL.get(v, v) for v in variants) + " |")
    out.append("|---|---|" + "---|" * len(variants))
    for lang in ("da_dk", "en_us"):
        for c in COND_ORDER:
            d = by.get((lang, c))
            if not d:
                continue
            best = max((d[v] for v in variants if v in d), default=None)
            cells = []
            for v in variants:
                if v in d:
                    cells.append(f"**{d[v]:.3f}**" if best is not None and abs(d[v] - best) < 1e-9 else f"{d[v]:.3f}")
                else:
                    cells.append("–")
            out.append(f"| {lang[:2]} | {COND_LABEL.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def model_table(large, turbo):
    by = defaultdict(dict)
    for name, rows in (("large-v3", large), ("large-v3-turbo", turbo)):
        for r in rows:
            by[(r["lang"], r["cond"])][name] = r
    out = ["| Language | Condition | large-v3 WER % | turbo WER % | large-v3 time (s) | turbo time (s) |", "|---|---|---|---|---|---|"]
    for lang in ("da_dk", "en_us"):
        for c in COND_ORDER:
            d = by.get((lang, c))
            if not d or len(d) < 2:
                continue
            a, b = d["large-v3"], d["large-v3-turbo"]
            out.append(f"| {lang[:2]} | {COND_LABEL.get(c, c)} | {a['wer'] * 100:.1f} | {b['wer'] * 100:.1f} | {a['asr_s']:.0f} | {b['asr_s']:.0f} |")
    keys = [k for k, d in by.items() if len(d) == 2]
    for lang in ("da_dk", "en_us"):
        ks = [k for k in keys if k[0] == lang]
        if ks:
            out.append(f"| {lang[:2]} | pooled | {pooled([by[k]['large-v3'] for k in ks]):.1f} | "
                       f"{pooled([by[k]['large-v3-turbo'] for k in ks]):.1f} | "
                       f"{sum(by[k]['large-v3']['asr_s'] for k in ks):.0f} | {sum(by[k]['large-v3-turbo']['asr_s'] for k in ks):.0f} |")
    return "\n".join(out) + "\n"


def main():
    d = Path(sys.argv[1])
    large, turbo = load(d / "model_large_v3.jsonl"), load(d / "model_turbo.jsonl")
    print("### Model comparison (recognition path, full 4-minute files)\n")
    print(model_table(large, turbo))
    m_t = load(d / "matrix_turbo.jsonl")
    if m_t:
        print("### Audio variants → recognition, Whisper large-v3-turbo (first 150 s of each file)\n")
        print(wer_table(m_t, ["raw", "raw_norm", "conserv", "df_plan", "df_full"], "large-v3-turbo"))
    m_l = load(d / "matrix_large.jsonl")
    if m_l:
        print("### Audio variants → recognition, Whisper large-v3 (first 90 s of each file)\n")
        print(wer_table(m_l, ["raw", "raw_norm", "conserv", "df_plan", "df_full"], "large-v3"))
    e = load(d / "estoi_all.jsonl")
    if e:
        print("### Listening path – intelligibility estimate (full 4-minute files)\n")
        print(estoi_table(e, ["raw", "conserv", "df_plan", "df_full"]))
    hdr = None
    for r in load(d / "matrix_turbo.jsonl") + load(d / "matrix_large.jsonl"):
        st = r.get("stats") or {}
        if st.get("dropped_artefacts") or st.get("repetitions_collapsed") or st.get("repaired_segments"):
            if hdr is None:
                print("### Integrity safeguards that triggered\n")
                print("| Model | Lang | Condition | Variant | dropped artefacts | repetitions collapsed | repaired segments | uncertain | unclear markers |")
                print("|---|---|---|---|---|---|---|---|---|")
                hdr = True
            print(f"| {r['model'].replace('faster-whisper-', '')} | {r['lang'][:2]} | {r['cond']} | {r['variant']} | "
                  f"{st.get('dropped_artefacts', 0)} | {st.get('repetitions_collapsed', 0)} | {st.get('repaired_segments', 0)} | "
                  f"{st.get('uncertain_segments', 0)} | {st.get('unclear_markers', 0)} |")


if __name__ == "__main__":
    main()
