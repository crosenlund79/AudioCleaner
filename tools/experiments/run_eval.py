"""Evaluate enhancement variants for intelligibility (ESTOI) and transcription accuracy (WER).

Runs the *product code* (audiorefine.*) on the corpus made by make_corpus.py.

Variants (audio fed to the recogniser):
  raw        decoded audio, 16 kHz, no level normalisation
  raw_norm   decoded audio with per-clip level normalisation       (AudioRefine's recognition path)
  conserv    high-pass/hum/EQ + leveller, no neural noise reduction
  df_plan    AudioRefine's listening path (DeepFilterNet with SNR-dependent attenuation limit)
  df_full    DeepFilterNet without attenuation limit + leveller
  fw_default faster-whisper's own long-form transcription with its VAD filter on raw audio

Usage: python run_eval.py --corpus /path/corpus --model /path/model --out results.jsonl [--variants ...] [--conds ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from audiorefine import analysis, enhance, vad  # noqa: E402
from audiorefine.rawaudio import Decimator3, RawWriter, raw_length, read_raw  # noqa: E402
from audiorefine.transcriber import TranscribeOptions, Transcriber  # noqa: E402

_PUNCT = re.compile(r"[^\w\s]")


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    t = t.replace("-", " ")
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


def wer(ref: str, hyp: str) -> dict:
    import jiwer

    o = jiwer.process_words(ref, hyp)
    n = len(ref.split())
    return {"wer": o.wer, "sub": o.substitutions, "del": o.deletions, "ins": o.insertions, "n": n}


def to16(x48: np.ndarray) -> np.ndarray:
    d = Decimator3()
    return np.concatenate([d.process(x48), d.flush()])


def estoi(clean16: np.ndarray, proc16: np.ndarray) -> float:
    from pystoi import stoi

    n = min(len(clean16), len(proc16))
    return float(stoi(clean16[:n], proc16[:n], 16000, extended=True))


def listening_variant(x48: np.ndarray, kind: str, probs: np.ndarray, work: Path) -> tuple[np.ndarray, dict]:
    src = work / "src.f32"
    with RawWriter(src) as w:
        w.write(x48)
    a, sp = analysis.analyse_source(src, probs, 0)
    plan = analysis.plan_enhancement(a)
    if kind == "conserv":
        plan.df_atten_db = 0.0
    elif kind == "df_full":
        plan.df_atten_db = 100.0
    out = work / "out.f32"
    with RawWriter(out) as w:
        rep = enhance.enhance_source(src, plan, sp, work, w, workers=4, is_first=True, is_last=True)
    y = read_raw(out)
    assert len(y) == len(x48), (len(y), len(x48))
    return y, {"analysis": a.__dict__, "plan": plan.to_dict(), "report": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=["raw", "raw_norm", "conserv", "df_plan", "df_full"])
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--langs", nargs="*", default=["da_dk", "en_us"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0, help="only use the first N seconds of each file")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    man = json.load(open(corpus / "manifest.json", encoding="utf-8"))
    model = None
    if not args.no_asr:
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    outf = open(args.out, "a", encoding="utf-8")
    for lang in args.langs:
        m = man[lang]
        refs = m["refs"]
        if args.max_seconds:
            refs = [r for r in refs if r["end"] <= args.max_seconds]
        ref_text = norm_text(" ".join(r["raw"] for r in refs))
        clean48, _ = sf.read(corpus / f"{lang}_reference_clean.wav", dtype="float32")
        if args.max_seconds:
            cut = int(max(r["end"] for r in refs) * 48000) + 24000
            clean48 = clean48[:cut]
        clean16 = to16(clean48)
        conds = args.conds or list(m["files"].keys())
        for cond in conds:
            x48, _ = sf.read(m["files"][cond], dtype="float32")
            if args.max_seconds:
                x48 = x48[: len(clean48)]
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                raw16 = to16(x48)
                with RawWriter(work / "a16.f32") as w:
                    w.write(raw16)
                probs = vad.speech_probabilities(work / "a16.f32")
                listen_cache = {}
                for var in args.variants:
                    rec = {"lang": lang, "cond": cond, "variant": var, "model": Path(args.model).name}
                    t0 = time.time()
                    if var in ("conserv", "df_plan", "df_full"):
                        y48, info = listening_variant(x48, var, probs, work)
                        rec["enhance_s"] = round(time.time() - t0, 1)
                        rec["plan"] = info["plan"]
                        rec["analysis"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in info["analysis"].items()}
                        asr16 = to16(y48)
                        rec["estoi"] = estoi(clean16, asr16)
                        listen_cache[var] = asr16
                        normalise = True
                    else:
                        asr16 = raw16
                        normalise = var != "raw"
                        if var in ("raw", "raw_norm"):
                            rec["estoi"] = estoi(clean16, raw16)
                    if model is not None:
                        t1 = time.time()
                        if var == "fw_default":
                            segs, info = model.transcribe(asr16, beam_size=5, vad_filter=True, word_timestamps=True)
                            hyp = " ".join(s.text for s in segs)
                            stats = {}
                        else:
                            with RawWriter(work / "asr.f32") as w:
                                w.write(asr16)
                            p2 = vad.speech_probabilities(work / "asr.f32") if var != "raw" and var != "raw_norm" else probs
                            regions = vad.speech_regions(p2, len(asr16))
                            tr = Transcriber(model, TranscribeOptions(normalise_level=normalise))
                            segs = tr.transcribe(work / "asr.f32", regions, probs=p2)
                            hyp = " ".join(s.text for s in segs if not s.marker)
                            stats = dict(tr.stats)
                            stats["language"] = tr.policy.summary()
                        rec["asr_s"] = round(time.time() - t1, 1)
                        rec.update(wer(ref_text, norm_text(hyp)))
                        rec["stats"] = stats
                        rec["hyp_head"] = hyp[:160]
                    print(json.dumps({k: v for k, v in rec.items() if k not in ("plan", "analysis", "hyp_head")}), flush=True)
                    outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    outf.flush()


if __name__ == "__main__":
    main()
