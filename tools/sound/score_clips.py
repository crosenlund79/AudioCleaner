"""Score evaluation clips exactly as the application does (ONNX model, 2 s windows every 1 s, each
centred in 10 s of silence). Writes raw per-window AudioSet scores for threshold tuning.

usage: score_clips.py LIST.json AUDIO_DIR OUT.npz [threads]
LIST.json: list of clip names (AUDIO_DIR/<name>.flac, 32 kHz mono)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audiorefine.sound.model import HOP_FRAMES, SR, WIN_FRAMES, SoundModel  # noqa: E402

names = json.load(open(sys.argv[1]))
adir = Path(sys.argv[2])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 2
m = SoundModel(threads=threads)
out, nwin, t_model = [], [], 0.0
t0 = time.time()
for k, n in enumerate(names):
    x, sr = sf.read(adir / f"{n}.flac", dtype="float32")
    assert sr == SR
    M = m.frontend.logmel(x)
    T = M.shape[1]
    starts = list(range(0, max(1, T - WIN_FRAMES + 1), HOP_FRAMES))
    if T > WIN_FRAMES and starts[-1] + WIN_FRAMES < T - HOP_FRAMES // 2:
        starts.append(T - WIN_FRAMES)
    W = np.stack([M[:, s:s + WIN_FRAMES] if s + WIN_FRAMES <= T else
                  np.pad(M[:, s:], ((0, 0), (0, WIN_FRAMES - (T - s))), constant_values=M.min()) for s in starts])
    t1 = time.time()
    out.append(m.scores(W).astype(np.float16))
    t_model += time.time() - t1
    nwin.append(len(starts))
    if k % 1000 == 0:
        print(k, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(sys.argv[3], names=np.array(names), nwin=np.array(nwin), scores=np.concatenate(out),
                    model_seconds=t_model)
print("done", len(names), "clips", sum(nwin), "windows", f"model {t_model:.0f}s total {time.time() - t0:.0f}s")
