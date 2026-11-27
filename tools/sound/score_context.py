"""10-second context scores (natural audio, the model's training condition) for evaluation clips.
usage: score_context.py LIST.json AUDIO_DIR OUT.npz [threads]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audiorefine.sound.model import SoundModel  # noqa: E402

names = json.load(open(sys.argv[1]))
m = SoundModel(threads=int(sys.argv[4]) if len(sys.argv) > 4 else 3)
out, t0 = [], time.time()
for k, n in enumerate(names):
    x, _ = sf.read(Path(sys.argv[2]) / f"{n}.flac", dtype="float32")
    M = m.frontend.logmel(x)
    out.append(m.context_scores(M)[0].max(axis=0).astype(np.float16))   # clips of 10.0x s have 2 windows
    if k % 1000 == 0:
        print(k, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(sys.argv[3], names=np.array(names), clip=np.stack(out))
print("done", len(names), f"{time.time() - t0:.0f}s")
