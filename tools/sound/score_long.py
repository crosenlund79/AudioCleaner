"""Window category scores for long recordings, computed exactly as the application does
(analyze_file with the candidate vocabulary). Used for false-positive rates per hour.

usage: score_long.py OUT_DIR FILE [FILE ...]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audiorefine.sound import vocab as V  # noqa: E402
from audiorefine.sound.analyze import analyze_file  # noqa: E402
from audiorefine.sound.model import SoundModel  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
m = SoundModel(threads=int(__import__("os").environ.get("THREADS", "2")))
voc = V.load()
for f in sys.argv[2:]:
    t0 = time.time()
    r = analyze_file(Path(f), m, voc, keep_scores=True)
    np.savez_compressed(out / (Path(f).stem + ".npz"), scores=r["_scores"].astype(np.float16), starts=r["_starts"],
                        duration=r["duration"], cats=np.array([c.id for c in voc.categories]))
    print(Path(f).name, f"{r['duration'] / 60:.1f} min in {time.time() - t0:.0f} s", flush=True)
