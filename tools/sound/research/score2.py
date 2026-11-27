"""EfficientAT scoring v2. Modes: clip (10 s input), frame (per-time-step head on the 10 s feature map),
pad (2 s windows every 1 s, each centred in 10 s of silence). Usage: score2.py MODEL SUBSET OUT modes..."""
import json, sys, time, io, contextlib
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/home/user/sed/EfficientAT"); sys.path.insert(0, "/home/user/sed/scripts")
from frontend_np import logmel
torch.set_num_threads(4)
MODEL, SUBSET, OUT = sys.argv[1:4]; modes = sys.argv[4:]
vids = json.load(open(SUBSET))
from models.mn.model import get_model as get_mn
from models.dymn.model import get_model as get_dymn
from helpers.utils import NAME_TO_WIDTH
with contextlib.redirect_stdout(io.StringIO()):
    net = (get_dymn if MODEL.startswith("dymn") else get_mn)(width_mult=NAME_TO_WIDTH(MODEL), pretrained_name=MODEL).eval()
SIL = np.float32((np.log(1e-5) + 4.5) / 5)
def clip(M):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(M[None, None]))[0]).numpy().reshape(-1)
def frame(M):
    with torch.no_grad():
        f = net.features(torch.from_numpy(M[None, None]))          # [1,960,F',T']
        v = f.mean(2).transpose(1, 2)                                # [1,T',960]
        h = net.classifier[2:](v)                                    # Linear, Hardswish, Dropout, Linear
        return torch.sigmoid(h)[0].numpy()                           # [T',527]
def pad(M):
    ws = []
    for i in range(9):
        w = np.full((128, 1000), SIL, np.float32); w[:, 400:600] = M[:, i * 100:i * 100 + 200]; ws.append(w)
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(np.stack(ws)[:, None]))[0]).numpy()
res = {m: [] for m in modes}; t = {m: 0.0 for m in modes}
for k, v in enumerate(vids):
    x, _ = sf.read(f"/home/user/sed/as_eval/{v}.flac", dtype="float32")
    x = np.pad(x[:320000], (0, max(0, 320000 - len(x))))
    M = logmel(x)[:, :1000]
    for m in modes:
        t0 = time.time(); res[m].append({"clip": clip, "frame": frame, "pad": pad}[m](M)); t[m] += time.time() - t0
    if k % 500 == 0: print(k, {m: round(s) for m, s in t.items()}, flush=True)
np.savez_compressed(OUT, vids=np.array(vids), **{m: np.stack(r).astype(np.float16) for m, r in res.items()},
                    **{"t_" + m: s for m, s in t.items()})
print("done", MODEL, {m: np.stack(r).shape for m, r in res.items()}, t)
