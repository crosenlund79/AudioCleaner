"""Time per 10 s clip for each candidate (4 threads, after warm-up)."""
import sys, time, json, io, contextlib
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/home/user/sed/EfficientAT"); sys.path.insert(0, "/home/user/sed/scripts")
torch.set_num_threads(4)
from frontend_np import logmel
vids = json.load(open("/home/user/sed/subset_cmp.json"))[:20]
X = []
for v in vids:
    x, _ = sf.read(f"/home/user/sed/as_eval/{v}.flac", dtype="float32"); X.append(np.pad(x[:320000], (0, max(0, 320000 - len(x)))))
res = {}
from models.mn.model import get_model as get_mn
from models.dymn.model import get_model as get_dymn
from helpers.utils import NAME_TO_WIDTH
for name in ["mn10_as", "dymn10_as", "dymn20_as", "mn40_as_ext"]:
    with contextlib.redirect_stdout(io.StringIO()):
        net = (get_dymn if name.startswith("dymn") else get_mn)(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).eval()
    M = [torch.from_numpy(logmel(x)[None, None, :, :1000]) for x in X]
    with torch.no_grad():
        net(M[0]); t = time.time()
        for m in M: net(m)
    res[name] = (time.time() - t) / len(M)
from panns_inference.models import Cnn14, Cnn14_DecisionLevelMax
for name, cls, f in [("cnn14", Cnn14, "Cnn14_mAP=0.431.pth"), ("cnn14_dlm", Cnn14_DecisionLevelMax, "Cnn14_DecisionLevelMax_mAP=0.385.pth")]:
    net = cls(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    net.load_state_dict(torch.load("/home/user/sed/weights/" + f, map_location="cpu")["model"]); net.eval()
    with torch.no_grad():
        net(torch.from_numpy(X[0])[None]); t = time.time()
        for x in X: net(torch.from_numpy(x)[None])
    res[name] = (time.time() - t) / len(X)
from transformers import ASTFeatureExtractor, ASTForAudioClassification
p = "MIT/ast-finetuned-audioset-10-10-0.4593"
fe = ASTFeatureExtractor.from_pretrained(p); net = ASTForAudioClassification.from_pretrained(p).eval()
from scipy.signal import resample_poly
with torch.no_grad():
    net(**fe(resample_poly(X[0], 1, 2), sampling_rate=16000, return_tensors="pt")); t = time.time()
    for x in X: net(**fe(resample_poly(x, 1, 2), sampling_rate=16000, return_tensors="pt"))
res["ast"] = (time.time() - t) / len(X)
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession("/home/user/AudioCleaner/assets/sound/efficientat_mn10_as.onnx", so)
M = [logmel(x)[None, None, :, :1000] for x in X]
s.run(None, {"mel": M[0]}); t = time.time()
for m in M: s.run(None, {"mel": m})
res["mn10_onnx"] = (time.time() - t) / len(M)
print(json.dumps({k: round(v, 4) for k, v in res.items()}))
