"""Score eval clips with candidate models. Saves per-model npz: clip [N,527], win2 [N,9,527] (2 s windows, 1 s hop),
optionally win1 [N,19,527] (1 s windows, 0.5 s hop). Usage: score_models.py MODEL SUBSET_JSON OUT_NPZ"""
import json, sys, time, os
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/home/user/sed/EfficientAT")
torch.set_num_threads(4)
MODEL, SUBSET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
vids = json.load(open(SUBSET))
SR = 32000
W = "/home/user/sed/weights/"

def load(v, sr=SR):
    x, s = sf.read(f"/home/user/sed/as_eval/{v}.flac", dtype="float32")
    x = x[:10 * SR]
    if len(x) < 10 * SR:
        x = np.pad(x, (0, 10 * SR - len(x)))
    if sr != SR:
        from scipy.signal import resample_poly
        x = resample_poly(x, sr, SR).astype(np.float32)
    return x

def windows(n_total, win, hop):
    return [(s, s + win) for s in range(0, n_total - win + 1, hop)]

res = {"clip": [], "win2": [], "win1": []}
t_model = 0.0
if MODEL.startswith(("mn", "dymn")):
    from models.preprocess import AugmentMelSTFT
    from models.mn.model import get_model as get_mn
    from models.dymn.model import get_model as get_dymn
    from helpers.utils import NAME_TO_WIDTH
    if MODEL.startswith("dymn"):
        net = get_dymn(width_mult=NAME_TO_WIDTH(MODEL), pretrained_name=MODEL)
    else:
        net = get_mn(width_mult=NAME_TO_WIDTH(MODEL), pretrained_name=MODEL)
    net.eval()
    mel = AugmentMelSTFT(n_mels=128, sr=SR, win_length=800, hopsize=320); mel.eval()
    def run(spec):  # spec [B,128,T]
        with torch.no_grad():
            return torch.sigmoid(net(spec.unsqueeze(1))[0]).numpy()
    for k, v in enumerate(vids):
        x = torch.from_numpy(load(v))[None]
        t0 = time.time()
        with torch.no_grad():
            S = mel(x)[0]  # [128, 1001]
        res["clip"].append(run(S[None]))
        w2 = torch.stack([S[:, a:b] for a, b in windows(S.shape[1], 200, 100)][:9])
        res["win2"].append(run(w2))
        if MODEL == "mn10_as":
            w1 = torch.stack([S[:, a:b] for a, b in windows(S.shape[1], 100, 50)][:19])
            res["win1"].append(run(w1))
        t_model += time.time() - t0
        if k % 500 == 0: print(k, f"{t_model:.0f}s", flush=True)
elif MODEL.startswith("cnn14"):
    from panns_inference.models import Cnn14, Cnn14_DecisionLevelMax
    if MODEL == "cnn14":
        net = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        ck = torch.load(W + "Cnn14_mAP=0.431.pth", map_location="cpu")
    else:
        net = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        ck = torch.load(W + "Cnn14_DecisionLevelMax_mAP=0.385.pth", map_location="cpu")
    net.load_state_dict(ck["model"]); net.eval()
    for k, v in enumerate(vids):
        x = torch.from_numpy(load(v))
        t0 = time.time()
        with torch.no_grad():
            out = net(x[None])
            res["clip"].append(out["clipwise_output"].numpy())
            if MODEL == "cnn14":
                if not os.environ.get("CLIP_ONLY"):
                    w2 = torch.stack([x[a:b] for a, b in windows(len(x), 2 * SR, SR)][:9])
                    res["win2"].append(net(w2)["clipwise_output"].numpy())
            else:
                fw = out["framewise_output"][0].numpy()  # [T,527] ~100 fps
                T = fw.shape[0]; fps = T / 10.0
                res["win2"].append(np.stack([fw[int(i * fps):int((i + 2) * fps)].max(0) for i in range(9)]))
        t_model += time.time() - t0
        if k % 200 == 0: print(k, f"{t_model:.0f}s", flush=True)
elif MODEL == "yamnet":
    import onnxruntime as ort
    sess = ort.InferenceSession(W + "yamnet/yamnet-onnx-float/yamnet_onnx/yamnet.onnx")
    ylab = [l.strip() for l in open(W + "yamnet/yamnet-onnx-float/labels.txt")]
    import csv
    names = [r[2] for r in list(csv.reader(open("/home/user/sed/class_labels_indices.csv")))[1:]]
    ymap = np.array([names.index(l) if l in names else -1 for l in ylab])
    print("unmapped", (ymap < 0).sum())
    # YAMNet front-end: 16 kHz, 25 ms periodic hann, 10 ms hop, fft 512, magnitude, 64 HTK mels 125-7500, log(x+0.001)
    import librosa
    melfb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=125, fmax=7500, htk=True, norm=None)
    win = np.hanning(401)[:400].astype(np.float32)  # periodic hann
    def logmel(x):
        n = 1 + (len(x) - 400) // 160
        fr = np.lib.stride_tricks.as_strided(x, (n, 400), (x.strides[0] * 160, x.strides[0]))
        mag = np.abs(np.fft.rfft(fr * win, 512))
        return np.log(mag @ melfb.T + 0.001).astype(np.float32)
    for k, v in enumerate(vids):
        x = load(v, 16000)
        t0 = time.time()
        L = logmel(x)  # [~998, 64]
        P = [L[s:s + 96] for s in range(0, L.shape[0] - 96 + 1, 48)]
        sc = np.stack([sess.run(None, {"audio": p[None, None]})[0][0] for p in P])  # [~19, 521]
        full = np.zeros((sc.shape[0], 527), np.float32)
        full[:, ymap[ymap >= 0]] = sc[:, ymap >= 0]
        res["clip"].append(full.mean(0, keepdims=True))
        # frames: 0.96 s patches at 0.48 s hop -> 2 s windows ~ max of the 3-4 patches inside
        fps = 1 / 0.48
        res["win2"].append(np.stack([full[int(i * fps):max(int(i * fps) + 1, int((i + 2) * fps) - 1)].max(0) for i in range(9)]))
        t_model += time.time() - t0
        if k % 500 == 0: print(k, f"{t_model:.0f}s", flush=True)
elif MODEL == "ast":
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    p = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(p); net = ASTForAudioClassification.from_pretrained(p).eval()
    import csv
    names = [r[2] for r in list(csv.reader(open("/home/user/sed/class_labels_indices.csv")))[1:]]
    order = [net.config.label2id[n] for n in names]
    for k, v in enumerate(vids):
        x = load(v, 16000)
        t0 = time.time()
        with torch.no_grad():
            inp = fe(x, sampling_rate=16000, return_tensors="pt")
            lo = net(**inp).logits[0].numpy()
        res["clip"].append(1 / (1 + np.exp(-lo[order]))[None])
        t_model += time.time() - t0
        if k % 200 == 0: print(k, f"{t_model:.0f}s", flush=True)
out = {k: np.stack(v).astype(np.float16) for k, v in res.items() if v}
out = {k: (a[:, 0] if k == "clip" else a) for k, a in out.items()}
np.savez_compressed(OUT, vids=np.array(vids), seconds=t_model, **out)
print("done", MODEL, {k: v.shape for k, v in out.items()}, f"model time {t_model:.0f}s for {len(vids)} clips")
