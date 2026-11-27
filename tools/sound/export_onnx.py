"""Export an EfficientAT AudioSet model to ONNX for AudioRefine (development tool, needs PyTorch).

    git clone https://github.com/fschmid56/EfficientAT   (MIT)
    pip install torch torchaudio torchvision onnx onnxruntime
    cd EfficientAT && python /path/to/export_onnx.py mn10_as /path/to/AudioRefine/assets/sound

Writes efficientat_<name>.onnx (input: log-mel [B, 1, 128, T]; output: sigmoid scores [B, 527];
batch size 1 returns [527]) and mel_basis_128x513.npy (the Kaldi mel filterbank of the training
front-end), and checks the export against PyTorch. The shipped file was exported from
mn10_as_mAP_471.pt (https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1); its SHA-256 is
recorded in src/audiorefine/sound/model.py.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

sys.path.insert(0, ".")
from helpers.utils import NAME_TO_WIDTH  # noqa: E402  (EfficientAT repository)
from models.dymn.model import get_model as get_dymn  # noqa: E402
from models.mn.model import get_model as get_mn  # noqa: E402

name, out = sys.argv[1], Path(sys.argv[2])
net = (get_dymn if name.startswith("dymn") else get_mn)(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).eval()


class Wrap(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.sigmoid(self.n(x)[0])


w = Wrap(net).eval()
x = torch.randn(3, 1, 128, 1000)
dst = out / f"efficientat_{name}.onnx"
torch.onnx.export(w, x, str(dst), input_names=["mel"], output_names=["scores"],
                  dynamic_axes={"mel": {0: "batch", 3: "frames"}, "scores": {0: "batch"}}, opset_version=17, dynamo=False)
mel, _ = torchaudio.compliance.kaldi.get_mel_banks(128, 1024, 32000, 0.0, 15000.0, vtln_low=100.0, vtln_high=-500.,
                                                   vtln_warp_factor=1.0)
np.save(out / "mel_basis_128x513.npy", torch.nn.functional.pad(mel, (0, 1), value=0).numpy().astype(np.float32))
with torch.no_grad():
    ref = w(x).numpy()
got = ort.InferenceSession(str(dst)).run(None, {"mel": x.numpy()})[0]
print(dst, "max abs difference vs PyTorch:", float(np.abs(ref - got).max()))
