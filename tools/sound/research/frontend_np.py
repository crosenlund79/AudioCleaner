import numpy as np
MEL = np.load("/home/user/sed/onnx/mel_basis_128x513.npy")
N_FFT, HOP, WIN = 1024, 320, 800
_w = np.zeros(N_FFT, np.float32)
_w[(N_FFT - WIN) // 2:(N_FFT - WIN) // 2 + WIN] = np.hanning(WIN).astype(np.float32)  # symmetric hann (periodic=False)

def logmel(x):
    x = np.asarray(x, np.float32)
    y = x[1:] - 0.97 * x[:-1]
    y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(y) - N_FFT) // HOP
    fr = np.lib.stride_tricks.as_strided(y, (n, N_FFT), (y.strides[0] * HOP, y.strides[0]))
    spec = np.fft.rfft(fr * _w, axis=1)
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    m = MEL @ p.T
    return ((np.log(m + 1e-5) + 4.5) / 5.0).astype(np.float32)   # [128, T]
