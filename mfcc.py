import wave
import numpy as np

SR = 16000
PRE_EMPH = 0.97
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
NFFT = 512
N_MEL = 26
N_MFCC = 13
CMVN = True          # False -> perilaku lama (hanya CMN)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"{path}: harus 16 kHz, mono, 16-bit"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def trim_silence(x, frame=FRAME, hop=HOP, margin_db=12.0, skip=60):
    """(LAMA, tidak dipakai lagi oleh features()) Buang hening awal/akhir. Threshold = noise floor + margin."""
    n = max(1, 1 + (len(x) - frame) // hop)
    energy = np.array([np.sum(x[i*hop:i*hop+frame] ** 2) for i in range(n)])
    energy_db = 10 * np.log10(energy + 1e-12)
    first = skip if n > skip + 10 else 0
    floor = np.percentile(energy_db[first:], 10)
    active = np.where((energy_db > floor + margin_db) & (np.arange(n) >= first))[0]
    if len(active) == 0:
        return x
    start = max(0, active[0] - 5) * hop
    end = min(n, active[-1] + 6) * hop + frame
    return x[start:end]


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbank():
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(SR / 2), N_MEL + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((NFFT + 1) * hz_points / SR).astype(int)
    fb = np.zeros((N_MEL, NFFT // 2 + 1))
    for m in range(1, N_MEL + 1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        for k in range(l, c):
            fb[m-1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m-1, k] = (r - k) / max(1, r - c)
    return fb


FB = mel_filterbank()
HAMMING = np.hamming(FRAME)
FREQS = np.arange(NFFT // 2 + 1) * SR / NFFT
SPEECH_BAND = (FREQS >= 300) & (FREQS <= 3400)


def dct2_matrix(n_out, n_in):
    """Matriks DCT-II (ditulis manual)."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))


DCT = dct2_matrix(N_MFCC, N_MEL)


def frame_signal(x):
    x = np.append(x[0], x[1:] - PRE_EMPH * x[:-1])     # pre-emphasis
    n = max(1, 1 + (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    if idx.max() >= len(x):
        x = np.pad(x, (0, idx.max() - len(x) + 1))
    return x[idx] * HAMMING


def power_spectrum(x):
    return np.abs(np.fft.rfft(frame_signal(x), n=NFFT)) ** 2


def mfcc(x):
    """(LAMA) MFCC + CMN dari sinyal. Dipertahankan untuk kompatibilitas."""
    spec = power_spectrum(x)
    mel = np.log(spec @ FB.T + 1e-10)
    c = mel @ DCT.T
    return c - c.mean(axis=0)


# ---------------------------------------------------------------- VAD baru
DIGITAL_SILENCE_DB = -85.0   # frame lebih sunyi dari ini = senyap digital (mic belum aktif), diabaikan


def _valid_frames(P):
    """Frame yang bukan senyap digital (awal rekaman sering berisi nol / hampir nol)."""
    e_all = 10 * np.log10(P.sum(1) / NFFT + 1e-12)
    valid = e_all > DIGITAL_SILENCE_DB
    return valid if valid.sum() > 20 else np.ones(len(P), bool)


def spectral_subtraction(P, valid, alpha=2.0, beta=0.02, pct=20):
    """Spektrum noise = rata-rata frame paling sunyi (persentil `pct` energi pita ucapan).
    P_bersih = max(P - alpha*N, beta*P)  (over-subtraction + spectral floor)."""
    e = P[:, SPEECH_BAND].sum(1)
    thr = np.percentile(e[valid], pct)
    noise = P[valid & (e <= thr)].mean(0)
    return np.maximum(P - alpha * noise, beta * P)


def vad_bounds(P, smooth=5, max_gap=20, join=40, min_len=25, pad=8):
    """Kembalikan (frame_awal, frame_akhir) ucapan.
    Energi pita 300-3400 Hz dari spektrum yang sudah didenoise, dihaluskan 5 frame.
    Aktif bila energi > max(lantai + 50% rentang, puncak - 20 dB).
    Ambil segmen dengan energi terbesar, gabungkan segmen lain yang berjarak < `join` frame,
    lalu perlebar `pad` frame di kiri-kanan."""
    valid = _valid_frames(P)
    Pd = spectral_subtraction(P, valid)
    e = 10 * np.log10(Pd[:, SPEECH_BAND].sum(1) + 1e-12)
    e = np.convolve(e, np.ones(smooth) / smooth, mode="same")
    n = len(e)
    floor, peak = np.percentile(e[valid], 10), np.percentile(e[valid], 99)
    thr = max(floor + 0.5 * (peak - floor), peak - 20.0)
    idx = np.where((e > thr) & valid)[0]
    if len(idx) == 0:
        return 0, n
    segs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap:                 # celah > 0,2 dtk -> segmen baru
            segs.append((s, p)); s = i
        p = i
    segs.append((s, p))
    score = [np.sum(e[a:b + 1] - thr) for a, b in segs]
    a, b = segs[int(np.argmax(score))]
    for a2, b2 in segs:                     # gabungkan kata-kata yang berdekatan
        if b2 < a and a - b2 < join: a = min(a, a2)
        if a2 > b and a2 - b < join: b = max(b, b2)
    if b - a < min_len:
        c = (a + b) // 2; a, b = c - min_len // 2, c + min_len // 2
    return max(0, a - pad), min(n, b + pad + 1)


def deltas(feat, N=2):
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    denom = 2 * sum(i * i for i in range(1, N + 1))
    out = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        out[t] = sum(i * (padded[t + N + i] - padded[t + N - i]) for i in range(1, N + 1)) / denom
    return out


def features_from_array(x):
    """39 dimensi per frame: MFCC + delta + delta-delta, dari array audio float (-1..1)."""
    P = power_spectrum(x)
    a, b = vad_bounds(P)
    c = np.log(P[a:b] @ FB.T + 1e-10) @ DCT.T       # MFCC hanya pada segmen ucapan (spektrum asli)
    c = c - c.mean(axis=0)                           # CMN
    if CMVN:
        c = c / (c.std(axis=0) + 1e-8)               # + normalisasi varians
    d = deltas(c)
    dd = deltas(d)
    return np.hstack([c, d, dd])


def features(path):
    return features_from_array(read_wav(path))
