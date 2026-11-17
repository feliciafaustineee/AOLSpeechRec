"""Logika penilaian (tanpa GUI) dipakai oleh app.py dan eval_penilai.py, supaya yang diuji = yang dipakai.

Dua keputusan DIPISAH:
  A. KALIMAT APA yang diucapkan  -> fusi: Vosk-grammar (norm) di-snap ke 10 kalimat, DTW atas fitur MFCC manual
                                    sebagai pemecah seri / cadangan / penolak 'other'. (dasar WER 3,5%)
  B. APAKAH KATA-KATANYA TERDENGAR BENAR -> transkripsi bebas Vosk (norm, TANPA grammar), dicocokkan kata per kata
                                    dengan target. Tanpa grammar, kata yang salah ucap tidak dipaksa jadi benar.
Aturan lulus: A benar DAN B cukup tinggi; di lingkungan terlalu berisik tidak dinilai sama sekali.
"""
import json
import re
import numpy as np
from vosk import KaldiRecognizer

from mfcc import features_from_array as mfcc_features, power_spectrum, vad_bounds, SPEECH_BAND
from scorer import OTHER, STEP
from dtw_knn import dtw_distance, wer

SR = 16000
PHRASES = ["Please pay with cash", "Please pay with card", "Please scan the QR code", "I don't have that",
           "Sold out", "How much is this?", "Thank you very much", "Please wait a moment",
           "Do you need a bag?", "Here is your change"]
GRAMMAR = json.dumps([p.lower().replace("?", "") for p in PHRASES] + ["[unk]"])

# ---- ambang keputusan (lihat eval_penilai.py untuk dasar pemilihan)
PASS_BAGUS = 0.75     # skor kata (transkripsi bebas) minimal untuk "Bagus!"  (kalimat 2-4 kata: boleh salah max 1 kata dari 4)
PASS_HAMPIR = 0.50    # di bawah PASS_BAGUS tapi >= ini (dan kalimat benar) -> "Hampir benar"
MIN_GAP_DB = 6.0      # selisih ucapan-noise (dB) minimal; di bawah ini -> "terlalu berisik", tidak dinilai


# ---------------------------------------------------------------- transkripsi kata demi kata
def normalize(text):
    """Huruf kecil, tanpa apostrof dan tanda baca, dan 'q r' digabung menjadi 'qr'."""
    t = text.lower().replace("'", "").replace("q r", "qr")
    return re.sub(r"[^a-z0-9 ]", " ", t).split()


def match_words(target, spoken):
    """Tandai kata target yang muncul di transkripsi dengan urutan benar (LCS)."""
    t, s = normalize(target), normalize(spoken)
    n, m = len(t), len(s)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if t[i] == s[j] else max(dp[i + 1][j], dp[i][j + 1])
    matched, i, j = [False] * n, 0, 0
    while i < n and j < m:
        if t[i] == s[j]:
            matched[i] = True; i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return list(zip(t, matched))


def score_of(pairs):
    return sum(1 for _, ok in pairs if ok) / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------- Vosk (offline, setelah rekaman selesai)
def _pcm(x):
    """Puncak dinormalisasi ke 0,7 (--norm) -> int16 bytes."""
    return (x / (np.abs(x).max() + 1e-9) * 0.7 * 32767).astype(np.int16).tobytes()


def _decode(rec, pcm):
    rec.Reset()
    for i in range(0, len(pcm), 8000):
        rec.AcceptWaveform(pcm[i:i + 8000])
    return json.loads(rec.FinalResult()).get("text", "")


_REC = {}


def _recognizer(model, grammar):
    key = (id(model), grammar)
    if key not in _REC:                                   # dibuat sekali; grammar tidak dikompilasi ulang
        _REC[key] = KaldiRecognizer(model, SR, GRAMMAR) if grammar else KaldiRecognizer(model, SR)
    return _REC[key]


def vosk_free(model, x):
    """Transkripsi BEBAS (norm, tanpa grammar) -> dasar penilaian kata."""
    return _decode(_recognizer(model, False), _pcm(x))


def vosk_grammar(model, x):
    """Transkripsi dibatasi 10 kalimat + [unk] (norm) -> dasar pengenalan kalimat."""
    return _decode(_recognizer(model, True), _pcm(x))


# ---------------------------------------------------------------- A. pengenalan kalimat (fusi)
def fused_label(hyp, feat, templates):
    """Vosk-grammar di-snap ke kalimat terdekat; DTW memutus seri, jadi cadangan bila Vosk kosong,
    dan menolak (OTHER) ucapan di luar 10 kalimat. Mengembalikan (label, jarak DTW ke label itu)."""
    cls = {lab: min(dtw_distance(feat, t) for t in feats) for lab, feats in templates.items()}
    dtw_pred = min(cls, key=cls.get)
    clean_h = " ".join(hyp.replace("[unk]", " ").split())
    if clean_h:
        vw = {p: wer(p, clean_h) for p in PHRASES}
        best = min(vw.values())
        pred = min([p for p in PHRASES if vw[p] == best], key=lambda p: cls.get(p, np.inf))
    else:
        best, pred = 1.0, dtw_pred
    if ("[unk]" in hyp and best > 0.5) or (dtw_pred == OTHER and best > 0):
        pred = OTHER
    return pred, cls.get(pred, cls[dtw_pred])


# ---------------------------------------------------------------- pagar kebisingan
def speech_noise_gap(x):
    """Selisih (dB) energi ucapan (persentil 90 di segmen VAD) dan noise (median di luar segmen),
    pada pita 300-3400 Hz. Bila hampir tidak ada bagian di luar ucapan -> NaN (tidak bisa dinilai = lolos)."""
    P = power_spectrum(x)
    a, b = vad_bounds(P)
    e = 10 * np.log10(P[:, SPEECH_BAND].sum(1) / 512 + 1e-12)
    out = np.r_[e[:a], e[b:]]
    out = out[out > -80]                                  # abaikan senyap digital
    if len(out) < 10:
        return float("nan")
    return float(np.percentile(e[a:b], 90) - np.percentile(out, 50))


# ---------------------------------------------------------------- keputusan akhir
def judge(target, free_text, label, gap_db):
    """Kembalikan (tingkat, skor_kata, pairs). tingkat: 'berisik' | 'bagus' | 'hampir' | 'ulang'."""
    pairs = match_words(target, free_text)
    s = score_of(pairs)
    if gap_db == gap_db and gap_db < MIN_GAP_DB:          # gap_db == gap_db  -> bukan NaN
        return "berisik", s, pairs
    if label == target and s >= PASS_BAGUS:
        return "bagus", s, pairs
    if label == target and s >= PASS_HAMPIR:
        return "hampir", s, pairs
    return "ulang", s, pairs


def judge_old(target, free_text, label):
    """Aturan lama di app.py (ATAU) -> hanya sebagai pembanding di eval_penilai.py."""
    pairs = match_words(target, free_text)
    return ("bagus" if (score_of(pairs) >= 0.8 or label == target) else "ulang"), score_of(pairs), pairs


def analyze(model, audio, templates):
    """Seluruh pipeline untuk satu rekaman. Mengembalikan dict."""
    feat = mfcc_features(audio)[::STEP]                    # fitur MFCC manual
    hyp_g = vosk_grammar(model, audio)
    label, dist = fused_label(hyp_g, feat, templates)
    free = vosk_free(model, audio)
    return dict(label=label, dist=dist, free=free, grammar=hyp_g, gap=speech_noise_gap(audio))
