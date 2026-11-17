"""Ukur WER transkripsi Vosk pada klip dataset (lokal, tanpa API eksternal).

Jalankan dari folder aol_v2:
    python wer_vosk.py                      # clean + noise (+ other tidak dipakai)
    python wer_vosk.py --conds clean,noise --model model
Hasil per klip: results/wer_vosk_per_klip.csv ; ringkasan dicetak ke layar.

WER per klip memakai wer() dari dtw_knn.py. WER agregat per grup = (jumlah kesalahan kata) / (jumlah kata referensi),
sehingga klip berkalimat panjang berbobot lebih besar (sama dengan cara loso.py menghitung WER keseluruhan).
Teks referensi = kolom 'label' di metadata.csv. Latency = waktu dari mulai memberi audio ke Vosk
sampai FinalResult (ms); RTF = latency / durasi klip.
"""
import argparse, csv, json, os, time, wave, collections
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
from dtw_knn import wer

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="dataset")
parser.add_argument("--model", default="model")
parser.add_argument("--conds", default="clean,noise")
parser.add_argument("--out", default="results")
parser.add_argument("--chunk", type=int, default=4000, help="sampel per chunk (4000 = 0,25 dtk, seperti streaming)")
parser.add_argument("--norm", action="store_true", help="normalisasi puncak volume ke 0,7 sebelum dikirim ke Vosk")
parser.add_argument("--grammar", action="store_true", help="batasi dekoder Vosk ke 10 kalimat target (+ [unk])")
parser.add_argument("--limit", type=int, default=0, help="hanya N klip pertama (untuk uji cepat)")
parser.add_argument("--speaker", default="", help="hanya speaker ini, mis. felicia")
parser.add_argument("--rep", type=int, default=0, help="hanya pengulangan (rep) ke-N, mis. 1")
parser.add_argument("--max-noise-rep", type=int, default=0, help="pakai klip noise rep 1..N saja (clean tetap semua), mis. 4")
parser.add_argument("--tag", default="", help="akhiran nama file hasil, mis. _norm_grammar")
args = parser.parse_args()
conds = args.conds.split(",")

SetLogLevel(-1)
model = Model(args.model)
rows = [r for r in csv.DictReader(open(f"{args.root}/metadata.csv", encoding="utf-8"))
        if r["status"] == "ok" and r["condition"] in conds]
if args.speaker:
    rows = [r for r in rows if r["speaker"] == args.speaker]
if args.max_noise_rep:
    rows = [r for r in rows if not (r["condition"] == "noise" and int(r["rep"]) > args.max_noise_rep)]
if args.rep:
    rows = [r for r in rows if int(r["rep"]) == args.rep]
if args.limit:
    rows = rows[::max(1, len(rows) // args.limit)][:args.limit]
print(f"{len(rows)} klip, kondisi {conds}, norm={args.norm}, grammar={args.grammar}")
PHRASES = ["Please pay with cash", "Please pay with card", "Please scan the QR code", "I don't have that", "Sold out",
           "How much is this?", "Thank you very much", "Please wait a moment", "Do you need a bag?", "Here is your change"]
GRAMMAR = json.dumps([p.lower().replace("?", "") for p in PHRASES] + ["[unk]"])


def edit_counts(ref, hyp):
    """(S, D, I, N) via backtrace; tokenisasi sama dengan wer() di dtw_knn.py."""
    r = ref.lower().replace("?", "").split()
    h = hyp.lower().replace("?", "").split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i-1, j-1] + (r[i-1] != h[j-1]), d[i-1, j] + 1, d[i, j-1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (r[i-1] != h[j-1]):
            S += int(r[i-1] != h[j-1]); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i-1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I, n


_REC = None
def get_recognizer():
    """Satu recognizer dipakai ulang (Reset) agar tidak membuat ratusan objek Kaldi -> menghindari segmentation fault."""
    global _REC
    if _REC is None:
        _REC = KaldiRecognizer(model, 16000, GRAMMAR) if args.grammar else KaldiRecognizer(model, 16000)
    elif hasattr(_REC, "Reset"):
        _REC.Reset()
    else:  # vosk lama tanpa Reset: buat baru
        _REC = KaldiRecognizer(model, 16000, GRAMMAR) if args.grammar else KaldiRecognizer(model, 16000)
    return _REC


def transcribe(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    if args.norm:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        raw = (x / (np.abs(x).max() + 1e-9) * 0.7 * 32767).astype(np.int16).tobytes()
    rec = get_recognizer()
    t0 = time.time()
    step = args.chunk * 2
    for i in range(0, len(raw), step):
        rec.AcceptWaveform(raw[i:i + step])
    text = json.loads(rec.FinalResult()).get("text", "")
    return text, (time.time() - t0) * 1000, len(raw) / 2 / 16000


out = []
for k, r in enumerate(rows):
    hyp, ms, dur = transcribe(f"{args.root}/wav/{r['file']}")
    S, D, I, N = edit_counts(r["label"], hyp)
    assert abs((S + D + I) / max(1, N) - wer(r["label"], hyp)) < 1e-9   # konsisten dengan wer() di dtw_knn.py
    out.append({"file": r["file"], "speaker": r["speaker"], "condition": r["condition"], "ref": r["label"],
                "hyp": hyp, "wer": round(wer(r["label"], hyp), 4), "S": S, "D": D, "I": I, "N": N,
                "exact": int(hyp.lower() == r["label"].lower().replace("?", "")), "latency_ms": round(ms, 1),
                "dur_s": round(dur, 2)})
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(rows)}", flush=True)
        os.makedirs(args.out, exist_ok=True)
        with open(f"{args.out}/wer_vosk_sebagian{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

os.makedirs(args.out, exist_ok=True)
path = f"{args.out}/wer_vosk_per_klip{args.tag}.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)


def summarize(sub):
    N = sum(x["N"] for x in sub)
    return (sum(x["S"] + x["D"] + x["I"] for x in sub) / N, sum(x["S"] for x in sub), sum(x["D"] for x in sub),
            sum(x["I"] for x in sub), N, np.mean([x["exact"] for x in sub]), np.mean([x["latency_ms"] for x in sub]),
            np.max([x["latency_ms"] for x in sub]))


print(f"\n{'grup':<22}{'klip':>5}{'WER':>8}{'S':>6}{'D':>6}{'I':>6}{'N':>6}{'persis':>8}{'lat_rata':>10}{'lat_maks':>10}")
def show(name, sub):
    wr, S, D, I, N, ex, la, lm = summarize(sub)
    print(f"{name:<22}{len(sub):>5}{wr:>8.1%}{S:>6}{D:>6}{I:>6}{N:>6}{ex:>8.1%}{la:>9.0f}ms{lm:>9.0f}ms")

for c in conds:
    show(f"SEMUA / {c}", [x for x in out if x["condition"] == c])
print()
for s in sorted({x["speaker"] for x in out}):
    for c in conds:
        show(f"{s} / {c}", [x for x in out if x["speaker"] == s and x["condition"] == c])
print()
print("WER per kalimat (semua speaker):")
for lab in sorted({x["ref"] for x in out}):
    for c in conds:
        sub = [x for x in out if x["ref"] == lab and x["condition"] == c]
        wr = summarize(sub)[0]
        print(f"  {c:<6}{wr:>7.1%}  {lab}")
print("\nHipotesis Vosk yang paling sering muncul (kosong = tidak ada kata):")
for (c, lab), cnt in sorted(collections.Counter((x["condition"], x["hyp"]) for x in out).most_common(8)):
    pass
for hyp, n in collections.Counter(x["hyp"] for x in out).most_common(8):
    print(f"  {n:>4}x  '{hyp}'")
print(f"\nHasil per klip: {path}")