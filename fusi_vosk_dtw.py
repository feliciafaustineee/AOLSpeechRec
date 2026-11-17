import argparse, csv, json, os, time, collections
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
from mfcc import features, read_wav
from dtw_knn import dtw_distance, wer

OTHER = "__OTHER__"
STEP = 2
PHRASES = ["Please pay with cash", "Please pay with card", "Please scan the QR code", "I don't have that", "Sold out",
           "How much is this?", "Thank you very much", "Please wait a moment", "Do you need a bag?", "Here is your change"]
GRAMMAR = json.dumps([p.lower().replace("?", "") for p in PHRASES] + ["[unk]"])

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="clean,noise")
ap.add_argument("--test", default="noise")
ap.add_argument("--k", type=int, default=20, help="template per kalimat (0 = semua klip latih)")
ap.add_argument("--root", default="dataset")
ap.add_argument("--model", default="model")
ap.add_argument("--out", default="results")
args = ap.parse_args()
train_conds, test_conds = args.train.split(","), args.test.split(",")

rows = [r for r in csv.DictReader(open(f"{args.root}/metadata.csv", encoding="utf-8")) if r["status"] == "ok"]
speakers = sorted({r["speaker"] for r in rows})
label_of = lambda r: OTHER if r["condition"] == "other" else r["label"]

t0 = time.time()
cache = {r["file"]: features(f"{args.root}/wav/{r['file']}")[::STEP] for r in rows}
print(f"fitur {len(cache)} klip: {time.time()-t0:.1f}s")

# ---------------- Vosk (sekali per klip uji)
SetLogLevel(-1)
vmodel = Model(args.model)
rec = KaldiRecognizer(vmodel, 16000, GRAMMAR)


def vosk_grammar(x):
    """x: float -1..1. Normalisasi puncak ke 0,7 lalu decode dengan grammar."""
    y = (x / (np.abs(x).max() + 1e-9) * 0.7 * 32767).astype(np.int16).tobytes()
    rec.Reset()
    for i in range(0, len(y), 8000):
        rec.AcceptWaveform(y[i:i + 8000])
    return json.loads(rec.FinalResult()).get("text", "")


def pick_templates(files, k):
    if k <= 0 or len(files) <= k:
        return list(files)
    dist = {(a, b): dtw_distance(cache[a], cache[b]) for a in files for b in files if a < b}
    d = lambda a, b: dist[(a, b) if a < b else (b, a)]
    totals = {a: sum(d(a, b) for b in files if b != a) for a in files}
    chosen = [min(files, key=lambda a: totals[a])]
    while len(chosen) < k:
        rest = [f for f in files if f not in chosen]
        chosen.append(max(rest, key=lambda f: min(d(f, c) for c in chosen)))
    return chosen


def counts(ref, hyp):
    if hyp == OTHER:
        n = len(ref.lower().split()); return n, n
    return round(wer(ref, hyp) * len(ref.split())), len(ref.split())


def decide(hyp, cls_dist, dtw_pred):
    words = hyp.replace("[unk]", " ").split()
    clean_h = " ".join(words)
    if not clean_h:
        pred = dtw_pred
        best = 1.0
    else:
        vw = {p: wer(p, clean_h) for p in PHRASES}
        best = min(vw.values())
        tied = [p for p in PHRASES if vw[p] == best]
        pred = min(tied, key=lambda p: cls_dist.get(p, np.inf))
    if ("[unk]" in hyp and best > 0.5) or (dtw_pred == OTHER and best > 0):
        pred = OTHER
    return pred


out = []
for held in speakers:
    tr = [r for r in rows if r["speaker"] != held and r["condition"] in train_conds + ["other"]]
    te = [r for r in rows if r["speaker"] == held and r["condition"] in test_conds]
    groups = collections.defaultdict(list)
    for r in tr:
        groups[label_of(r)].append(r["file"])
    T = {lab: (fs if lab == OTHER else pick_templates(fs, args.k)) for lab, fs in groups.items()}
    for r in te:
        t1 = time.time()
        cls = {lab: min(dtw_distance(cache[r["file"]], cache[f]) for f in fs) for lab, fs in T.items()}
        dtw_pred = min(cls, key=cls.get)
        t_dtw = time.time() - t1
        t1 = time.time()
        hyp = vosk_grammar(read_wav(f"{args.root}/wav/{r['file']}"))
        t_vosk = time.time() - t1
        pred = decide(hyp, cls, dtw_pred)
        true = label_of(r)
        e, n = counts(true, pred) if true != OTHER else (0, 0)
        ed, _ = counts(true, dtw_pred) if true != OTHER else (0, 0)
        out.append(dict(held=held, file=r["file"], condition=r["condition"], true=true, vosk=hyp, dtw=dtw_pred,
                        pred=pred, correct=int(pred == true), err=e, err_dtw=ed, N=n,
                        ms_dtw=round(t_dtw * 1000), ms_vosk=round(t_vosk * 1000)))
    print(f"held-out {held}: {len(te)} klip, {sum(len(v) for v in T.values())} template", flush=True)

os.makedirs(args.out, exist_ok=True)
path = f"{args.out}/fusi_{args.train.replace(',', '+')}_vs_{args.test.replace(',', '+')}_k{args.k}.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

print(f"\n{'grup':<20}{'klip':>6}{'WER DTW':>10}{'WER fusi':>10}{'akurasi':>9}")
for c in test_conds:
    for s in speakers + ["SEMUA"]:
        sub = [x for x in out if x["condition"] == c and (s == "SEMUA" or x["held"] == s)]
        if not sub: continue
        if c == "other":
            print(f"{c + ' / ' + s:<20}{len(sub):>6}   ditolak benar: {sum(x['correct'] for x in sub)}/{len(sub)}"
                  f"  (DTW saja: {sum(x['dtw'] == OTHER for x in sub)}/{len(sub)})")
            continue
        N = sum(x["N"] for x in sub)
        print(f"{c + ' / ' + s:<20}{len(sub):>6}{sum(x['err_dtw'] for x in sub)/N:>10.1%}"
              f"{sum(x['err'] for x in sub)/N:>10.1%}{np.mean([x['correct'] for x in sub]):>9.1%}")
print(f"latency rata-rata: DTW {np.mean([x['ms_dtw'] for x in out]):.0f} ms, Vosk {np.mean([x['ms_vosk'] for x in out]):.0f} ms")
print(f"Hasil per klip: {path}")
