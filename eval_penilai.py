"""Evaluasi aturan lulus (LOSO, seperti fusi_vosk_dtw.py):  python eval_penilai.py [--k 20]

Template DTW dari speaker LAIN saja; Vosk tidak dilatih ulang, jadi tidak ada kebocoran speaker.
 A. Ucapan benar dinilai terhadap kalimatnya sendiri  -> seharusnya lulus ("salah-tolak" bila tidak).
 B. Ucapan dinilai terhadap kalimat LAIN sebagai target, dan klip 'other' terhadap 10 target
    -> seharusnya TIDAK lulus ("salah-terima" bila lulus).
B menguji salah-terima level KALIMAT. Salah ucap halus (kata yang sama diucapkan keliru) belum ada datanya.
Membandingkan aturan LAMA di app.py (skor>=0,8 ATAU fusi benar) dengan aturan BARU (penilai.judge).
"""
import argparse, csv, collections, time
import numpy as np
from vosk import Model, SetLogLevel
from mfcc import features, read_wav
from dtw_knn import dtw_distance
from scorer import OTHER, STEP
import penilai as P

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--root", default="dataset")
ap.add_argument("--model", default="model")
args = ap.parse_args()

rows = [r for r in csv.DictReader(open(f"{args.root}/metadata.csv", encoding="utf-8")) if r["status"] == "ok"]
speakers = sorted({r["speaker"] for r in rows})
lab_of = lambda r: OTHER if r["condition"] == "other" else r["label"]
feats = {r["file"]: features(f"{args.root}/wav/{r['file']}")[::STEP] for r in rows}
SetLogLevel(-1)
model = Model(args.model)


def pick(files, k):
    if len(files) <= k:
        return list(files)
    d = {(a, b): dtw_distance(feats[a], feats[b]) for a in files for b in files if a < b}
    dd = lambda a, b: d[(a, b) if a < b else (b, a)]
    tot = {a: sum(dd(a, b) for b in files if b != a) for a in files}
    ch = [min(files, key=lambda a: tot[a])]
    while len(ch) < k:
        ch.append(max([f for f in files if f not in ch], key=lambda f: min(dd(f, c) for c in ch)))
    return ch


res = []
for held in speakers:
    g = collections.defaultdict(list)
    for r in rows:
        if r["speaker"] != held and r["condition"] in ("clean", "noise", "other"):
            g[lab_of(r)].append(r["file"])
    templ = {lab: [feats[f] for f in (fs if lab == OTHER else pick(fs, args.k))] for lab, fs in g.items()}
    t0 = time.time()
    for r in rows:
        if r["speaker"] != held:
            continue
        a = P.analyze(model, read_wav(f"{args.root}/wav/{r['file']}"), templ)
        res.append(dict(file=r["file"], speaker=held, cond=r["condition"], true=lab_of(r),
                        pred=a["label"], free=a["free"], gap=a["gap"]))
    print(f"held-out {held}: {time.time()-t0:.0f}s", flush=True)

NAN = float("nan")


def tier_new(x, tgt, gate=True):
    return P.judge(tgt, x["free"], x["pred"], x["gap"] if gate else NAN)[0]


def tier_old(x, tgt):
    return P.judge_old(tgt, x["free"], x["pred"])[0]


def pct(n, d):
    return f"{n / max(1, d):6.1%}"


print(f"\nAmbang: PASS_BAGUS={P.PASS_BAGUS}  PASS_HAMPIR={P.PASS_HAMPIR}  MIN_GAP_DB={P.MIN_GAP_DB}")
print("\n=== A. Ucapan BENAR terhadap kalimatnya sendiri ===")
print(f"{'kondisi':<22}{'n':>5}  {'LAMA lulus':>11}   {'BARU: bagus':>12}{'hampir':>8}{'ulang':>8}{'berisik':>9}")
for c in ("clean", "noise"):
    it = [x for x in res if x["cond"] == c]
    old = sum(tier_old(x, x["true"]) == "bagus" for x in it)
    for gate in (False, True):
        cnt = collections.Counter(tier_new(x, x["true"], gate) for x in it)
        name = f"{c} ({'dengan' if gate else 'tanpa'} pagar noise)"
        print(f"{name:<22}{len(it):>5}  {pct(old, len(it)):>11}   {pct(cnt['bagus'], len(it)):>12}{pct(cnt['hampir'], len(it)):>8}"
              f"{pct(cnt['ulang'], len(it)):>8}{pct(cnt['berisik'], len(it)):>9}")
print("  (tiap baris: LAMA = aturan ATAU di app.py lama; 'ulang' = salah-tolak ucapan yang benar;")
print("   'berisik' = tidak dinilai, user diminta pindah tempat; di kolom 'ulang'+'berisik' user harus mengulang)")
for s in speakers:
    it = [x for x in res if x["cond"] == "noise" and x["speaker"] == s]
    cnt = collections.Counter(tier_new(x, x["true"]) for x in it)
    print(f"  noise {s:<8} n={len(it)}  bagus {pct(cnt['bagus'], len(it))}  hampir {pct(cnt['hampir'], len(it))}  "
          f"ulang {pct(cnt['ulang'], len(it))}  berisik {pct(cnt['berisik'], len(it))}")

print("\n=== B. Kalimat SALAH sebagai target (salah-terima; makin kecil makin baik) ===")
print(f"{'sumber ucapan':<22}{'pasangan':>9}  {'LAMA lulus':>11}   {'BARU bagus':>11}{'bagus+hampir':>14}")
for c, name in (("clean", "clean"), ("noise", "noise"), ("other", "other (di luar 10)")):
    it = [x for x in res if x["cond"] == c]
    pairs = [(x, t) for x in it for t in P.PHRASES if t != x["true"]]
    old = sum(tier_old(x, t) == "bagus" for x, t in pairs)
    new = collections.Counter(tier_new(x, t) for x, t in pairs)
    print(f"{name:<22}{len(pairs):>9}  {pct(old, len(pairs)):>11}   {pct(new['bagus'], len(pairs)):>11}"
          f"{pct(new['bagus'] + new['hampir'], len(pairs)):>14}")

print("\n=== C. Pagar noise: selisih ucapan-noise (dB) vs skor kata (klip noise, benar) ===")
nz = [x for x in res if x["cond"] == "noise" and x["gap"] == x["gap"]]
for lo, hi in ((-99, 6), (6, 9), (9, 12), (12, 99)):
    it = [x for x in nz if lo <= x["gap"] < hi]
    if not it:
        continue
    ok = sum(P.score_of(P.match_words(x["true"], x["free"])) >= P.PASS_BAGUS for x in it)
    print(f"  gap {lo if lo > -99 else '<'}..{hi if hi < 99 else '>'} dB  n={len(it):>3}  skor kata >= {P.PASS_BAGUS}: {pct(ok, len(it))}")

import pickle
pickle.dump(res, open("results/eval_penilai_hasil.pkl", "wb"))
print("\nHasil mentah: results/eval_penilai_hasil.pkl")
