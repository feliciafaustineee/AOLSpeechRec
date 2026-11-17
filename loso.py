"""Leave-one-speaker-out (LOSO) dengan DTW + medoid, plus analisis S/D/I per kalimat.
Setiap speaker menjadi data uji sekali; model dilatih dari speaker lain.
Jalankan dari folder yang berisi dataset/, mfcc.py, dan dtw_knn.py:
    python loso.py --train clean,noise --test noise
    python loso.py --train clean --test noise
Hasil per klip disimpan ke folder results/.
"""
import argparse, csv, os, time, collections
import numpy as np
from mfcc import features
from dtw_knn import dtw_distance

OTHER = "__OTHER__"
STEP = 2  # ambil setiap frame kedua untuk mempercepat DTW

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="clean,noise", help="kondisi latih, contoh: clean,noise")
parser.add_argument("--test", default="noise", help="kondisi uji, contoh: noise")
parser.add_argument("--root", default="dataset")
parser.add_argument("--out", default="results")
parser.add_argument("--k", type=int, default=3, help="jumlah template per kalimat target")
args = parser.parse_args()

HEADER = ["file", "speaker", "phrase_id", "label", "condition", "rep", "status"]
rows = [r for r in csv.DictReader(open(f"{args.root}/metadata.csv", encoding="utf-8"))
        if r["status"] == "ok"]
speakers = sorted({r["speaker"] for r in rows})
train_conds = args.train.split(",")
test_conds = args.test.split(",")
print("speaker:", speakers)
for s in speakers:
    print(f"  {s}: {sum(1 for r in rows if r['speaker'] == s)} klip")


def label_of(r):
    return OTHER if r["condition"] == "other" else r["label"]


t0 = time.time()
cache = {r["file"]: features(f"{args.root}/wav/{r['file']}")[::STEP] for r in rows}
print(f"fitur dihitung untuk {len(cache)} klip dalam {time.time()-t0:.1f} s\n")


def pick_templates(files, k):
    """Pilih k klip yang beragam: mulai dari medoid, lalu tambah klip yang paling jauh dari yang sudah dipilih."""
    if len(files) <= k:
        return list(files)
    dist = {(a, b): dtw_distance(cache[a], cache[b]) for a in files for b in files if a != b}
    totals = {a: sum(dist[(a, b)] for b in files if b != a) for a in files}
    chosen = [min(files, key=lambda a: totals[a])]
    while len(chosen) < k:
        rest = [f for f in files if f not in chosen]
        nxt = max(rest, key=lambda f: min(dist[(f, c)] for c in chosen))
        chosen.append(nxt)
    return chosen


def build_templates(train_rows):
    groups = collections.defaultdict(list)
    for r in train_rows:
        groups[label_of(r)].append(r["file"])
    # k template per kalimat target; semua klip 'other' dipakai sebagai template
    return {lab: (files if lab == OTHER else pick_templates(files, args.k)) for lab, files in groups.items()}


def classify(feat, templates):
    best_lab, best_d = None, float("inf")
    for lab, files in templates.items():
        for f in files:
            d = dtw_distance(feat, cache[f])
            if d < best_d:
                best_lab, best_d = lab, d
    return best_lab, best_d


def align_counts(ref, hyp):
    """Jumlah substitusi (S), penghapusan (D), dan penyisipan (I) kata, dengan backtrace."""
    r = ref.lower().replace("?", "").split()
    h = hyp.lower().replace("?", "").split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (0 if r[i - 1] == h[j - 1] else 1)
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            if r[i - 1] != h[j - 1]:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, n


results, lat = [], []
for held in speakers:
    tr = [r for r in rows if r["speaker"] != held and r["condition"] in train_conds + ["other"]]
    te = [r for r in rows if r["speaker"] == held and r["condition"] in test_conds + ["other"]]
    assert not ({r["speaker"] for r in tr} & {r["speaker"] for r in te}), "kebocoran speaker!"
    templates = build_templates(tr)
    for r in te:
        t1 = time.time()
        pred, dist = classify(cache[r["file"]], templates)
        lat.append(time.time() - t1)
        true = label_of(r)
        S, D, I, N = align_counts(true, pred) if true != OTHER else (0, 0, 0, 0)
        results.append({"held_speaker": held, "file": r["file"], "condition": r["condition"],
                        "true": true, "pred": pred, "dist": round(float(dist), 4),
                        "correct": int(pred == true), "S": S, "D": D, "I": I, "N": N})
    print(f"held-out {held}: {len(te)} klip uji, {sum(len(v) for v in templates.values())} template")

os.makedirs(args.out, exist_ok=True)
name = f"{args.out}/loso_{args.train.replace(',', '+')}_vs_{args.test.replace(',', '+')}.csv"
with open(name, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

phrase = [x for x in results if x["true"] != OTHER]
other = [x for x in results if x["true"] == OTHER]
print("\n=== RINGKASAN (LOSO) ===")
if phrase:
    acc = np.mean([x["correct"] for x in phrase])
    S = sum(x["S"] for x in phrase); D = sum(x["D"] for x in phrase); I = sum(x["I"] for x in phrase)
    N = sum(x["N"] for x in phrase)
    print(f"Kalimat target benar: {sum(x['correct'] for x in phrase)}/{len(phrase)} = {acc:.1%}")
    print(f"WER keseluruhan: {(S+D+I)/N:.1%}  (S={S}, D={D}, I={I}, N={N} kata)")
if other:
    print(f"Kalimat 'other' ditolak dengan benar: {sum(x['correct'] for x in other)}/{len(other)}")
print(f"Latency per klip: rata-rata {np.mean(lat)*1000:.0f} ms, maksimum {np.max(lat)*1000:.0f} ms")

print("\nPer speaker (uji):")
for s in speakers:
    sub = [x for x in phrase if x["held_speaker"] == s]
    if sub:
        wer_s = (sum(x["S"]+x["D"]+x["I"] for x in sub)) / sum(x["N"] for x in sub)
        print(f"  {s}: akurasi {np.mean([x['correct'] for x in sub]):.1%}, WER {wer_s:.1%}")

print("\nAkurasi per kalimat:")
per = collections.defaultdict(lambda: [0, 0])
for x in results:
    per[x["true"]][0] += x["correct"]
    per[x["true"]][1] += 1
for k, (c, n) in sorted(per.items()):
    print(f"  {c}/{n}  {k}")

print("\nKesalahan kata terbanyak (S/D/I per kalimat):")
err = collections.Counter()
for x in phrase:
    err[x["true"]] += x["S"] + x["D"] + x["I"]
for k, n in err.most_common(5):
    print(f"  {n} kesalahan  {k}")
print(f"\nHasil per klip disimpan di: {name}")
