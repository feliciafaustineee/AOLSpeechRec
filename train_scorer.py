"""Latih scorer dari dataset/ (semua speaker, kondisi yang dipilih).
Jalankan dari folder yang berisi dataset/, mfcc.py, dtw_knn.py, dan scorer.py:
    python train_scorer.py --conds clean --k 3
    python train_scorer.py --conds clean,noise --k 3
Hasil disimpan ke models/scorer_<kondisi>.pkl
"""
import argparse, csv, os, pickle, time, collections
import numpy as np
from mfcc import features
from dtw_knn import dtw_distance
from scorer import OTHER, STEP

parser = argparse.ArgumentParser()
parser.add_argument("--conds", default="clean", help="contoh: clean atau clean,noise")
parser.add_argument("--k", type=int, default=3, help="jumlah template per kalimat target")
parser.add_argument("--root", default="dataset")
parser.add_argument("--out", default="models")
args = parser.parse_args()
conds = args.conds.split(",")

rows = [r for r in csv.DictReader(open(f"{args.root}/metadata.csv", encoding="utf-8"))
        if r["status"] == "ok" and r["condition"] in conds + ["other"]]


def label_of(r):
    return OTHER if r["condition"] == "other" else r["label"]


t0 = time.time()
cache = {r["file"]: features(f"{args.root}/wav/{r['file']}")[::STEP] for r in rows}
print(f"{len(rows)} klip, fitur dihitung dalam {time.time()-t0:.1f} s")


def pick_templates(files, k):
    """k klip yang beragam: mulai dari medoid, lalu tambah klip paling jauh dari yang sudah dipilih."""
    if len(files) <= k:
        return list(files)
    dist = {(a, b): dtw_distance(cache[a], cache[b]) for a in files for b in files if a != b}
    totals = {a: sum(dist[(a, b)] for b in files if b != a) for a in files}
    chosen = [min(files, key=lambda a: totals[a])]
    while len(chosen) < k:
        rest = [f for f in files if f not in chosen]
        chosen.append(max(rest, key=lambda f: min(dist[(f, c)] for c in chosen)))
    return chosen


groups = collections.defaultdict(list)
for r in rows:
    groups[label_of(r)].append(r["file"])

templates = {}
for lab, files in groups.items():
    chosen = files if lab == OTHER else pick_templates(files, args.k)
    templates[lab] = [cache[f] for f in chosen]

os.makedirs(args.out, exist_ok=True)
out_path = f"{args.out}/scorer_{'+'.join(conds)}.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"templates": templates, "conds": conds, "k": args.k, "step": STEP}, f)

n_t = sum(len(v) for v in templates.values())
print(f"{len(templates)} kelas, {n_t} template")
print(f"Model disimpan di: {out_path}")
