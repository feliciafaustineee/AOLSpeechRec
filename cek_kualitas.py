"""Cek kualitas klip noise (hanya baca; tidak mengubah apa pun).
 lantai  = level frame paling sunyi (persentil-10, setelah 0,6 dtk pertama), dB
 gap     = selisih frame paling keras (persentil-95) terhadap lantai, dB  (makin besar = ucapan makin jelas di atas noise)
 senyap  = % frame < -85 dB setelah 0,6 dtk pertama (senyap digital -> tanda gate / peredam noise aktif)
 clip    = % sampel >= 0,99 (pecah)
Tanda:  GATE (senyap>5%)  TENGGELAM (gap<6 dB)  PECAH (clip>0,1%)  PELAN (puncak<0,02)
Pakai:  python cek_kualitas.py --speaker felicia --cond noise [--max-rep 4]
"""
import argparse, csv, numpy as np
from mfcc import read_wav
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="dataset"); ap.add_argument("--speaker", default=""); ap.add_argument("--cond", default="noise")
ap.add_argument("--max-rep", type=int, default=0); ap.add_argument("--show", type=int, default=15, help="tampilkan N klip bermasalah")
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(f"{a.root}/metadata.csv", encoding="utf-8")) if r["status"] == "ok" and r["condition"] == a.cond
        and (not a.speaker or r["speaker"] == a.speaker) and (not a.max_rep or int(r["rep"]) <= a.max_rep)]
out = []
for r in rows:
    x = read_wav(f"{a.root}/wav/{r['file']}"); n = 1 + (len(x) - 400) // 160
    e = 10 * np.log10(np.array([np.mean(x[i*160:i*160+400] ** 2) for i in range(n)]) + 1e-12)
    fl = np.percentile(e[60:], 10); gap = np.percentile(e, 95) - fl
    sil = np.mean(e[60:] < -85) * 100; clip = np.mean(np.abs(x) >= 0.99) * 100; pk = np.abs(x).max()
    flags = [t for t, c in (("GATE", sil > 5), ("TENGGELAM", gap < 6), ("PECAH", clip > 0.1), ("PELAN", pk < 0.02)) if c]
    out.append((r["file"], fl, gap, sil, clip, pk, flags))
print(f"{len(out)} klip ({a.cond}{', ' + a.speaker if a.speaker else ''})")
if not out: raise SystemExit
arr = np.array([[o[1], o[2], o[3], o[4], o[5]] for o in out])
print(f"lantai median {np.median(arr[:,0]):.1f} dB | gap median {np.median(arr[:,1]):.1f} dB (min {arr[:,1].min():.1f}, maks {arr[:,1].max():.1f}) | puncak median {np.median(arr[:,4]):.3f}")
for t in ("GATE", "TENGGELAM", "PECAH", "PELAN"):
    print(f"  {t:<10}: {sum(t in o[6] for o in out)}/{len(out)} klip")
bad = [o for o in out if o[6]]
print(f"\nKlip bermasalah (maks {a.show}):")
for o in bad[:a.show]: print(f"  {o[0]:<28} lantai {o[1]:6.1f} gap {o[2]:5.1f} senyap {o[3]:4.0f}% puncak {o[5]:.3f}  {' '.join(o[6])}")