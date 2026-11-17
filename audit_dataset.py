"""Audit dataset (hanya baca; tidak mengubah apa pun).
Cek: jumlah baris CSV vs file WAV, nama ganda, format (16 kHz mono 16-bit), durasi, WAV tanpa baris / baris tanpa WAV,
kecocokan nama file dengan kolom speaker/condition/rep, dan jumlah per speaker x kondisi.
Pakai:  python audit_dataset.py
"""
import csv, os, wave, collections, re
ROOT = "dataset"
rows = list(csv.DictReader(open(f"{ROOT}/metadata.csv", encoding="utf-8")))
wavs = sorted(f for f in os.listdir(f"{ROOT}/wav") if f.lower().endswith(".wav"))
names = [r["file"] for r in rows]
print(f"baris CSV: {len(rows)} | file WAV: {len(wavs)}")
dup = [k for k, v in collections.Counter(names).items() if v > 1]
print("nama ganda di CSV:", dup or "tidak ada")
print("WAV tanpa baris CSV:", sorted(set(wavs) - set(names)) or "tidak ada")
print("baris CSV tanpa WAV:", sorted(set(names) - set(wavs)) or "tidak ada")
print("status:", dict(collections.Counter(r["status"] for r in rows)))
bad = []; fmts = collections.Counter(); durs = collections.defaultdict(list)
for r in rows:
    p = f"{ROOT}/wav/{r['file']}"
    m = re.match(r"(\d+)_(.+)_(clean|noise|other)_(\d+)\.wav$", r["file"])
    if not m or m.group(2) != r["speaker"] or m.group(3) != r["condition"] or int(m.group(4)) != int(r["rep"]) or int(m.group(1)) != int(r["phrase_id"]):
        bad.append(("nama tidak cocok dengan kolom CSV", r["file"]))
    if not os.path.exists(p): continue
    with wave.open(p) as w:
        fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth() * 8); d = w.getnframes() / w.getframerate()
    fmts[fmt] += 1; durs[(r["speaker"], r["condition"])].append(d)
    if fmt != (16000, 1, 16): bad.append((f"format {fmt} (harus 16000 Hz, mono, 16-bit)", r["file"]))
    if abs(d - 4.5) > 0.05: bad.append((f"durasi {d:.2f} dtk (harus 4,5)", r["file"]))
print("format:", dict(fmts))
cnt = collections.Counter((r["speaker"], r["condition"]) for r in rows)
print("\nspeaker kondisi  baris  durasi(min-maks)")
for k in sorted(cnt): print(f"  {k[0]:<10}{k[1]:<7}{cnt[k]:>4}   {min(durs[k]):.2f}-{max(durs[k]):.2f}" if durs[k] else f"  {k} {cnt[k]}")
print(f"\nMasalah ditemukan: {len(bad)}")
for m, f in bad[:30]: print("  -", f, ":", m)