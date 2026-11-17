import argparse, csv, os, time
import sounddevice as sd
import soundfile as sf

SR = 16000
WINDOW = 4.5        # total durasi rekam (detik), mulai saat Enter ditekan
HEADER = ["file", "speaker", "phrase_id", "label", "condition", "rep", "status"]

PHRASES = [
    "Please pay with cash",
    "Please pay with card",
    "Please scan the QR code",
    "I don't have that",
    "Sold out",
    "How much is this?",
    "Thank you very much",
    "Please wait a moment",
    "Do you need a bag?",
    "Here is your change",
]
OTHER = ["Good morning", "Yes", "No", "Where is the toilet?", "Hello"]

parser = argparse.ArgumentParser()
parser.add_argument("--speaker", required=True)
parser.add_argument("--condition", required=True, choices=["clean", "noise", "other"])
parser.add_argument("--reps", type=int, default=8)
parser.add_argument("--redo", nargs="+", help="contoh: 01_2 05_1 03_2")
args = parser.parse_args()

os.makedirs("dataset/wav", exist_ok=True)
csv_path = "dataset/metadata.csv"

# baca CSV yang sudah ada (kunci = nama file)
rows = {}
if os.path.exists(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = [r[h] for h in HEADER]

def save():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows.values())

if args.condition == "other":
    items = [(90 + i, t) for i, t in enumerate(OTHER, start=1)]   # 91..95
else:
    items = list(enumerate(PHRASES, start=1))                     # 1..10

if args.redo:
    text_map = dict(items)
    jobs = []
    for tok in args.redo:
        pid_s, rep_s = tok.split("_")
        pid, rep = int(pid_s), int(rep_s)
        jobs.append((pid, text_map[pid], rep))
else:
    jobs = [(pid, text, rep) for rep in range(1, args.reps + 1) for pid, text in items]

for pid, text, rep in jobs:
    fname = f"{pid:02d}_{args.speaker}_{args.condition}_{rep:02d}.wav"
    input(f"\n[{fname}] Ucapkan: \"{text}\"  -> tekan Enter, lalu langsung bicara")

    audio = sd.rec(int(WINDOW * SR), samplerate=SR, channels=1, dtype="int16")
    for i in [3, 2, 1]:
        print(f"  {i}...")
        time.sleep(0.7)
    print("  BICARA")
    sd.wait()

    sf.write(os.path.join("dataset/wav", fname), audio, SR, subtype="PCM_16")
    rows[fname] = [fname, args.speaker, pid, text, args.condition, rep, "ok"]
    save()
    print("  tersimpan:", fname)

print("\nSelesai.")