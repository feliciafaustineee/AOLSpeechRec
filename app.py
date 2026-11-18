import json
import os
import pickle
import queue
import re
import sys
import time
import tkinter as tk
import threading
import pyttsx3
from tkinter import messagebox

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import penilai as P         
from scorer import OTHER

MODEL_DIR = "model"
SCORER_PATH = os.path.join("models", "scorer_clean+noise.pkl")   # python train_scorer.py --conds clean,noise --k 20
SR = 16000
BLOCK = 4000          # 0.25 detik per blok audio
MIN_SECONDS = 0.3     # rekaman lebih pendek dari ini dianggap tidak terdengar
ALLOW_NEXT_ON_HAMPIR = True   # "Hampir benar" tetap boleh lanjut (False = harus "Bagus!" dulu)

LESSONS = [
    ("Please pay with cash", "Tolong bayar dengan uang tunai"),
    ("Please pay with card", "Tolong bayar dengan kartu"),
    ("Please scan the QR code", "Tolong pindai kode QR"),
    ("I don't have that", "Saya tidak punya itu"),
    ("Sold out", "Sudah habis"),
    ("How much is this?", "Ini berapa harganya?"),
    ("Thank you very much", "Terima kasih banyak"),
    ("Please wait a moment", "Tolong tunggu sebentar"),
    ("Do you need a bag?", "Kamu butuh tas?"),
    ("Here is your change", "Ini kembaliannya"),
]

PRONUNCIATION = {
    "Please pay with cash": "/pliːz peɪ wɪð kæʃ/",
    "Please pay with card": "/pliːz peɪ wɪð kɑːrd/",
    "Please scan the QR code": "/pliːz skæn ðə kjuː ɑːr koʊd/",
    "I don't have that": "/aɪ doʊnt hæv ðæt/",
    "Sold out": "/soʊld aʊt/",
    "How much is this?": "/haʊ mʌtʃ ɪz ðɪs/",
    "Thank you very much": "/θæŋk juː ˈvɛri mʌtʃ/",
    "Please wait a moment": "/pliːz weɪt ə ˈmoʊmənt/",
    "Do you need a bag?": "/duː juː niːd ə bæg/",
    "Here is your change": "/hɪr ɪz jɔːr tʃeɪndʒ/",
}

assert [t for t, _ in LESSONS] == P.PHRASES, "LESSONS harus sama dengan penilai.PHRASES"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Latihan Speaking - Bahasa Inggris")
        root.geometry("680x560")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.vosk = Model(MODEL_DIR)
        with open(SCORER_PATH, "rb") as f:
            self.templates = pickle.load(f)["templates"]

        self.idx = 0
        self.recording = False
        self.stream = None
        self.rec = None
        self.audio_q = queue.Queue()
        self.chunks = []
        self.final_text = ""

        self.lbl_progress = tk.Label(root, text="", font=("Arial", 10), fg="grey")
        self.lbl_progress.pack(pady=(12, 0))
        self.lbl_meaning = tk.Label(root, text="", font=("Arial", 13))
        self.lbl_meaning.pack(pady=(8, 0))
        self.lbl_target = tk.Label(root, text="", font=("Arial", 22, "bold"), wraplength=620)
        self.lbl_target.pack(pady=(4, 12))

        self.lbl_pron = tk.Label(root, text="", font=("Arial", 13), fg="#1565c0")
        self.lbl_pron.pack(pady=(0, 8))

        self.lbl_live = tk.Label(root, text="(tekan Mulai merekam lalu bicara)", font=("Arial", 12),
                                 fg="grey", wraplength=620)
        self.lbl_live.pack(pady=(0, 8))

        tk.Label(root, text="Transkripsi (Vosk, merah = kata tidak terdengar benar):", font=("Arial", 9), fg="grey").pack()
        self.txt_result = tk.Text(root, height=3, width=60, font=("Arial", 14), wrap="word", bd=0)
        self.txt_result.tag_configure("ok", foreground="#1a8c2b")
        self.txt_result.tag_configure("miss", foreground="#c62828")
        self.txt_result.pack(pady=(0, 8))
        self.txt_result.config(state="disabled")

        self.lbl_verdict = tk.Label(root, text="", font=("Arial", 14, "bold"), wraplength=620)
        self.lbl_verdict.pack(pady=(4, 0))
        self.lbl_detail = tk.Label(root, text="", font=("Arial", 10), fg="grey", wraplength=620)
        self.lbl_detail.pack(pady=(2, 0))
        self.lbl_latency = tk.Label(root, text="", font=("Arial", 9), fg="grey")
        self.lbl_latency.pack(pady=(2, 8))

        frame = tk.Frame(root)
        frame.pack(pady=8)
        self.btn_rec = tk.Button(frame, text="Mulai merekam", width=16, font=("Arial", 11),
                                 command=self.toggle_record)
        self.btn_rec.grid(row=0, column=0, padx=6)
        self.btn_retry = tk.Button(frame, text="Coba lagi", width=12, font=("Arial", 11),
                                   command=self.retry, state="disabled")
        self.btn_retry.grid(row=0, column=1, padx=6)
        self.btn_next = tk.Button(frame, text="Berikut", width=12, font=("Arial", 11),
                                  command=self.next_lesson, state="disabled")
        self.btn_next.grid(row=0, column=2, padx=6)
        self.btn_speak = tk.Button(frame, text="Dengarkan contoh", width=20, font=("Arial", 11),
                                   command=self.speak_target)
        self.btn_speak.grid(row=1, column=0, columnspan=3, pady=(8, 0))

        self.show_lesson()

    def current(self):
        return LESSONS[self.idx]

    def show_lesson(self):
        target, meaning = self.current()
        self.lbl_progress.config(text=f"Kalimat {self.idx + 1} dari {len(LESSONS)}")
        self.lbl_meaning.config(text=meaning)
        self.lbl_target.config(text=target)
        self.lbl_pron.config(text=PRONUNCIATION.get(target, ""))
        self.reset_attempt()

    def reset_attempt(self):
        self.final_text = ""
        self.chunks = []
        self.lbl_live.config(text="(tekan Mulai merekam lalu bicara)")
        self.set_result_text([])
        self.lbl_verdict.config(text="")
        self.lbl_detail.config(text="")
        self.lbl_latency.config(text="")
        self.btn_rec.config(text="Mulai merekam", state="normal")
        self.btn_retry.config(state="disabled")
        self.btn_next.config(state="disabled")

    def set_result_text(self, pairs):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", "end")
        for k, (word, ok) in enumerate(pairs):
            if k:
                self.txt_result.insert("end", " ")
            self.txt_result.insert("end", word, "ok" if ok else "miss")
        self.txt_result.config(state="disabled")

   
    def toggle_record(self):
        if self.recording:
            self.stop_record()
        else:
            self.start_record()

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_q.put(bytes(indata))

    def start_record(self):
        self.final_text = ""
        self.chunks = []
        self.set_result_text([])
        self.lbl_verdict.config(text="")
        self.lbl_detail.config(text="")
        self.lbl_latency.config(text="")
        self.lbl_live.config(text="Mendengarkan...")
        self.rec = KaldiRecognizer(self.vosk, SR)
        self.rec.SetWords(True)
        self.stream = sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                                        channels=1, callback=self.audio_callback)
        self.stream.start()
        self.recording = True
        self.btn_rec.config(text="Berhenti")
        self.btn_retry.config(state="disabled")
        self.btn_next.config(state="disabled")
        self.root.after(50, self.poll)

    def drain_queue(self):
        while not self.audio_q.empty():
            data = self.audio_q.get_nowait()
            self.chunks.append(np.frombuffer(data, dtype=np.int16))
            if self.rec.AcceptWaveform(data):
                text = json.loads(self.rec.Result()).get("text", "")
                if text:
                    self.final_text = (self.final_text + " " + text).strip()
            else:
                partial = json.loads(self.rec.PartialResult()).get("partial", "")
                shown = (self.final_text + " " + partial).strip()
                self.lbl_live.config(text=shown or "Mendengarkan...")

    def poll(self):
        if not self.recording:
            return
        self.drain_queue()
        self.root.after(50, self.poll)

    def stop_record(self):
        self.recording = False
        t0 = time.time()
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.drain_queue()
        last = json.loads(self.rec.FinalResult()).get("text", "")
        if last:
            self.final_text = (self.final_text + " " + last).strip()
        self.show_result(t0)

    def show_result(self, t0):
        self.btn_speak.config(state="normal")
        target, _ = self.current()
        audio = np.concatenate(self.chunks).astype(np.float64) / 32768.0 if self.chunks else np.zeros(0)

        if len(audio) < MIN_SECONDS * SR:
            self.lbl_live.config(text="Tidak ada suara terdengar")
            self.set_result_text([])
            self.lbl_verdict.config(text="Tidak terdengar, coba lagi", fg="#c62828")
            self.lbl_detail.config(text="")
        else:
            
            a = P.analyze(self.vosk, audio, self.templates)
            tier, score, pairs = P.judge(target, a["free"], a["label"], a["gap"])
            # transkripsi yang ditampilkan = transkripsi BEBAS (norm, tanpa grammar); kata merah = tidak terdengar benar
            self.lbl_live.config(text=f"Terdengar: \"{a['free']}\"" if a["free"] else "Tidak ada kata terdengar")
            self.set_result_text(pairs)
            if tier == "berisik":
                self.lbl_verdict.config(text="Terlalu banyak gangguan. Pindah ke tempat lebih tenang, lalu coba lagi.", fg="#e65100")
            elif tier == "bagus":
                self.lbl_verdict.config(text="Bagus!", fg="#1a8c2b")
                self.btn_next.config(state="normal")
            elif tier == "hampir":
                self.lbl_verdict.config(text="Hampir benar. Ucapkan kata yang merah dengan lebih jelas.", fg="#e65100")
                self.btn_next.config(state="normal" if ALLOW_NEXT_ON_HAMPIR else "disabled")
            else:
                self.lbl_verdict.config(text="Coba lagi", fg="#c62828")
            if a["label"] == OTHER:
                dikenali = "di luar 10 kalimat latihan"
            else:
                dikenali = "sama dengan target" if a["label"] == target else f"mirip \"{a['label']}\""
            gap = f"{a['gap']:.0f} dB" if a["gap"] == a["gap"] else "-"
            self.lbl_detail.config(text=f"Kalimat terdeteksi: {dikenali}   |   skor kata: {score:.0%}   |   ucapan di atas noise: {gap}")

        self.btn_retry.config(state="normal")
        self.btn_rec.config(text="Mulai merekam")
        self.lbl_latency.config(text=f"Waktu pemrosesan setelah berhenti: {(time.time() - t0) * 1000:.0f} ms")

    def retry(self):
        self.reset_attempt()

    def speak_target(self):
        if self.recording:
            return
        target, _ = self.current()
        threading.Thread(target=self._say, args=(target,), daemon=True).start()

    def _say(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)
            for v in engine.getProperty("voices"):
                if "Zira" in v.name or "David" in v.name:
                    engine.setProperty("voice", v.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def next_lesson(self):
        self.idx = (self.idx + 1) % len(LESSONS)
        self.show_lesson()

    def on_close(self):
        if self.recording:
            self.recording = False
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    missing = []
    if not os.path.isdir(MODEL_DIR):
        missing.append(f"Folder '{MODEL_DIR}' (model Vosk)")
    if not os.path.isfile(SCORER_PATH):
        missing.append(f"File '{SCORER_PATH}' (jalankan train_scorer.py dulu)")
    if missing:
        messagebox.showerror("Berkas tidak ditemukan", "\n".join(missing))
        root.destroy()
        sys.exit(1)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
