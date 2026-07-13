"""Voxa benchmark — real numbers for this machine.

Measures, per model, on a clean sample and a noisier version of it:
  - WER (word error rate) vs a known reference
  - realtime factor (audio seconds / wall seconds; higher = faster)
  - peak process RAM

CPU + int8 only here (no CUDA on this box), so it reports RAM, not VRAM.
Self-contained: no jiwer / datasets — a small stdlib WER is plenty for this.

Run:  python benchmark/voxa_benchmark.py
"""

import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxa import Transcriber, formats  # noqa: E402
from voxa.engine import decode_audio  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")
JFK = os.path.join(DATA, "jfk.flac")
JFK_REF = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)
MODELS = ["tiny", "base"]
SR = 16000


def normalize(text):
    keep = "abcdefghijklmnopqrstuvwxyz0123456789 "
    text = "".join(c if c in keep else " " for c in text.lower())
    return text.split()


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    # Levenshtein distance over words
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(1, len(r))


def peak_ram_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return (kb / 1048576) if sys.platform == "darwin" else (kb / 1024)


def add_noise(audio, snr_db=10.0):
    rng = np.random.default_rng(0)
    sig_power = np.mean(audio**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def run(model_size, audio, ref, dur):
    t = Transcriber(model_size, device="cpu", compute_type="int8")
    start = time.perf_counter()
    result = t.transcribe(audio, vad_filter=False)
    elapsed = time.perf_counter() - start
    text = formats.to_txt(result)
    return {
        "wer": wer(ref, text),
        "rtf": dur / elapsed,
        "elapsed": elapsed,
        "ram": peak_ram_mb(),
    }


def main():
    clean = decode_audio(JFK, sampling_rate=SR)
    dur = len(clean) / SR
    conditions = [("clean", clean)]
    for snr in (5.0, 0.0):
        conditions.append((f"noisy@{snr:.0f}dB", add_noise(clean, snr_db=snr)))
    print(f"Sample: jfk.flac  ({dur:.1f}s audio)  CPU + int8\n")
    header = f"{'model':<8}{'sample':<12}{'WER':>8}{'RTF':>10}{'time(s)':>10}{'RAM(MB)':>10}"
    print(header)
    print("-" * len(header))
    for m in MODELS:
        for label, audio in conditions:
            r = run(m, audio, JFK_REF, dur)
            print(f"{m:<8}{label:<12}{r['wer']*100:>7.1f}%{r['rtf']:>9.1f}x"
                  f"{r['elapsed']:>10.2f}{r['ram']:>10.0f}")


if __name__ == "__main__":
    main()
